package com.example.snoredetection;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Jerry MainActivity.java";

    private static final String SNORE_PREDICTION_MODEL_PATH = "snore_predict.tflite";
    private static final String VGG_EXTRACT_MODEL_PATH = "vggish_feature_extraction_model.tflite";

    TextView textView;
    Button btnStart, btnStop, btnPredict;
    String pathSave = "/sdcard/JerrySnorePrediction";

    final int REQUEST_PERMISSION_CODE = 1000;


    // AudioRecorder variables
    private static final int RECORDER_SAMPLERATE = 16000;
    private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
    private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord recorder = null;
    private Thread recordingThread = null;
    private boolean isRecording = false;
    int BufferElements2Rec = 1024; // want to play 2048 (2K) since 2 bytes we use only 1024
    int BytesPerElement = 2; // 2 bytes in 16bit format

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "onCreate() start");

        // Init View
        btnStart = (Button)findViewById(R.id.btnStart);
        btnStop = (Button)findViewById(R.id.btnStop);
        btnPredict = (Button)findViewById(R.id.btnPredict);

        enableButtons(false);

        int bufferSize = AudioRecord.getMinBufferSize(RECORDER_SAMPLERATE,
                RECORDER_CHANNELS, RECORDER_AUDIO_ENCODING);

        if (checkPermissionFromDevice()){
            btnStart.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableButtons(true);
                    startRecording();
                }
            });

            btnStop.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableButtons(false);
                    stopRecording();
                }
            });

            btnPredict.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    // Generate WAV from PCM first
                    File f1 = new File(pathSave+".pcm"); // The location of your PCM file
                    File f2 = new File(pathSave+".wav"); // The location where you want your WAV file
                    try {
                        PCM2WAV.rawToWave(f1, f2);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
//                        audioToPrediction(pathSave+".wav");
                        audioToPrediction("/sdcard/test.wav");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

        }else{
            requestPermission();
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void audioToPrediction(String wav) throws IOException {
        Log.d(TAG, "AudioToPrediction() start");
        VGGInput vggInput = new VGGInput();
        File f = new File(pathSave+".pcm");
        float[][][] examples_batch = vggInput.pcm_to_examples(f);

        float[][] embedding_batch = new float[examples_batch.length][128];
        System.out.println("Jerry MainActivity.java: before loading tflite");
        try (Interpreter tflite_vgg_extraction = new Interpreter(loadModelFile(getAssets(), VGG_EXTRACT_MODEL_PATH))) {
            for (int i = 0; i < examples_batch.length; i++){
                float[][][] input = {examples_batch[i]};
                float[][] output = new float[1][128];
                tflite_vgg_extraction.run(input, output);
                embedding_batch[i] = output[0];
            }
            tflite_vgg_extraction.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        VGGPostproc vggPostproc = new VGGPostproc();
        float[][] postprocessed_batch = vggPostproc.postprocess(this, embedding_batch);
        System.out.println("Jerry MainActivity.java: finished extracting vgg features");

        // Finished extracting VGGish feature
        // Feed it into pre-trained snore-detection model, get prediction values
        float[] prediction = new float[postprocessed_batch.length];
        try (Interpreter tflite_snore_prediction = new Interpreter(loadModelFile(getAssets(), SNORE_PREDICTION_MODEL_PATH))) {
            System.out.println("Jerry MainActivity.java: after loading prediction tflite");
            for (int i = 0; i < postprocessed_batch.length; i++){
                float[][] input = {postprocessed_batch[i]};
                float[][] output = new float[1][2];
                tflite_snore_prediction.run(input, output);
                prediction[i] = output[0][1];
            }
            tflite_snore_prediction.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println(Arrays.toString(prediction));
        // Turn 1-sec based prediction into multi-sec based prediction
        float[] multi_pred = frameMultiSeconds(prediction, 2);
        System.out.println(Arrays.toString(multi_pred));
        System.out.println("Jerry MainActivity.java: audioToPrediction()");
        textView = findViewById(R.id.predictText);
        textView.setText(Arrays.toString(multi_pred));

    }

    public float[] frameMultiSeconds(float[] pred_single_second, int frameN){
        float[] res = new float[pred_single_second.length - frameN + 1];
        if (frameN >= pred_single_second.length) {
            return new float[]{sum(pred_single_second) / pred_single_second.length};
        }else{
            for (int i=0; i<res.length; i++){
                float[] tmp = Arrays.copyOfRange(pred_single_second, i, i+frameN);
                res[i] = Math.round((sum(tmp) / tmp.length) *1000)/1000.0f;
            }
        }
        return res;
    }

    public float sum(float[] array) {
        float sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    public MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // Load tflite helper function
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }






    // Get permission related functions
    private boolean checkPermissionFromDevice() {
        int write_external_storage_result = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int read_external_storage_result = ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE);
        int record_audio_result = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO);
        return write_external_storage_result == PackageManager.PERMISSION_GRANTED &&
                record_audio_result == PackageManager.PERMISSION_GRANTED &&
                read_external_storage_result == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission() {
        ActivityCompat.requestPermissions(this, new String[]{
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.RECORD_AUDIO
        }, REQUEST_PERMISSION_CODE);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode){
            case REQUEST_PERMISSION_CODE:{
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
                    Toast.makeText(this, "Permission Granted. Please reopen app to continue.", Toast.LENGTH_SHORT).show();
                else
                    Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
            }
        }
    }





    // Recording audios related functions
    private byte[] short2byte(short[] sData) {
        int shortArrsize = sData.length;
        byte[] bytes = new byte[shortArrsize * 2];
        for (int i = 0; i < shortArrsize; i++) {
            bytes[i * 2] = (byte) (sData[i] & 0x00FF);
            bytes[(i * 2) + 1] = (byte) (sData[i] >> 8);
            sData[i] = 0;
        }
        return bytes;
    }
    private void startRecording() {
        recorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                RECORDER_SAMPLERATE, RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING, BufferElements2Rec * BytesPerElement);

        recorder.startRecording();
        isRecording = true;
        recordingThread = new Thread(new Runnable() {
            public void run() {
                writeAudioDataToFile();
            }
        }, "AudioRecorder Thread");
        recordingThread.start();
    }
    private void writeAudioDataToFile() {
        // Write the output audio in byte

        String pcmPath = pathSave + ".pcm";
        short sData[] = new short[BufferElements2Rec];

        FileOutputStream os = null;
        try {
            os = new FileOutputStream(pcmPath);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        while (isRecording) {
            // gets the voice output from microphone to byte format

            recorder.read(sData, 0, BufferElements2Rec);
            System.out.println("Short wirting to file" + sData.toString());
            try {
                // // writes the data to file from buffer
                // // stores the voice buffer
                byte bData[] = short2byte(sData);
                os.write(bData, 0, BufferElements2Rec * BytesPerElement);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            os.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void stopRecording() {
        // stops the recording activity
        if (null != recorder) {
            isRecording = false;
            recorder.stop();
            recorder.release();
            recorder = null;
            recordingThread = null;
        }
    }
    private void enableButton(int id, boolean isEnable) {
        ((Button) findViewById(id)).setEnabled(isEnable);
    }
    private void enableButtons(boolean isRecording) {
        enableButton(R.id.btnStart, !isRecording);
        enableButton(R.id.btnStop, isRecording);
    }









}