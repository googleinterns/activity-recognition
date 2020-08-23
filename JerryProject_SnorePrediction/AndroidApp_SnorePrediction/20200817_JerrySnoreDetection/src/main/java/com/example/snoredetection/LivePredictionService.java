package com.example.snoredetection;

import android.annotation.SuppressLint;
import android.app.Service;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Message;
import android.util.Log;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;

public class LivePredictionService extends Service {
    public static final String TAG = "Jerry LivePredictionService.java";

    private static final String SNORE_PREDICTION_MODEL_PATH = "snore_predict.tflite";
    private static final String VGG_EXTRACT_MODEL_PATH = "vggish_feature_extraction_model.tflite";

    // hashset, hashtable[time]
    private HashMap<Integer, Float> multi_pred = new HashMap<Integer, Float>();

    // AudioRecorder variables
    private static final int RECORDER_SAMPLERATE = 16000;
    private static final int RECORDER_CHANNELS = AudioFormat.CHANNEL_IN_MONO;
    private static final int RECORDER_AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord liveRecorder = null;
    private Thread livePredictingThread = null;
    private boolean isLivePredicting = false;
    int BufferElements2Rec = 1024; // want to play 2048 (2K) since 2 bytes we use only 1024
    int BytesPerElement = 2; // 2 bytes in 16bit format

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }


    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {

        liveRecorder = new AudioRecord(MediaRecorder.AudioSource.MIC,
                RECORDER_SAMPLERATE, RECORDER_CHANNELS,
                RECORDER_AUDIO_ENCODING, BufferElements2Rec * BytesPerElement);

        liveRecorder.startRecording();
        isLivePredicting = true;

        livePredictingThread = new Thread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.KITKAT)
            public void run() {
                try {
                    livePredict();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }, "LivePredict Thread");
        livePredictingThread.start();
        return START_STICKY;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        // stops the recording activity
        if (null != liveRecorder) {
            isLivePredicting = false;
            liveRecorder.stop();
            liveRecorder.release();
            liveRecorder = null;
            livePredictingThread = null;
        }
    }



    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    private void livePredict() throws IOException {
        // Predict snoring on live
        float ratio2sample = 32768.0f;

        short sData[] = new short[16000];
        float[] singleSec = new float[16000];


        multi_pred = new HashMap<Integer, Float>();
        int count = 0;

        while (isLivePredicting) {
            // gets the voice output from microphone to byte format

            // This line takes 1 second
            liveRecorder.read(sData, 0, 16000);

            for (int i=0; i<sData.length; i++){
                singleSec[i] = sData[i] / ratio2sample;
            }
            Thread tmp = new Thread(new MyRunnable(singleSec.clone(), count));
            tmp.start();
            count += 1;
        }
    }


    public class MyRunnable implements Runnable {
        private float[] single_sec;
        private int count;
        public MyRunnable(float[] single_sec, int count) {
            this.single_sec = single_sec;
            this.count = count;
        }

        @RequiresApi(api = Build.VERSION_CODES.KITKAT)
        public void run() {
            float[][][] examples_batch = VGGInput.waveform_to_examples(this.single_sec);
            float[] pred = new float[0];
            try {
                pred = examplesToPrediction(examples_batch, 1);
            } catch (IOException e) {
                e.printStackTrace();
            }
            multi_pred.put(count, pred[0]);

            float[] multi_pred_array = new float[multi_pred.size()];
            for (int i=0; i<multi_pred_array.length; i++){
                multi_pred_array[i] = multi_pred.get(i);
            }
            // Log.d(TAG, Arrays.toString(multi_pred_array));

            // Broadcast result to MainActivity
            Intent intent = new Intent("Live Predict");
            intent.putExtra("Value", Arrays.toString(multi_pred_array));
            sendBroadcast(intent);
        }
    }




    // Functions for predicting snoring
    //
    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public float[] examplesToPrediction(float[][][] examples_batch, int frameN) throws IOException {
        float[][] embedding_batch = new float[examples_batch.length][128];
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

        // Finished extracting VGGish feature
        // Feed it into pre-trained snore-detection model, get prediction values
        float[] prediction = new float[postprocessed_batch.length];
        try (Interpreter tflite_snore_prediction = new Interpreter(loadModelFile(getAssets(), SNORE_PREDICTION_MODEL_PATH))) {
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
        // Turn 1-sec based prediction into multi-sec based prediction
        float[] multi_pred = frameMultiSeconds(prediction, frameN);
        return multi_pred;
    }

    public float[] frameMultiSeconds(float[] pred_single_second, int frameN){
        float[] res = new float[pred_single_second.length - frameN + 1];
        if (frameN >= pred_single_second.length) {
            return new float[]{Math.round((sum(pred_single_second) / pred_single_second.length)*1000)/1000.0f};
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
//        fileDescriptor.close();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
