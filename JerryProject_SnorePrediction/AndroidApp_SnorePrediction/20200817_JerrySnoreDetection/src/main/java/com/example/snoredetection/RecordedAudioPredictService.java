package com.example.snoredetection;

import android.app.Service;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class RecordedAudioPredictService extends Service {
    public static final String TAG = "Jerry RecordedAudioPredictService.java";

    private static String pathSave = "/sdcard/JerrySnorePrediction"; // Must be identical to AudioRecordService.pathSave

    private static final String SNORE_PREDICTION_MODEL_PATH = "snore_predict.tflite";
    private static final String VGG_EXTRACT_MODEL_PATH = "vggish_feature_extraction_model.tflite";

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        try {
            audioToPrediction(pathSave + ".pcm");
        } catch (IOException e) {
            e.printStackTrace();
        }
        return START_STICKY;
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    public void audioToPrediction(String wav) throws IOException {
        VGGInput vggInput = new VGGInput();
        File f = new File(wav);
        float[][][] examples_batch = vggInput.pcm_to_examples(f);
        float[] multi_pred = examplesToPrediction(examples_batch, 1);

        // Broadcast result to MainActivity
        Intent intent = new Intent("File Predict");
        intent.putExtra("Value", Arrays.toString(multi_pred));
        sendBroadcast(intent);

        stopSelf();
    }

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
