package com.example.snoredetection;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Jerry MainActivity.java";

    private static final String SNORE_PREDICTION_MODEL_PATH = "snore_predict.tflite";
    private static final String VGG_EXTRACT_MODEL_PATH = "vggish_feature_extraction_model.tflite";

    TextView filePredictText, livePredictText;
    Button btnStart, btnStop, btnPredict, btnLiveStart, btnLiveStop;
    String pathSave = "/sdcard/JerrySnorePrediction"; // Must be identical to AudioRecordService.pathSave

    final int REQUEST_PERMISSION_CODE = 1000;


    //LIST OF ARRAY STRINGS WHICH WILL SERVE AS LIST ITEMS
    ArrayList<String> listItems=new ArrayList<String>();
    int liveCount = 0;
    //DEFINING A STRING ADAPTER WHICH WILL HANDLE THE DATA OF THE LISTVIEW
    ArrayAdapter<String> adapter;

    private BroadcastReceiver broadcastReceiver = new BroadcastReceiver(){
        @Override
        public void onReceive(Context context, Intent intent) {
            switch(intent.getAction()){
                case "File Predict":
                    filePredictText.setText(intent.getStringExtra("Value"));
                    break;
                case "Live Predict":
//                    livePredictText.setText(intent.getStringExtra("Value"));
                    String value = intent.getStringExtra("Value");
                    String ex = "Snore";
                    if (Float.parseFloat(value) <= 0.5){
                        ex = "Non-snore";
                    }
                    listItems.add("At " + liveCount + " second: " + value + "  " + ex);
                    liveCount += 1;
                    Log.d(TAG, String.valueOf(listItems));
                    adapter.notifyDataSetChanged();
                    break;
            }
        }
    };

    @Override
    protected void onStart() {
        super.onStart();
        IntentFilter filter = new IntentFilter();
        filter.addAction("File Predict");
        filter.addAction("Live Predict");
        registerReceiver(broadcastReceiver, filter);
    }

    @Override
    protected void onStop() {
        super.onStop();
        unregisterReceiver(broadcastReceiver);
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d(TAG, "onCreate() start");

        adapter=new ArrayAdapter<String>(this, android.R.layout.simple_list_item_1, listItems);
        ((ListView) findViewById(R.id.list)).setAdapter(adapter);

        // Init View
        btnStart = (Button)findViewById(R.id.btnStart);
        btnStop = (Button)findViewById(R.id.btnStop);
        btnPredict = (Button)findViewById(R.id.btnPredict);
        btnLiveStart = (Button)findViewById(R.id.btnLiveStart);
        btnLiveStop = (Button)findViewById(R.id.btnLiveStop);
        filePredictText = findViewById(R.id.predictText);
        livePredictText = findViewById(R.id.livePredictText);

        enableRecordButtons(false);
        enableLiveButtons(false);


        if (checkPermissionFromDevice()){
            btnStart.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableRecordButtons(true);
                    startService(new Intent(view.getContext(), AudioRecordService.class));
                }
            });

            btnStop.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableRecordButtons(false);
                    stopService(new Intent(view.getContext(), AudioRecordService.class));
                }
            });

            btnPredict.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    startService(new Intent(view.getContext(), RecordedAudioPredictService.class));
                }
            });

            btnLiveStart.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableLiveButtons(true);
                    startService(new Intent(view.getContext(), LivePredictionService.class));
                    listItems.clear();
                    adapter.notifyDataSetChanged();
                    liveCount = 0;
                }
            });

            btnLiveStop.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    enableLiveButtons(false);
                    stopService(new Intent(view.getContext(), LivePredictionService.class));
                }
            });


        }else{
            requestPermission();
        }

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



    private void enableRecordButton(int id, boolean isEnable) {
        ((Button) findViewById(id)).setEnabled(isEnable);
    }
    private void enableRecordButtons(boolean isRecording) {
        enableRecordButton(R.id.btnStart, !isRecording);
        enableRecordButton(R.id.btnStop, isRecording);
    }

    private void enableLiveButton(int id, boolean isEnable) {
        ((Button) findViewById(id)).setEnabled(isEnable);
    }
    private void enableLiveButtons(boolean isLivePredicting) {
        enableLiveButton(R.id.btnLiveStart, !isLivePredicting);
        enableLiveButton(R.id.btnLiveStop, isLivePredicting);
    }


}