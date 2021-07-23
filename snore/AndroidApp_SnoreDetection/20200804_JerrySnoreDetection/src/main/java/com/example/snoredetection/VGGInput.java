package com.example.snoredetection;

import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

public class VGGInput {
    //  NOTE: currently, we only accept and perform operations on audio data with sample rate 16KHz in mono channel.

    private static final String TAG = "Jerry VggishInput.java";

    private static final int SAMPLE_RATE = 16000;
    private static final float LOG_OFFSET = 0.01f;  // Offset used for stabilized log of input mel-spectrogram.
    private static final float STFT_WINDOW_LENGTH_SECONDS = 0.025f;
    private static final float STFT_HOP_LENGTH_SECONDS = 0.010f;
    private static final int NUM_MEL_BINS = 64;
    private static final float MEL_MIN_HZ = 125.0f;
    private static final float MEL_MAX_HZ = 7500.0f;
    private static final float EXAMPLE_WINDOW_SECONDS = 0.96f;  // Each example contains 96 10ms frames
    private static final float EXAMPLE_HOP_SECONDS = 0.96f;     // with zero overlap.

    public static float[][][] pcm_to_examples(File f) throws IOException {
        /* Replace function: vggish_input.py wavfile_to_examples(str wav_path)
        Now, instead of reading a WAV file, we directly read raw data from PCM

        Convenience wrapper around waveform_to_examples() for a common WAV format.

        Args:
        wav_file: String path to a file, or a file-like object. The file
        is assumed to contain WAV audio data with signed 16-bit PCM samples.

        Returns:
        See waveform_to_examples.
         */

        float ratio2sample = 32768.0f;
        short[] wav_data = pcm_read(f);
        float[] samples = new float[wav_data.length];
        for (int i=0; i<wav_data.length; i++){
            samples[i] = wav_data[i] / ratio2sample;
        }

        return waveform_to_examples(samples);


    }

    public static float[][][] waveform_to_examples(float[] data){
        /* Replace function: vggish_input.py waveform_to_examples(float[] data, int sample_rate)
        Since we restrict input audio to be 16kHz, we omit resample process.

        Converts audio waveform into an array of examples for VGGish.

                Args:
        data: np.array of either one dimension (mono) or two dimensions
                (multi-channel, with the outer dimension representing channels).
                Each sample is generally expected to lie in the range [-1.0, +1.0],
        although this is not required.
                sample_rate: Sample rate of data.

        Returns:
        3-D np.array of shape [num_examples, num_frames, num_bands] which represents
        a sequence of examples, each of which contains a patch of log mel
        spectrogram, covering num_frames frames of audio and num_bands mel frequency
        bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
         */
        MelFeatures mf = new MelFeatures();
        float[][] log_mel = mf.log_mel_spectrogram(data, SAMPLE_RATE, LOG_OFFSET, STFT_WINDOW_LENGTH_SECONDS,
                STFT_HOP_LENGTH_SECONDS, NUM_MEL_BINS, MEL_MIN_HZ, MEL_MAX_HZ);

        float features_sample_rate = 1.0f / STFT_HOP_LENGTH_SECONDS;
        int example_window_length = Math.round(EXAMPLE_WINDOW_SECONDS * features_sample_rate);
        int example_hop_length = Math.round(EXAMPLE_HOP_SECONDS * features_sample_rate);

        float[][][] log_mel_examples = mf.frame(log_mel, example_window_length, example_hop_length);

        return log_mel_examples;
    }


    public static short[] pcm_read(File f) throws IOException{
        // Replace function: vggish_input.py wav_read(str wav_path)
        // Now, instead of reading a WAV file, we directly read raw data from PCM
        int size = (int) f.length();
        byte bytes[] = new byte[size];
        byte tmpBuff[] = new byte[size];
        FileInputStream fis= new FileInputStream(f);
        try {

            int read = fis.read(bytes, 0, size);
            if (read < size) {
                int remain = size - read;
                while (remain > 0) {
                    read = fis.read(tmpBuff, 0, remain);
                    System.arraycopy(tmpBuff, 0, bytes, size - remain, read);
                    remain -= read;
                }
            }
        }  catch (IOException e){
            throw e;
        } finally {
            fis.close();
        }
        short[] res = byte2short(bytes);
        return res;
    }

    public static short[] byte2short(byte buf[]) {
        short shorts[] = new short[buf.length / 2];
        for(int i = 0; i < shorts.length; i++) {
            shorts[i] = (short) ((buf[i*2] & 0xFF) | ((buf[i*2 + 1] & 0xFF) << 8));
        }
        return shorts;
    }

}
