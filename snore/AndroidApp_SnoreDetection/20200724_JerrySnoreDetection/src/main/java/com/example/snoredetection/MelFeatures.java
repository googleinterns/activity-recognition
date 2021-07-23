package com.example.snoredetection;

import android.util.Log;
import java.util.Arrays;

public class MelFeatures {
    private static final String TAG = "Jerry MelFeatures.java";

    public static float[][] log_mel_spectrogram(float[] data, int audio_sample_rate,
                                           float log_offset, float window_length_secs,
                                           float hop_length_secs, int num_mel_bins,
                                           float mel_min_hz, float mel_max_hz){
        /* Replace function: mel_features.py log_mel_spectrogram(......)
        Convert waveform to a log magnitude mel-frequency spectrogram.

                Args:
        data: 1D np.array of waveform data.
        audio_sample_rate: The sampling rate of data.
                log_offset: Add this to values when taking log to avoid -Infs.
                window_length_secs: Duration of each window to analyze.
        hop_length_secs: Advance between successive analysis windows.
                **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.

        Returns:
        2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
        magnitudes for successive frames.
         */
        int window_length_samples = Math.round(audio_sample_rate * window_length_secs);
        int hop_length_samples = Math.round(audio_sample_rate * hop_length_secs);
        int fft_length = (int) Math.pow(2, Math.ceil(Math.log(window_length_samples) / Math.log(2.0)));

        float[][] spectrogram = stft_magnitude(
                data,
                fft_length,
                hop_length_samples,
                window_length_samples);
        float[][] mel_weights_matrix = spectrogram_to_mel_matrix(num_mel_bins, spectrogram[0].length, audio_sample_rate, mel_min_hz, mel_max_hz);

        float [][] mel_spectrogram = new float[spectrogram.length][mel_weights_matrix[0].length];

        // Implementation of np.dot(spectrogram, mel_weights_matrix)
        for (int i = 0; i < spectrogram.length; i++) {
            for (int j = 0; j < mel_weights_matrix[0].length; j++) {
                for (int k = 0; k < spectrogram[0].length; k++) {
                    mel_spectrogram[i][j] += spectrogram[i][k] * mel_weights_matrix[k][j];
                }
            }
        }

        float[][] log_mel = new float[mel_spectrogram.length][mel_spectrogram[0].length];

        for (int i=0; i<mel_spectrogram.length; i++){
            for (int j=0; j<mel_spectrogram[0].length; j++){
                log_mel[i][j] = (float) Math.log(mel_spectrogram[i][j] + log_offset);
            }
        }
        return log_mel;
    }

    public static float[][] spectrogram_to_mel_matrix(int num_mel_bins,
                                                 int num_spectrogram_bins,
                                                 int audio_sample_rate,
                                                 float lower_edge_hertz,
                                                 float upper_edge_hertz){
        /* Replace function: mel_features.py spectrogram_to_mel_matrix(......)
        Return a matrix that can post-multiply spectrogram rows to make mel.

                Returns a np.array matrix A that can be used to post-multiply a matrix S of
        spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
        "mel spectrogram" M of frames x num_mel_bins.  M = S A.

                The classic HTK algorithm exploits the complementarity of adjacent mel bands
        to multiply each FFT bin by only one mel weight, then add it, with positive
        and negative signs, to the two adjacent mel bands to which that bin
        contributes.  Here, by expressing this operation as a matrix multiply, we go
        from num_fft multiplies per frame (plus around 2*num_fft adds) to around
        num_fft^2 multiplies and adds.  However, because these are all presumably
        accomplished in a single call to np.dot(), it's not clear which approach is
        faster in Python.  The matrix multiplication has the attraction of being more
        general and flexible, and much easier to read.

        Args:
        num_mel_bins: How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
        num_spectrogram_bins: How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
        audio_sample_rate: Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
                lower_edge_hertz: Lower bound on the frequencies to be included in the mel
        spectrum.  This corresponds to the lower edge of the lowest triangular
        band.
                upper_edge_hertz: The desired top edge of the highest frequency band.

                Returns:
        An np.array with shape (num_spectrogram_bins, num_mel_bins).

                Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
         */

        int nyquist_hertz = audio_sample_rate / 2;
        float[] spectrogram_bins_hertz = linspace(0.0f, nyquist_hertz, num_spectrogram_bins);
        float[] spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz);

        float[] band_edges_mel = linspace(hertz_to_mel(lower_edge_hertz), hertz_to_mel(upper_edge_hertz), num_mel_bins + 2);

        float[][] mel_weights_matrix = new float[num_spectrogram_bins][num_mel_bins];
        for (int i=0; i < num_mel_bins; i++){
            float lower_edge_mel = band_edges_mel[i];
            float center_mel = band_edges_mel[i+1];
            float upper_edge_mel = band_edges_mel[i+2];
            for (int j=0; j<spectrogram_bins_mel.length; j++){
                float lower_slope = ((spectrogram_bins_mel[j] - lower_edge_mel) / (center_mel - lower_edge_mel));
                float upper_slope = ((upper_edge_mel - spectrogram_bins_mel[j]) / (upper_edge_mel - center_mel));
                if (j == 0)
                    mel_weights_matrix[j][i] = 0.0f;
                else
                    mel_weights_matrix[j][i] = (float) Math.max(0.0, Math.min(lower_slope, upper_slope));
            }
        }
        return mel_weights_matrix;
    }


    public static float hertz_to_mel(float frequencies_hertz){
        /* Replace function: mel_features.py hertz_to_mel(......)
        Convert frequencies to mel scale using HTK formula.

        Args:
        frequencies_hertz: Scalar or np.array of frequencies in hertz.

                Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
         */
        float MEL_BREAK_FREQUENCY_HERTZ = 700.0f;
        float MEL_HIGH_FREQUENCY_Q = 1127.0f;
        return (float) (MEL_HIGH_FREQUENCY_Q * Math.log(1.0 + (frequencies_hertz / MEL_BREAK_FREQUENCY_HERTZ)));
    }

    public static float[] hertz_to_mel(float[] frequencies_hertz){
        float MEL_BREAK_FREQUENCY_HERTZ = 700.0f;
        float MEL_HIGH_FREQUENCY_Q = 1127.0f;
        float[] res = new float[frequencies_hertz.length];
        for (int i=0; i<frequencies_hertz.length; i++){
            res[i] = (float) (MEL_HIGH_FREQUENCY_Q * Math.log(1.0 + (frequencies_hertz[i] / MEL_BREAK_FREQUENCY_HERTZ)));
        }
        return res;
    }



    public static float[][] stft_magnitude(float[] signal, int fft_length,
                                      int hop_length, int window_length){
        /* Replace function: mel_features.py stft_magnitude(......)
        Calculate the short-time Fourier transform magnitude.

                Args:
        signal: 1D np.array of the input time-domain signal.
                fft_length: Size of the FFT to apply.
        hop_length: Advance (in samples) between each frame passed to FFT.
        window_length: Length of each block of samples to pass to FFT.

        Returns:
        2D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
         */

        // return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

        float[][] frames = frame(signal, window_length, hop_length);
        float[] window = periodic_hann(window_length);
        float[][] windowed_frames = new float[frames.length][window.length];
        for (int i=0; i<frames.length; i++){
            for (int j=0; j<window.length; j++){
                windowed_frames[i][j] = frames[i][j] * window[j];
            }
        }

        float[][] res = new float[windowed_frames.length][windowed_frames[0].length];
        for (int i=0; i<windowed_frames.length; i++){
            res[i] = RFFT.rfftAbs(RFFT.rfft(windowed_frames[i], fft_length));
        }
        return res;
    }

    public static float[] periodic_hann(int window_length){
        /* Replace function: mel_features.py periodic_hann(......)
        Calculate a "periodic" Hann window.

        The classic Hann window is defined as a raised cosine that starts and
        ends on zero, and where every value appears twice, except the middle
        point for an odd-length window.  Matlab calls this a "symmetric" window
        and np.hanning() returns it.  However, for Fourier analysis, this
        actually represents just over one cycle of a period N-1 cosine, and
        thus is not compactly expressed on a length-N Fourier basis.  Instead,
                it's better to use a raised cosine that ends just before the final
        zero value - i.e. a complete cycle of a period-N cosine.  Matlab
        calls this a "periodic" window. This routine calculates it.

        Args:
        window_length: The number of points in the returned window.

        Returns:
        A 1D np.array containing the periodic hann window
         */

        float[] arange_window_length = new float[window_length];
        for (int i=0; i<window_length; i++){
            arange_window_length[i] = (float) (0.5 - (0.5 * Math.cos(2 * Math.PI / window_length * i)));
        }
        return arange_window_length;
    }

    public static float[][] frame(float[] data, int window_length, int hop_length){
        /* Replace function: mel_features.py frame(......)
        Convert array into a sequence of successive possibly overlapping frames.

        An n-dimensional array of shape (num_samples, ...) is converted into an
        (n+1)-D array of shape (num_frames, window_length, ...), where each frame
        starts hop_length points after the preceding one.

                This is accomplished using stride_tricks, so the original data is not
        copied.  However, there is no zero-padding, so any incomplete frames at the
        end are not included.

        Args:
        data: np.array of dimension N >= 1.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.

                Returns:
        (N+1)-D np.array with as many rows as there are complete frames that can be
        extracted.Convert array into a sequence of successive possibly overlapping frames.

        An n-dimensional array of shape (num_samples, ...) is converted into an
        (n+1)-D array of shape (num_frames, window_length, ...), where each frame
        starts hop_length points after the preceding one.

                This is accomplished using stride_tricks, so the original data is not
        copied.  However, there is no zero-padding, so any incomplete frames at the
        end are not included.

        Args:
        data: np.array of dimension N >= 1.
        window_length: Number of samples in each frame.
        hop_length: Advance (in samples) between each window.

                Returns:
        (N+1)-D np.array with as many rows as there are complete frames that can be
        extracted.
         */
        // For case testAndroid.pcm, return np.lib.stride_tricks.as_strided(data, shape=(900,400), strides=(1280, 8))
        // explanation: shape depends on preset parameters, stides[1] == 8 is the data type size in python
        // in this case, float types take size 8 to store
        // strides[0] / strides[1] == hop_length == 160, that is if reformed[row][col] is data[n],
        // then reformed[row+1][col] would be data[n+160]
        // Return value should be shape[0] (900) arrays each containing shape[1] (400) subarrays

        int num_samples = data.length;
        int num_frames = (int) (1 + Math.floor((num_samples - window_length) / hop_length));
        int[] shape = {num_frames, window_length};
        float[][] reformed = new float[num_frames][window_length];
        int hop_count = 0;
        for (int i=0; i<num_frames; i++){
            for (int j=0; j<window_length; j++){
                reformed[i][j] = data[hop_count * hop_length + j];
            }
            hop_count++;
        }
        return reformed;
    }

    public static float[][][] frame(float[][] data, int window_length, int hop_length){
        int num_samples = data.length;
        int num_frames = (int) (1 + Math.floor((num_samples - window_length) / hop_length));
        int[] shape = {num_frames, window_length, data[0].length};
        float[][][] reformed = new float[num_frames][window_length][data[0].length];
        int hop_count = 0;
        for (int i=0; i<num_frames; i++){
            for (int j=0; j<window_length; j++){
                reformed[i][j] = data[hop_count * hop_length + j];
            }
            hop_count++;
        }
        return reformed;
    }

    public static float[] linspace(float start, float end, int step){
        // Replace function: numpy.linspace(......)
        float hop = (end - start) / (step - 1);
        float[] res = new float[step];
        for (int i=0; i<res.length; i++){
            res[i] = start + i*hop;
        }
        return res;
    }

}
