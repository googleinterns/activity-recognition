package com.example.snoredetection;

import java.util.Arrays;

public class RFFT extends FFT {

    public static Complex[] rfft(float[] r, int N) {
//        N: Number of points along transformation axis in the input to use.
//        If n is smaller than the length of the input, the input is cropped.
//        If it is larger, the input is padded with zeros.
//        If n is not given, the length of the input along the axis specified by axis is used.
        if (N == -1){
            N = r.length;
        }
        Complex[] x = toComplex(r, N);
        Complex[] y = fft(x);
        int return_length = N / 2 + 1;
        Complex[] res = Arrays.copyOfRange(y, 0, return_length);
        return res;
    }

    public static Complex[] toComplex(float[] r, int N){
        int rlen = r.length;
        Complex[] x = new Complex[N];
        for (int i = 0; i < N; i++) {
            if (i < rlen)
                x[i] = new Complex(r[i], 0);
            else
                x[i] = new Complex(0, 0);
        }
        return x;
    }

    public static float[] rfftAbs(Complex[] c){
        float[] res = new float[c.length];
        for (int i=0; i<c.length; i++){
            res[i] = (float) c[i].abs();
        }
        return res;
    }
}