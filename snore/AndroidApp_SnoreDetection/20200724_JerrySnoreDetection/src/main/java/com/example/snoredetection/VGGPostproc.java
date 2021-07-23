package com.example.snoredetection;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Scanner;

public class VGGPostproc {
    private static final String TAG = "Jerry VGGPostproc.java";

    private static final String pca_params_path = "pca_params.txt";
    private static final float QUANTIZE_MIN_VAL = -2.0f;
    private static final float QUANTIZE_MAX_VAL = 2.0f;

    public static float[][] postprocess(Context c, float[][] embedding_batch) throws IOException {
        /*Applies postprocessing to a batch of embeddings.
          Args:
            embeddings_batch: An nparray of shape [batch_size, embedding_size]
              containing output from the embedding layer of VGGish.
          Returns:
            An nparray of the same shape as the input but of type uint8,
            containing the PCA-transformed and quantized version of the input.*/
        float[][] pca_matrix = get_PCA_params(c);
        float[][] pca_means = get_PCA_means();

        float[][] embedding_T = transpose(embedding_batch);
        float[][] embedding_norm = new float[embedding_T.length][embedding_T[0].length];
        for (int i=0; i<embedding_norm.length; i++){
            for (int j=0; j<embedding_norm[0].length; j++) {
                embedding_norm[i][j] = embedding_T[i][j] - pca_means[i][0];
            }
        }
        float[][] pca_applied = transpose(dotProduct(pca_matrix, embedding_norm));

        // Quantize by: clipping to [min, max] range
        float[][] clipped_embeddings = clip(pca_applied, QUANTIZE_MIN_VAL, QUANTIZE_MAX_VAL);
        // - convert to 8-bit in range [0.0, 255.0]
        float[][] quantized_embeddings = new float[clipped_embeddings.length][clipped_embeddings[0].length];
        for (int i=0; i<clipped_embeddings.length; i++){
            for (int j=0; j<clipped_embeddings[0].length; j++){
                quantized_embeddings[i][j] = (clipped_embeddings[i][j] - QUANTIZE_MIN_VAL) * (255.0f / (QUANTIZE_MAX_VAL - QUANTIZE_MIN_VAL));
            }
        }
        return quantized_embeddings;
    }

    static float[][] transpose(float[][] A)
    {
        float[][] T = new float[A[0].length][A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                T[j][i] = A[i][j];
        return T;
    }

    static float[][] dotProduct(float[][] A, float[][] B){
        float[][] res = new float[A.length][B[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                for (int k = 0; k < A[0].length; k++) {
                    res[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return res;
    }

    static float[][] clip(float[][] data, float min, float max){
        float[][] res = new float[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                float tmp = data[i][j];
                if (tmp > max)
                    tmp = max;
                else if (tmp < min)
                    tmp = min;
                res[i][j] = tmp;
            }
        }
        return res;
    }

//    pca_applied = np.dot(pca_matrix,
//            (embeddings_batch.T - pca_means)).T
//
//  # Quantize by:
//            # - clipping to [min, max] range
//            clipped_embeddings = np.clip(
//            pca_applied, vggish_params.QUANTIZE_MIN_VAL,
//            vggish_params.QUANTIZE_MAX_VAL)
//  # - convert to 8-bit in range [0.0, 255.0]
//            # - cast 8-bit float to uint8
//            quantized_embeddings = quantized_embeddings.astype(np.uint8)
//    quantized_embeddings = np.asarray(quantized_embeddings, order='C')
//
//
//            return quantized_embeddings


    private static float[][] get_PCA_means(){
        float[] pca_means = new float[]{-0.414f,0.025f,0.127f,-0.638f,-0.191f,-0.703f,-0.196f,-0.329f,-0.593f,-0.454f,-0.258f,-0.435f,-0.677f,-0.245f,-0.179f,0.071f,0.122f,0.142f,-0.299f,-0.1f,-0.026f,-0.156f,-0.062f,-0.565f,0.139f,0.113f,0.118f,0.009f,-0.166f,-0.167f,-0.184f,0.016f,-0.076f,0.088f,0.149f,-0.26f,-0.287f,-0.538f,-0.434f,-0.301f,0.222f,-0.47f,0.043f,-0.221f,0.098f,-0.24f,0.206f,0.11f,-0.243f,0.261f,-0.051f,-0.436f,-0.672f,-0.534f,0.186f,-0.355f,0.256f,-0.233f,-0.118f,0.085f,0.109f,-0.159f,-0.332f,-0.456f,-0.096f,-0.251f,-0.253f,0.111f,-0.039f,-0.213f,0.084f,-0.031f,-0.787f,-0.371f,0.019f,-0.214f,-0.518f,0.392f,0.041f,0.326f,-0.24f,0.247f,-0.568f,-0.467f,-0.135f,0.067f,-0.054f,0.009f,-0.383f,0.049f,-0.457f,0.207f,-0.137f,0.563f,0.089f,0.237f,-0.177f,-0.623f,-0.408f,0.026f,-0.22f,0.304f,-0.166f,0.009f,-0.653f,-0.541f,0.196f,-0.841f,0.091f,-0.228f,-0.266f,-0.506f,-0.041f,-0.24f,-0.4f,-0.171f,-0.197f,0.087f,-0.692f,-0.0f,-0.152f,0.056f,-0.346f,0.021f,-0.12f,-0.458f,0.045f,-0.026f};
        float[][] res = new float[pca_means.length][1];
        for (int i=0; i<pca_means.length; i++){
            res[i][0] = pca_means[i];
        }
        return res;
    }

    private static float[][] get_PCA_params(Context c) throws IOException {
        float[][] pca_params = new float[128][128];
        Scanner sc = new Scanner(new BufferedReader(new InputStreamReader(c.getAssets().open(pca_params_path))));
        while (sc.hasNextLine()){
            for (int i=0; i<pca_params.length; i++){
                String[] line = sc.nextLine().trim().split(" ");
                for (int j=0; j<line.length; j++){
                    pca_params[i][j] = Float.parseFloat(line[j]);
                }
            }
        }
        return pca_params;
    }
}
