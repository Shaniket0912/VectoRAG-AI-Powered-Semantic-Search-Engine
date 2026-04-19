package com.vectordb.util;

import java.util.function.BiFunction;

/**
 * The three distance functions from the C++ code, now as Java lambdas/statics.
 */
public final class Metrics {

    public static float euclidean(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na  += a[i] * a[i];
            nb  += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / ((float) Math.sqrt(na) * (float) Math.sqrt(nb));
    }

    public static float manhattan(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
        return s;
    }

    /** Returns the distance function for a metric name string. */
    public static BiFunction<float[], float[], Float> get(String metric) {
        return switch (metric) {
            case "cosine"    -> Metrics::cosine;
            case "manhattan" -> Metrics::manhattan;
            default          -> Metrics::euclidean;
        };
    }

    private Metrics() {}
}
