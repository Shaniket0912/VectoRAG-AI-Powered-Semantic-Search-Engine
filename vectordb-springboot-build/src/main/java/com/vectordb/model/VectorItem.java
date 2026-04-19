package com.vectordb.model;

/**
 * Equivalent to C++ struct VectorItem — holds one vector entry in the demo DB.
 */
public class VectorItem {
    public final int id;
    public final String metadata;
    public final String category;
    public final float[] embedding;

    public VectorItem(int id, String metadata, String category, float[] embedding) {
        this.id        = id;
        this.metadata  = metadata;
        this.category  = category;
        this.embedding = embedding;
    }
}
