package com.vectordb.model;

/**
 * One text chunk stored in the DocumentDB for RAG.
 */
public class DocItem {
    public final int     id;
    public final String  title;
    public final String  text;
    public final float[] embedding;

    public DocItem(int id, String title, String text, float[] embedding) {
        this.id        = id;
        this.title     = title;
        this.text      = text;
        this.embedding = embedding;
    }
}
