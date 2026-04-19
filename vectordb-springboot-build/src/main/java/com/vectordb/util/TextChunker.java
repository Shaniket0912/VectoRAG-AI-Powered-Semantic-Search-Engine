package com.vectordb.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a document into overlapping word-based chunks.
 * Equivalent to C++ chunkText() function.
 */
public final class TextChunker {

    public static List<String> chunk(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length == 0)         return List.of();
        if (words.length <= chunkWords) return List.of(text);

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }

    /** Default: 250-word chunks with 30-word overlap. */
    public static List<String> chunk(String text) {
        return chunk(text, 250, 30);
    }

    private TextChunker() {}
}
