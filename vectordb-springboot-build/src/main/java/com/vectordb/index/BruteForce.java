package com.vectordb.index;

import com.vectordb.model.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

/**
 * O(N) linear scan — exact nearest-neighbour search.
 * Equivalent to C++ class BruteForce.
 */
public class BruteForce {

    private final List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) {
        items.add(v);
    }

    /**
     * Returns the k nearest neighbours sorted by distance ascending.
     * Each pair is (distance, id).
     */
    public List<float[]> knn(float[] query, int k, BiFunction<float[], float[], Float> dist) {
        List<float[]> scores = new ArrayList<>(items.size());
        for (VectorItem v : items) {
            scores.add(new float[]{dist.apply(query, v.embedding), v.id});
        }
        scores.sort(Comparator.comparingDouble(p -> p[0]));
        return scores.subList(0, Math.min(k, scores.size()));
    }

    public void remove(int id) {
        items.removeIf(v -> v.id == id);
    }

    public List<VectorItem> getItems() {
        return Collections.unmodifiableList(items);
    }
}
