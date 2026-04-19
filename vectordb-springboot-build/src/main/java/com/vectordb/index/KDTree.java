package com.vectordb.index;

import com.vectordb.model.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

/**
 * KD-Tree for approximate nearest-neighbour search.
 * Equivalent to C++ class KDTree.
 */
public class KDTree {

    private static class Node {
        VectorItem item;
        Node left, right;
        Node(VectorItem item) { this.item = item; }
    }

    private Node root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    // ── Insert ────────────────────────────────────────────────────────

    public void insert(VectorItem v) {
        root = insert(root, v, 0);
    }

    private Node insert(Node n, VectorItem v, int depth) {
        if (n == null) return new Node(v);
        int axis = depth % dims;
        if (v.embedding[axis] < n.item.embedding[axis]) n.left  = insert(n.left,  v, depth + 1);
        else                                             n.right = insert(n.right, v, depth + 1);
        return n;
    }

    // ── KNN search ───────────────────────────────────────────────────

    /** Returns k nearest neighbours as (distance, id) pairs, sorted ascending. */
    public List<float[]> knn(float[] query, int k, BiFunction<float[], float[], Float> dist) {
        // Max-heap of size k: heap.peek() = worst (largest distance)
        PriorityQueue<float[]> heap = new PriorityQueue<>(
                k + 1, Comparator.<float[], Float>comparing(p -> p[0]).reversed());
        knn(root, query, k, 0, dist, heap);

        List<float[]> result = new ArrayList<>(heap);
        result.sort(Comparator.comparingDouble(p -> p[0]));
        return result;
    }

    private void knn(Node n, float[] q, int k, int depth,
                     BiFunction<float[], float[], Float> dist,
                     PriorityQueue<float[]> heap)
    {
        if (n == null) return;

        float dn = dist.apply(q, n.item.embedding);
        heap.add(new float[]{dn, n.item.id});
        if (heap.size() > k) heap.poll();   // remove worst

        int axis = depth % dims;
        float diff = q[axis] - n.item.embedding[axis];
        Node closer  = diff < 0 ? n.left  : n.right;
        Node farther = diff < 0 ? n.right : n.left;

        knn(closer, q, k, depth + 1, dist, heap);

        // Only explore the far branch if it could contain a closer point
        if (heap.size() < k || Math.abs(diff) < heap.peek()[0]) {
            knn(farther, q, k, depth + 1, dist, heap);
        }
    }

    // ── Rebuild (after deletions) ─────────────────────────────────────

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }
}
