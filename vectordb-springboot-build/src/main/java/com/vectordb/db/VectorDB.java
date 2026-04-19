package com.vectordb.db;

import com.vectordb.index.BruteForce;
import com.vectordb.index.HNSW;
import com.vectordb.index.KDTree;
import com.vectordb.model.VectorItem;
import com.vectordb.util.Metrics;

import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiFunction;

/**
 * The demo 16D vector database.
 * Equivalent to C++ class VectorDB.
 * Thread-safe with a read-write lock.
 */
public class VectorDB {

    public final int dims;
    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf;
    private final KDTree     kdt;
    private final HNSW       hnsw;
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private int nextId = 1;

    public VectorDB(int dims) {
        this.dims = dims;
        this.bf   = new BruteForce();
        this.kdt  = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
    }

    // ── Insert ────────────────────────────────────────────────────────

    public int insert(String metadata, String category, float[] embedding,
                      BiFunction<float[], float[], Float> dist)
    {
        lock.writeLock().lock();
        try {
            VectorItem v = new VectorItem(nextId++, metadata, category, embedding);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally { lock.writeLock().unlock(); }
    }

    // ── Remove ───────────────────────────────────────────────────────

    public boolean remove(int id) {
        lock.writeLock().lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally { lock.writeLock().unlock(); }
    }

    // ── Search ────────────────────────────────────────────────────────

    public static class SearchResult {
        public final List<Hit> hits;
        public final long latencyUs;
        public final String algo, metric;

        public SearchResult(List<Hit> hits, long us, String algo, String metric) {
            this.hits = hits; this.latencyUs = us; this.algo = algo; this.metric = metric;
        }
    }

    public static class Hit {
        public final int id;
        public final String meta, cat;
        public final float[] emb;
        public final float dist;

        public Hit(int id, String meta, String cat, float[] emb, float dist) {
            this.id = id; this.meta = meta; this.cat = cat; this.emb = emb; this.dist = dist;
        }
    }

    public SearchResult search(float[] query, int k, String metric, String algo) {
        lock.readLock().lock();
        try {
            BiFunction<float[], float[], Float> dfn = Metrics.get(metric);
            long t0 = System.nanoTime();

            List<float[]> raw;
            if ("bruteforce".equals(algo))     raw = bf.knn(query, k, dfn);
            else if ("kdtree".equals(algo))    raw = kdt.knn(query, k, dfn);
            else                               raw = hnsw.knn(query, k, 50, dfn);

            long us = (System.nanoTime() - t0) / 1000;

            List<Hit> hits = new ArrayList<>();
            for (float[] r : raw) {
                int id = (int) r[1];
                VectorItem v = store.get(id);
                if (v != null) hits.add(new Hit(id, v.metadata, v.category, v.embedding, r[0]));
            }
            return new SearchResult(hits, us, algo, metric);
        } finally { lock.readLock().unlock(); }
    }

    // ── Benchmark ─────────────────────────────────────────────────────

    public static class BenchResult {
        public final long bfUs, kdUs, hnswUs;
        public final int n;
        public BenchResult(long bf, long kd, long hnsw, int n) {
            bfUs = bf; kdUs = kd; hnswUs = hnsw; this.n = n;
        }
    }

    public BenchResult benchmark(float[] query, int k, String metric) {
        lock.readLock().lock();
        try {
            BiFunction<float[], float[], Float> dfn = Metrics.get(metric);
            long t;

            t = System.nanoTime(); bf.knn(query, k, dfn);
            long bfUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime(); kdt.knn(query, k, dfn);
            long kdUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime(); hnsw.knn(query, k, 50, dfn);
            long hnswUs = (System.nanoTime() - t) / 1000;

            return new BenchResult(bfUs, kdUs, hnswUs, store.size());
        } finally { lock.readLock().unlock(); }
    }

    // ── Accessors ─────────────────────────────────────────────────────

    public List<VectorItem> all() {
        lock.readLock().lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.readLock().unlock(); }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.readLock().lock();
        try { return hnsw.getInfo(); }
        finally { lock.readLock().unlock(); }
    }

    public int size() {
        lock.readLock().lock();
        try { return store.size(); }
        finally { lock.readLock().unlock(); }
    }
}
