package com.vectordb.db;

import com.vectordb.index.BruteForce;
import com.vectordb.index.HNSW;
import com.vectordb.model.DocItem;
import com.vectordb.model.VectorItem;
import com.vectordb.util.Metrics;

import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Document store backed by HNSW over real Ollama 768D embeddings.
 * Equivalent to C++ class DocumentDB.
 */
public class DocumentDB {

    private final Map<Integer, DocItem> store = new LinkedHashMap<>();
    private final HNSW       hnsw = new HNSW(16, 200);
    private final BruteForce bf   = new BruteForce();
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private int nextId = 1;
    private int dims   = 0;  // set on first insert

    // ── Insert ────────────────────────────────────────────────────────

    public int insert(String title, String text, float[] embedding) {
        lock.writeLock().lock();
        try {
            if (dims == 0) dims = embedding.length;
            DocItem item = new DocItem(nextId++, title, text, embedding);
            store.put(item.id, item);
            VectorItem vi = new VectorItem(item.id, title, "doc", embedding);
            hnsw.insert(vi, Metrics::cosine);
            bf.insert(vi);
            return item.id;
        } finally { lock.writeLock().unlock(); }
    }

    // ── Search ────────────────────────────────────────────────────────

    public static class DocHit {
        public final float  distance;
        public final DocItem item;
        public DocHit(float dist, DocItem item) { this.distance = dist; this.item = item; }
    }

    public List<DocHit> search(float[] query, int k) {
        return search(query, k, 0.7f);
    }

    public List<DocHit> search(float[] query, int k, float maxDist) {
        lock.readLock().lock();
        try {
            if (store.isEmpty()) return Collections.emptyList();
            List<float[]> raw = (store.size() < 10)
                    ? bf.knn(query, k, Metrics::cosine)
                    : hnsw.knn(query, k, 50, Metrics::cosine);

            List<DocHit> out = new ArrayList<>();
            for (float[] r : raw) {
                int id = (int) r[1];
                DocItem item = store.get(id);
                if (item != null && r[0] <= maxDist) out.add(new DocHit(r[0], item));
            }
            return out;
        } finally { lock.readLock().unlock(); }
    }

    // ── Remove ───────────────────────────────────────────────────────

    public boolean remove(int id) {
        lock.writeLock().lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally { lock.writeLock().unlock(); }
    }

    // ── Accessors ─────────────────────────────────────────────────────

    public List<DocItem> all() {
        lock.readLock().lock();
        try { return new ArrayList<>(store.values()); }
        finally { lock.readLock().unlock(); }
    }

    public int size() {
        lock.readLock().lock();
        try { return store.size(); }
        finally { lock.readLock().unlock(); }
    }

    public int getDims() { return dims; }
}
