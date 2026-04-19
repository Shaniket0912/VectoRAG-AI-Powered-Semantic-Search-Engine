package com.vectordb.controller;

import com.vectordb.VectorDBApplication;
import com.vectordb.db.VectorDB;
import com.vectordb.index.HNSW;
import com.vectordb.model.VectorItem;
import com.vectordb.util.Metrics;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST endpoints for the 16D demo vector database.
 * Equivalent to C++ /search, /insert, /delete, /items, /benchmark, /hnsw-info
 */
@RestController
@CrossOrigin(origins = "*")
public class VectorController {

    private final VectorDB db;

    public VectorController(VectorDB db) {
        this.db = db;
    }

    // ── GET /search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw ───────────

    @GetMapping("/search")
    public Map<String, Object> search(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric,
            @RequestParam(defaultValue = "hnsw") String algo)
    {
        float[] query = parseVec(v);
        if (query.length != VectorDBApplication.DIMS) {
            return Map.of("error", "need " + VectorDBApplication.DIMS + "D vector");
        }
        VectorDB.SearchResult out = db.search(query, k, metric, algo);

        List<Map<String, Object>> results = out.hits.stream().map(h -> Map.<String, Object>of(
                "id",        h.id,
                "metadata",  h.meta,
                "category",  h.cat,
                "distance",  h.dist,
                "embedding", toList(h.emb)
        )).toList();

        return Map.of(
                "results",   results,
                "latencyUs", out.latencyUs,
                "algo",      out.algo,
                "metric",    out.metric
        );
    }

    // ── POST /insert  {"metadata":"...","category":"...","embedding":[...]} ──

    @PostMapping("/insert")
    public Map<String, Object> insert(@RequestBody Map<String, Object> body) {
        String meta = (String) body.get("metadata");
        String cat  = (String) body.getOrDefault("category", "");
        float[] emb = parseEmbedding(body.get("embedding"));

        if (meta == null || emb == null || emb.length != VectorDBApplication.DIMS) {
            return Map.of("error", "invalid body");
        }
        int id = db.insert(meta, cat, emb, Metrics.get("cosine"));
        return Map.of("id", id);
    }

    // ── DELETE /delete/{id} ───────────────────────────────────────────

    @DeleteMapping("/delete/{id}")
    public Map<String, Object> delete(@PathVariable int id) {
        return Map.of("ok", db.remove(id));
    }

    // ── GET /items ───────────────────────────────────────────────────

    @GetMapping("/items")
    public List<Map<String, Object>> items() {
        return db.all().stream().map(v -> Map.<String, Object>of(
                "id",        v.id,
                "metadata",  v.metadata,
                "category",  v.category,
                "embedding", toList(v.embedding)
        )).toList();
    }

    // ── GET /benchmark?v=...&k=5&metric=cosine ────────────────────────

    @GetMapping("/benchmark")
    public Map<String, Object> benchmark(
            @RequestParam String v,
            @RequestParam(defaultValue = "5") int k,
            @RequestParam(defaultValue = "cosine") String metric)
    {
        float[] query = parseVec(v);
        if (query.length != VectorDBApplication.DIMS) {
            return Map.of("error", "need " + VectorDBApplication.DIMS + "D vector");
        }
        VectorDB.BenchResult b = db.benchmark(query, k, metric);
        return Map.of(
                "bruteforceUs", b.bfUs,
                "kdtreeUs",     b.kdUs,
                "hnswUs",       b.hnswUs,
                "itemCount",    b.n
        );
    }

    // ── GET /hnsw-info ───────────────────────────────────────────────

    @GetMapping("/hnsw-info")
    public Map<String, Object> hnswInfo() {
        HNSW.GraphInfo gi = db.hnswInfo();

        List<Map<String, Object>> nodes = gi.nodes.stream().map(n -> Map.<String, Object>of(
                "id",       n.id,
                "metadata", n.metadata,
                "category", n.category,
                "maxLyr",   n.maxLyr
        )).toList();

        List<Map<String, Object>> edges = gi.edges.stream().map(e -> Map.<String, Object>of(
                "src", e.src,
                "dst", e.dst,
                "lyr", e.lyr
        )).toList();

        return Map.of(
                "topLayer",      gi.topLayer,
                "nodeCount",     gi.nodeCount,
                "nodesPerLayer", toList(gi.nodesPerLayer),
                "edgesPerLayer", toList(gi.edgesPerLayer),
                "nodes",         nodes,
                "edges",         edges
        );
    }

    // ── GET /stats ───────────────────────────────────────────────────

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        return Map.of(
                "count",      db.size(),
                "dims",       VectorDBApplication.DIMS,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics",    List.of("euclidean", "cosine", "manhattan")
        );
    }

    // ── Helpers ───────────────────────────────────────────────────────

    private float[] parseVec(String s) {
        if (s == null || s.isBlank()) return new float[0];
        String[] parts = s.split(",");
        try {
            float[] v = new float[parts.length];
            for (int i = 0; i < parts.length; i++) v[i] = Float.parseFloat(parts[i].trim());
            return v;
        } catch (NumberFormatException e) { return new float[0]; }
    }

    @SuppressWarnings("unchecked")
    private float[] parseEmbedding(Object obj) {
        if (!(obj instanceof List)) return null;
        List<Number> list = (List<Number>) obj;
        float[] v = new float[list.size()];
        for (int i = 0; i < list.size(); i++) v[i] = list.get(i).floatValue();
        return v;
    }

    private List<Float> toList(float[] arr) {
        List<Float> l = new java.util.ArrayList<>(arr.length);
        for (float f : arr) l.add(f);
        return l;
    }

    private List<Integer> toList(int[] arr) {
        List<Integer> l = new java.util.ArrayList<>(arr.length);
        for (int i : arr) l.add(i);
        return l;
    }
}
