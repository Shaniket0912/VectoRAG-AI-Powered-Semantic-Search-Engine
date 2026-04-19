package com.vectordb.index;

import com.vectordb.model.VectorItem;

import java.util.*;
import java.util.function.BiFunction;

/**
 * Hierarchical Navigable Small World graph.
 * Full Java port of the C++ HNSW class.
 * O(log N) approximate nearest-neighbour search.
 */
public class HNSW {

    // ── Internal node ─────────────────────────────────────────────────

    private static class Node {
        VectorItem item;
        int maxLayer;
        List<List<Integer>> nbrs; // nbrs.get(layer) = neighbour IDs at that layer

        Node(VectorItem item, int maxLayer) {
            this.item     = item;
            this.maxLayer = maxLayer;
            this.nbrs     = new ArrayList<>();
            for (int i = 0; i <= maxLayer; i++) this.nbrs.add(new ArrayList<>());
        }
    }

    // ── Graph state ───────────────────────────────────────────────────

    private final Map<Integer, Node> graph = new HashMap<>();
    private final int    M;          // max neighbours per layer > 0
    private final int    M0;         // max neighbours at layer 0
    private final int    efBuild;    // beam width during construction
    private final double mL;         // level multiplier
    private int topLayer = -1;
    private int entryPt  = -1;

    private final Random rng = new Random(42);

    public HNSW(int m, int efBuild) {
        this.M        = m;
        this.M0       = 2 * m;
        this.efBuild  = efBuild;
        this.mL       = 1.0 / Math.log(m);
    }

    // ── Random level sampling ─────────────────────────────────────────

    private int randomLevel() {
        return (int) Math.floor(-Math.log(rng.nextDouble()) * mL);
    }

    // ── Greedy layer search ───────────────────────────────────────────

    /**
     * Search a single layer from entry point ep, return ef candidates.
     * Returns list of (distance, id) sorted ascending.
     */
    private List<float[]> searchLayer(float[] query, int ep, int ef, int layer,
                                      BiFunction<float[], float[], Float> dist)
    {
        Set<Integer> visited = new HashSet<>();
        // Min-heap for candidates
        PriorityQueue<float[]> cands  = new PriorityQueue<>(Comparator.comparingDouble(p -> p[0]));
        // Max-heap for found (worst on top so we can prune)
        PriorityQueue<float[]> found  = new PriorityQueue<>(Comparator.<float[], Float>comparing(p -> p[0]).reversed());

        float d0 = dist.apply(query, graph.get(ep).item.embedding);
        visited.add(ep);
        cands.add(new float[]{d0, ep});
        found.add(new float[]{d0, ep});

        while (!cands.isEmpty()) {
            float[] cur = cands.poll();
            if (found.size() >= ef && cur[0] > found.peek()[0]) break;

            Node curNode = graph.get((int) cur[1]);
            if (curNode == null || layer >= curNode.nbrs.size()) continue;

            for (int nid : curNode.nbrs.get(layer)) {
                if (visited.contains(nid) || !graph.containsKey(nid)) continue;
                visited.add(nid);
                float nd = dist.apply(query, graph.get(nid).item.embedding);
                if (found.size() < ef || nd < found.peek()[0]) {
                    cands.add(new float[]{nd, nid});
                    found.add(new float[]{nd, nid});
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<float[]> res = new ArrayList<>(found);
        res.sort(Comparator.comparingDouble(p -> p[0]));
        return res;
    }

    // ── Neighbour selection ───────────────────────────────────────────

    private List<Integer> selectNeighbours(List<float[]> cands, int maxM) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
            result.add((int) cands.get(i)[1]);
        }
        return result;
    }

    // ── Insert ────────────────────────────────────────────────────────

    public void insert(VectorItem item, BiFunction<float[], float[], Float> dist) {
        int id  = item.id;
        int lvl = randomLevel();
        graph.put(id, new Node(item, lvl));

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(item.embedding, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<float[]> W   = searchLayer(item.embedding, ep, efBuild, lc, dist);
            int maxM          = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNeighbours(W, maxM);

            Node curNode = graph.get(id);
            while (curNode.nbrs.size() <= lc) curNode.nbrs.add(new ArrayList<>());
            curNode.nbrs.get(lc).addAll(sel);

            // Prune neighbours of selected nodes
            for (int nid : sel) {
                Node nNode = graph.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) nNode.nbrs.add(new ArrayList<>());
                nNode.nbrs.get(lc).add(id);
                if (nNode.nbrs.get(lc).size() > maxM) {
                    List<float[]> ds = new ArrayList<>();
                    for (int c : nNode.nbrs.get(lc)) {
                        if (graph.containsKey(c))
                            ds.add(new float[]{dist.apply(nNode.item.embedding, graph.get(c).item.embedding), c});
                    }
                    ds.sort(Comparator.comparingDouble(p -> p[0]));
                    List<Integer> pruned = new ArrayList<>();
                    for (int i = 0; i < Math.min(maxM, ds.size()); i++) pruned.add((int) ds.get(i)[1]);
                    nNode.nbrs.set(lc, pruned);
                }
            }
            if (!W.isEmpty()) ep = (int) W.get(0)[1];
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    // ── KNN search ───────────────────────────────────────────────────

    /** Returns k nearest neighbours as (distance, id) pairs sorted ascending. */
    public List<float[]> knn(float[] query, int k, int ef,
                             BiFunction<float[], float[], Float> dist)
    {
        if (entryPt == -1) return Collections.emptyList();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = graph.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<float[]> W = searchLayer(query, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = (int) W.get(0)[1];
            }
        }
        List<float[]> W = searchLayer(query, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) W = W.subList(0, k);
        return W;
    }

    // ── Remove ───────────────────────────────────────────────────────

    public void remove(int id) {
        if (!graph.containsKey(id)) return;
        for (Node nd : graph.values()) {
            for (List<Integer> layer : nd.nbrs) layer.remove((Integer) id);
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : graph.keySet()) {
                if (nid != id) { entryPt = nid; break; }
            }
        }
        graph.remove(id);
    }

    // ── Graph info (for /hnsw-info endpoint) ─────────────────────────

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer  = topLayer;
        gi.nodeCount = graph.size();
        int maxL = Math.max(topLayer + 1, 1);
        gi.nodesPerLayer = new int[maxL];
        gi.edgesPerLayer = new int[maxL];

        for (Map.Entry<Integer, Node> e : graph.entrySet()) {
            int id   = e.getKey();
            Node nd  = e.getValue();
            NodeView nv = new NodeView();
            nv.id       = id;
            nv.metadata = nd.item.metadata;
            nv.category = nd.item.category;
            nv.maxLyr   = nd.maxLayer;
            gi.nodes.add(nv);

            for (int lc = 0; lc <= nd.maxLayer && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            EdgeView ev = new EdgeView();
                            ev.src = id; ev.dst = nid; ev.lyr = lc;
                            gi.edges.add(ev);
                        }
                    }
                }
            }
        }
        return gi;
    }

    public int size() { return graph.size(); }

    // ── Graph info DTOs ───────────────────────────────────────────────

    public static class GraphInfo {
        public int   topLayer, nodeCount;
        public int[] nodesPerLayer, edgesPerLayer;
        public List<NodeView> nodes = new ArrayList<>();
        public List<EdgeView> edges = new ArrayList<>();
    }

    public static class NodeView {
        public int id, maxLyr;
        public String metadata, category;
    }

    public static class EdgeView {
        public int src, dst, lyr;
    }
}
