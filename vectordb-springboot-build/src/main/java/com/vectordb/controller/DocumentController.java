package com.vectordb.controller;

import com.vectordb.db.DocumentDB;
import com.vectordb.db.VectorDB;
import com.vectordb.model.DocItem;
import com.vectordb.ollama.OllamaClient;
import com.vectordb.util.TextChunker;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * REST endpoints for RAG document pipeline.
 * Equivalent to C++ /doc/insert, /doc/delete, /doc/list, /doc/search, /doc/ask, /status
 */
@RestController
@CrossOrigin(origins = "*")
public class DocumentController {

    private final DocumentDB  docDB;
    private final VectorDB    db;
    private final OllamaClient ollama;

    public DocumentController(DocumentDB docDB, VectorDB db, OllamaClient ollama) {
        this.docDB  = docDB;
        this.db     = db;
        this.ollama = ollama;
    }

    // ── POST /doc/insert  {"title":"...","text":"..."} ────────────────

    @PostMapping("/doc/insert")
    public Map<String, Object> insertDoc(@RequestBody Map<String, String> body) {
        String title = body.get("title");
        String text  = body.get("text");

        if (title == null || text == null) {
            return Map.of("error", "need title and text");
        }

        List<String> chunks = TextChunker.chunk(text);
        List<Integer> ids   = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                return Map.of("error",
                    "Ollama unavailable. Install from https://ollama.com " +
                    "then run: ollama pull nomic-embed-text && ollama pull llama3.2");
            }
            String chunkTitle = (chunks.size() > 1)
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        return Map.of(
                "ids",    ids,
                "chunks", chunks.size(),
                "dims",   docDB.getDims()
        );
    }

    // ── DELETE /doc/delete/{id} ───────────────────────────────────────

    @DeleteMapping("/doc/delete/{id}")
    public Map<String, Object> deleteDoc(@PathVariable int id) {
        return Map.of("ok", docDB.remove(id));
    }

    // ── GET /doc/list ─────────────────────────────────────────────────

    @GetMapping("/doc/list")
    public List<Map<String, Object>> listDocs() {
        return docDB.all().stream().map(d -> {
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
            int words = d.text.split("\\s+").length;
            return Map.<String, Object>of(
                    "id",      d.id,
                    "title",   d.title,
                    "preview", preview,
                    "words",   words
            );
        }).toList();
    }

    // ── POST /doc/search  {"question":"...","k":3} ────────────────────

    @PostMapping("/doc/search")
    public Map<String, Object> searchDoc(@RequestBody Map<String, Object> body) {
        String question = (String) body.get("question");
        int k = body.containsKey("k") ? ((Number) body.get("k")).intValue() : 3;

        if (question == null) return Map.of("error", "need question");

        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) return Map.of("error", "Ollama unavailable");

        List<Map<String, Object>> contexts = docDB.search(qEmb, k).stream().map(h ->
                Map.<String, Object>of(
                        "id",       h.item.id,
                        "title",    h.item.title,
                        "distance", h.distance
                )
        ).toList();

        return Map.of("contexts", contexts);
    }

    // ── POST /doc/ask  {"question":"...","k":3} — full RAG pipeline ───

    @PostMapping("/doc/ask")
    public Map<String, Object> askDoc(@RequestBody Map<String, Object> body) {
        String question = (String) body.get("question");
        int k = body.containsKey("k") ? ((Number) body.get("k")).intValue() : 3;

        if (question == null) return Map.of("error", "need question");

        // Step 1: Embed the question
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) return Map.of("error", "Ollama unavailable");

        // Step 2: Retrieve top-k relevant chunks
        var hits = docDB.search(qEmb, k);

        // Step 3: Build prompt
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ")
               .append(hits.get(i).item.title).append(":\n")
               .append(hits.get(i).item.text).append("\n\n");
        }
        String prompt =
            "You are a helpful assistant. Answer the user's question directly. " +
            "Use the provided context if it contains relevant information. " +
            "If it doesn't, just use your own general knowledge. " +
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like " +
            "'the context doesn't mention'. Just answer the question naturally.\n\n" +
            "Context:\n" + ctx +
            "Question: " + question + "\n\nAnswer:";

        // Step 4: Generate answer
        String answer = ollama.generate(prompt);

        // Step 5: Return everything
        List<Map<String, Object>> contexts = hits.stream().map(h ->
                Map.<String, Object>of(
                        "id",       h.item.id,
                        "title",    h.item.title,
                        "text",     h.item.text,
                        "distance", h.distance
                )
        ).toList();

        return Map.of(
                "answer",   answer,
                "model",    ollama.genModel,
                "contexts", contexts,
                "docCount", docDB.size()
        );
    }

    // ── GET /status ───────────────────────────────────────────────────

    @GetMapping("/status")
    public Map<String, Object> status() {
        boolean up = ollama.isAvailable();
        return Map.of(
                "ollamaAvailable", up,
                "embedModel",      ollama.embedModel,
                "genModel",        ollama.genModel,
                "docCount",        docDB.size(),
                "docDims",         docDB.getDims(),
                "demoDims",        com.vectordb.VectorDBApplication.DIMS,
                "demoCount",       db.size()
        );
    }
}
