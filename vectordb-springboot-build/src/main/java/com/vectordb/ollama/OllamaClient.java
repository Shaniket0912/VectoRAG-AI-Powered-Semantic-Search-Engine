package com.vectordb.ollama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

/**
 * Wraps the local Ollama REST API.
 * Uses Jackson (comes with Spring Boot) — no external JSON library needed.
 */
public class OllamaClient {

    private final String host;
    private final int    port;
    private final ObjectMapper mapper = new ObjectMapper();

    public String embedModel = "nomic-embed-text";
    public String genModel   = "llama3.2";

    public OllamaClient() { this("127.0.0.1", 11434); }

    public OllamaClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public boolean isAvailable() {
        try {
            HttpURLConnection conn = open("/api/tags", "GET");
            conn.setConnectTimeout(2000);
            return conn.getResponseCode() == 200;
        } catch (Exception e) { return false; }
    }

    public float[] embed(String text) {
        try {
            HttpURLConnection conn = open("/api/embeddings", "POST");
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(30000);
            String body = mapper.writeValueAsString(
                    mapper.createObjectNode().put("model", embedModel).put("prompt", text));
            send(conn, body);
            if (conn.getResponseCode() != 200) return new float[0];
            JsonNode arr = mapper.readTree(read(conn)).get("embedding");
            if (arr == null || !arr.isArray()) return new float[0];
            float[] vec = new float[arr.size()];
            for (int i = 0; i < arr.size(); i++) vec[i] = (float) arr.get(i).asDouble();
            return vec;
        } catch (Exception e) { return new float[0]; }
    }

    public String generate(String prompt) {
        try {
            HttpURLConnection conn = open("/api/generate", "POST");
            conn.setConnectTimeout(3000);
            conn.setReadTimeout(180_000);
            String body = mapper.writeValueAsString(
                    mapper.createObjectNode()
                          .put("model", genModel).put("prompt", prompt).put("stream", false));
            send(conn, body);
            if (conn.getResponseCode() != 200)
                return "ERROR: Ollama unavailable. Run: ollama serve";
            JsonNode el = mapper.readTree(read(conn)).get("response");
            return el != null ? el.asText() : "ERROR: empty response";
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve (" + e.getMessage() + ")";
        }
    }

    private HttpURLConnection open(String path, String method) throws IOException {
        URL url = new URL("http", host, port, path);
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setRequestMethod(method);
        c.setRequestProperty("Content-Type", "application/json");
        if ("POST".equals(method)) c.setDoOutput(true);
        return c;
    }

    private void send(HttpURLConnection conn, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        try (OutputStream os = conn.getOutputStream()) { os.write(bytes); }
    }

    private String read(HttpURLConnection conn) throws IOException {
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) sb.append(line);
            return sb.toString();
        }
    }
}
