package com.vectordb;

import com.vectordb.db.DemoData;
import com.vectordb.db.DocumentDB;
import com.vectordb.db.VectorDB;
import com.vectordb.ollama.OllamaClient;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class VectorDBApplication {

    public static final int DIMS = 16;

    public static void main(String[] args) {
        SpringApplication.run(VectorDBApplication.class, args);
    }

    // ── Shared beans — injected into controllers ──────────────────────

    @Bean
    public VectorDB vectorDB() {
        VectorDB db = new VectorDB(DIMS);
        DemoData.load(db);
        return db;
    }

    @Bean
    public DocumentDB documentDB() {
        return new DocumentDB();
    }

    @Bean
    public OllamaClient ollamaClient() {
        OllamaClient ollama = new OllamaClient();
        // Print startup info
        boolean up = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java + Spring Boot) ===");
        System.out.println("http://localhost:8080");
        System.out.println("Ollama: " + (up ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (up) System.out.printf("  embed: %s   gen: %s%n", ollama.embedModel, ollama.genModel);
        return ollama;
    }
}
