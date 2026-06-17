# 🚀 VectoRAG — AI-Powered Semantic Search Engine
[![Live Demo](https://img.shields.io/badge/🌐%20LIVE%20DEMO-Click%20Here-brightgreen?style=for-the-badge)](http://YOUR_VM_PUBLIC_IP:9090)

![Java](https://img.shields.io/badge/Java-17-orange?style=flat-square)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.2-brightgreen?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A fully functional **Vector Database Engine** built from scratch in Java, replicating the core architecture of production AI tools like **Pinecone**, **Weaviate**, and **Chroma**.

Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms with a complete **RAG (Retrieval-Augmented Generation)** pipeline powered by a local LLM via Ollama — **no cloud, no paid APIs, runs 100% on your machine.** 🔒

> 💡 Built to understand how AI Search Engines actually work under the hood — not just call an API.

---

## ✨ What This Project Does

| Feature | Description |
|---------|-------------|
| 🧠 **3 Search Algorithms** | HNSW, KD-Tree, Brute Force — run all three and compare speed |
| 📐 **3 Distance Metrics** | Cosine Similarity, Euclidean Distance, Manhattan Distance |
| 🎯 **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| 📊 **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| 📄 **Real Document Embedding** | Paste any text → Ollama embeds it with nomic-embed-text (768D) |
| 🤖 **RAG Pipeline** | Ask questions → HNSW retrieves context → local LLM answers |
| 🔌 **Full REST API** | 13 endpoints: insert, delete, search, benchmark, hnsw-info, doc/ask |
| ⚡ **Algorithm Benchmark** | Compare latency of all 3 algorithms side by side |

---

## 🧩 How RAG Works — The Core Idea

```
📝 Your Text
      │
      ▼
🔢 Ollama (nomic-embed-text)     ← converts text to a 768D vector
      │
      ▼
🕸️  HNSW Index (Java)            ← indexes the vector in a multilayer graph
      │
      ▼
🔍 Semantic Search               ← finds nearest neighbours in vector space
      │
      ▼
🤖 Ollama (llama3.2)             ← reads retrieved chunks, generates answer
      │
      ▼
✅ Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| ☕ **Java 17** | Core language |
| 🍃 **Spring Boot 3.2** | HTTP server + REST API |
| 🔧 **Jackson** | JSON serialization (comes with Spring Boot) |
| 🦙 **Ollama** | Local LLM — embeddings + text generation |
| 📦 **nomic-embed-text** | Converts text to 768D vectors |
| 💬 **llama3.2** | Generates answers from retrieved context |

---

## 📁 Project Structure

```
src/main/java/com/vectordb/
│
├── 🚀 VectorDBApplication.java        ← Spring Boot entry point, bean config
│
├── 🎮 controller/
│   ├── VectorController.java          ← Demo vector endpoints
│   └── DocumentController.java        ← RAG endpoints
│
├── ⚙️  index/
│   ├── HNSW.java                      ← Hierarchical Navigable Small World O(log N)
│   ├── KDTree.java                    ← KD-Tree spatial index O(log N)
│   └── BruteForce.java                ← Linear scan — exact but O(N)
│
├── 🗄️  db/
│   ├── VectorDB.java                  ← 16D demo vector database (thread-safe)
│   ├── DocumentDB.java                ← RAG document store (thread-safe)
│   └── DemoData.java                  ← 20 pre-loaded demo vectors
│
├── 📦 model/
│   ├── VectorItem.java                ← Demo vector record
│   └── DocItem.java                   ← RAG document chunk record
│
├── 🦙 ollama/
│   └── OllamaClient.java              ← HTTP client for local Ollama API
│
└── 🔧 util/
    ├── Metrics.java                   ← Euclidean / Cosine / Manhattan distance
    └── TextChunker.java               ← Splits long text into overlapping chunks

src/main/resources/
├── 🌐 static/index.html               ← Web UI (served by Spring Boot)
└── ⚙️  application.properties          ← Server port config
```

---

## ✅ Prerequisites

You need **3 things** installed:

### 1. ☕ Java 17+
- Download from: https://adoptium.net
- Verify: `java -version`

### 2. 🔨 Maven 3.8+
- Download from: https://maven.apache.org/download.cgi
- Extract and add `bin` folder to system PATH
- Verify: `mvn -version`

### 3. 🦙 Ollama
- Download from: https://ollama.com
- Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

---

## ▶️ Setup & Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/VectoRAG.git
cd VectoRAG
```

### Step 2 — Build
```bash
mvn package
```

### Step 3 — Run
```bash
java -jar target/vectordb.jar
```

### Step 4 — Open in browser 🌐
```
http://localhost:8080
```

You should see:
```
=== VectoRAG — AI-Powered Semantic Search Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed: nomic-embed-text   gen: llama3.2
```

---

## 💡 Running in IntelliJ IDEA

1. `File → Open` → select project folder
2. Click **Trust Project**
3. Wait for Maven dependencies to download
4. Open `VectorDBApplication.java`
5. Click the green **▶ Run** button
6. Open `http://localhost:8080`

> 🔧 To change port: edit `src/main/resources/application.properties` → `server.port=9090`

---

## 🔌 REST API Endpoints

### 🎯 Demo Vector Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | KNN search |
| `POST` | `/insert` | Insert a vector |
| `DELETE` | `/delete/{id}` | Delete a vector |
| `GET` | `/items` | List all vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare algorithm speeds |
| `GET` | `/hnsw-info` | HNSW graph structure |
| `GET` | `/stats` | Database stats |

### 🤖 RAG Document Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/doc/insert` | Embed and store a document |
| `DELETE` | `/doc/delete/{id}` | Delete a document chunk |
| `GET` | `/doc/list` | List all stored documents |
| `POST` | `/doc/search` | Semantic search over documents |
| `POST` | `/doc/ask` | Full RAG — ask a question |
| `GET` | `/status` | Ollama + DB status |

### 📋 Example — Insert a Document
```json
POST /doc/insert
{
  "title": "My Notes",
  "text": "Your long text goes here..."
}
```

### 📋 Example — Ask a Question
```json
POST /doc/ask
{
  "question": "What is HNSW?",
  "k": 3
}
```

---

## 🧠 Algorithms Explained

### ⚡ HNSW — Hierarchical Navigable Small World
- Used by **Pinecone, Weaviate, Chroma** in production
- Builds a multilayer graph — top layers sparse, bottom dense
- Search starts at top layer and zooms in
- **O(log N)** complexity — extremely fast at scale

### 🌲 KD-Tree — K-Dimensional Tree
- Partitions space along alternating axes
- Efficient for low-dimensional vectors
- **O(log N)** average case

### 🔍 Brute Force
- Compares query against every single vector
- Always finds the exact nearest neighbour
- **O(N)** — slow at scale but 100% accurate
- Used as ground truth for benchmarking

---

## 📐 Distance Metrics

| Metric | Best For |
|--------|----------|
| 📊 **Cosine Similarity** | Text embeddings, semantic search |
| 📏 **Euclidean Distance** | Spatial data, image features |
| 🗺️ **Manhattan Distance** | Grid-based problems |

---

## 🎯 Key Concepts for Interviews

**Q: What is a Vector Database?**
> Stores data as high-dimensional vectors and retrieves them by **semantic similarity** rather than exact match.

**Q: What is RAG?**
> Retrieval-Augmented Generation — embed documents, retrieve relevant chunks for a query, pass as context to LLM for answer generation.

**Q: Why HNSW over Brute Force?**
> Brute Force is O(N) — with 1 million vectors it checks all 1M. HNSW is O(log N) — checks only ~20-30 nodes. Same accuracy, **1000x faster.**

**Q: Why Spring Boot?**
> Industry standard Java framework — handles HTTP server, dependency injection, JSON serialization. Used in most enterprise Java applications.

---

## 📸 Screenshots
<img width="1486" height="882" alt="image" src="https://github.com/user-attachments/assets/2589dfd0-133f-40e3-acb9-656ec99e2d47" />


<img width="1482" height="884" alt="image" src="https://github.com/user-attachments/assets/3afd3c2e-3493-40df-bcf3-182dfc312d60" />


<img width="1480" height="892" alt="image" src="https://github.com/user-attachments/assets/6b836cd9-50ed-4d43-8c9a-9c85c1360f18" />




https://github.com/user-attachments/assets/78452042-bf11-43f7-8581-9df60f708565








---

## 👨‍💻 Author

**Shaniket Tiwari**
MCA Student — NIET Greater Noida

---

## 📄 License

MIT License — free to use, modify, and distribute.
