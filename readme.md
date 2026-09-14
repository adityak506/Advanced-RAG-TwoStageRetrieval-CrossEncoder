# 📘 Advanced RAG Mini-Project — Two-Stage Retrieval + Cross-Encoder Re-Ranking

This mini-project demonstrates one of the most **powerful retrieval pipelines** used in modern enterprise RAG applications:

✅ Stage-1: **Semantic Retrieval (FAISS)**
✅ Stage-2: **Cross-Encoder Reranking (SentenceTransformer)**
✅ Final Answer: **LLM grounded in reranked top-k chunks**

This pipeline is used by:

* Bing Search
* Perplexity AI
* Meta RAG stack
* OpenAI Retrieval API (under the hood)

You now have a fully modular, production-style implementation of it.

---

# 🚀 Project Features

### ✅ 1. **Semantic Retrieval (FAISS) — Stage 1**

* Retrieves a *large candidate pool* (default: top-30 chunks)
* Uses OpenAI or Gemini embeddings
* Provider-aware: separates FAISS directories to avoid dimensionality conflicts
* Fast and scalable

### ✅ 2. **Cross-Encoder Re-Ranking — Stage 2**

Powered by SentenceTransformers:

```
cross-encoder/ms-marco-MiniLM-L6-v2
```

The re-ranker scores each pair:

```
(query, document_chunk)
```

✅ Much more accurate than embeddings
✅ Scores can be positive or negative (expected behavior)
✅ Produces a *sorted*, truly relevant final top-k

### ✅ 3. **LLM Answering with Final Reranked Context**

The final top-k reranked documents are merged and passed to:

* GPT-4o-mini **or**
* Gemini 2.5 Flash

Answer is strictly grounded using a guardrail prompt.

### ✅ 4. **Unified Pipeline**

A single retriever class orchestrates everything:

```
FAISS → CrossEncoder → Final Docs
```

### ✅ 5. **Modular Codebase**

You can reuse all utilities across future RAG/Agentic modules.

---

# 📂 Project Structure

```
14-Advanced-RAG-Part3/
│
├── app.py                      # Interactive CLI app
├── config.yaml                 # LLM choice, reranking parameters
├── requirements.txt            # Includes sentence-transformers + torch
│
├── data/
│   ├── raw/
│   │   └── insurance_docs/     # Sample documents
│   └── embeddings/
│       ├── faiss_openai/       # Auto-created
│       └── faiss_gemini/
│
└── utils/
    ├── loader.py               # Chunking + loader
    ├── retriever_faiss.py      # Base retriever (provider-aware)
    ├── reranker_cross_encoder.py  # Stage-2 reranker
    ├── two_stage_retriever.py     # Pipeline controller
```

---

# ⚙️ Installation

### 1️⃣ Create a virtual environment

```bash
python -m venv myenv
myenv\Scripts\activate     # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add API keys

Create `.env`:

```
OPENAI_API_KEY=xxxx
GEMINI_API_KEY=xxxx
```

### 4️⃣ Validate torch installation

```python
import torch
print(torch.__version__)
```

Should show a `+cpu` build on Windows.

---

# ✅ How It Works

### **Stage 1 — FAISS Retrieval**

```python
candidates = base_retriever.invoke(query)[:initial_k]
```

* Returns top-30 semantic matches
* Fast but not perfect for ranking

---

### **Stage 2 — Cross-Encoder Reranking**

```python
ranked = reranker.rerank(query, candidates, top_k=final_k)
```

* Re-scores (query, chunk) with a transformer
* Produces highly accurate final ordering

Scores example:

```
+6.9  → highly relevant  
-7.9  → irrelevant  
-10   → very irrelevant  
```

This is expected.

---

### **Final Answer — LLM With Grounded Context**

```python
answer = llm.invoke(...)
```

Context is strictly provided so hallucinations drop sharply.

---

# ▶️ Running the App

```
python app.py
```

Example:

```
🔍 Enter your question: What are key terms in an insurance contract?
```

You will see:

✅ Stage 1 — Candidate retrieval
✅ Stage 2 — Cross-encoder re-ranking w/ scores
✅ Final grounded answer
✅ Source chunks from reranked top-k

This gives students **instant clarity** on why reranking matters.

---

# ✅ Notebook Smoke Test (Recommended)

```python
from utils.two_stage_retriever import TwoStageRetriever
from utils.loader import load_and_chunk_docs

chunks = load_and_chunk_docs("./data/raw/insurance_docs")
retriever = TwoStageRetriever(chunks_if_needed=chunks)

docs = retriever.invoke("key terms of insurance contract")

for i, d in enumerate(docs):
    print(i+1, d.page_content[:150])
```

---

# 🎯 Learning Outcomes

Students completing this project will learn:

✅ Why simple semantic search is not enough
✅ What candidate generation vs. reranking means
✅ How cross-encoders drastically improve accuracy
✅ How to merge FAISS & cross-encoder in production
✅ How to structure multi-stage pipelines cleanly
✅ How to support multiple embedding providers safely

This module sets the foundation for:

✅ Agentic RAG
✅ Retrieval fusion
✅ Advanced ranking logic
✅ LTR (Learning to Rank)

---

# ✅ Status: COMPLETED ✅

You now have a clean, reproducible, enterprise-grade **Two-Stage Retrieval + Reranking** mini-project.