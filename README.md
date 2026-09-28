# 🚀 End-to-End RAG Application

> **RAG = Retrieval + Augmented + Generation**
> Ask any question on your data (PDFs, docs, etc.) and get accurate, grounded answers with references.

Built following the **End-to-End RAG Application Workflow** diagram — from Problem Statement → Data → Retrieval → Augmented Generation → Evaluation → Deployment.

---

## 📁 Project Structure

```
Antigravity_RAG/
├── data/
│   ├── raw/          ← Drop your source files here (.pdf, .docx, .txt, .md, .json)
│   ├── cleaned/      ← Auto-generated cleaned JSON files (Step 6)
│   └── chunks/       ← Auto-generated chunk JSON files (Step 7)
├── vector_store/     ← FAISS index + metadata (Step 9-10)
├── data_processing.py  ← Steps 5-6: ETL / text extraction
├── chunking.py         ← Step 7: Chunking strategies
├── vector_store_manager.py  ← Steps 8-9: Embeddings + FAISS
├── rag_pipeline.py     ← Steps 11-19: Query pipeline
├── main.py             ← Orchestrator / CLI entry point
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key
```bash
# Copy the example and fill in your key
copy .env.example .env
# Edit .env and set: GEMINI_API_KEY=your_key_here
```
Get your API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🏃 Quick Start

### Step 1 – Add your documents
Drop `.pdf`, `.docx`, `.txt`, `.md`, or `.json` files into `data/raw/`.  
A sample file (`rag_overview.txt`) is already there to test with.

### Step 2 – Ingest (build the knowledge base)
```bash
python main.py --ingest
```
This runs steps 5–10: ETL → Chunking → Embeddings → FAISS index.

### Step 3 – Query (interactive chat)
```bash
python main.py --query
```

### Or do both in one command
```bash
python main.py --ingest --query
```

---

## 🔧 CLI Options

| Flag | Description |
|------|-------------|
| `--ingest` | Run ingestion pipeline (steps 1-10) |
| `--query` | Run query pipeline (steps 11-19) |
| `--strategy` | Chunking strategy: `semantic` (default), `fixed`, `recursive`, `hierarchical` |
| `--top-k` | Number of chunks to retrieve (default: 5) |
| `--question "..."` | Single question (non-interactive mode) |
| `--evaluate` | Enable LLM-as-a-judge evaluation after each answer |
| `--raw-dir` | Path to raw documents directory (default: `data/raw`) |

**Examples:**
```bash
# Use hierarchical chunking
python main.py --ingest --strategy hierarchical

# Ask a single question
python main.py --query --question "What is RAG?"

# Full pipeline with evaluation
python main.py --ingest --query --evaluate --top-k 7
```

---

## 🗺️ Pipeline Steps (from the diagram)

| # | Step | Implementation |
|---|------|----------------|
| 1 | Problem Statement | Define your use case |
| 2 | Feasibility Analysis | RAG vs Fine-tuning decision |
| 3 | Project Documents | POC, SLA, SOW |
| 4 | Data Collection | Add files to `data/raw/` |
| 5 | Data Processing (ETL) | `data_processing.py` |
| 6 | Save Cleaned Data | `data/cleaned/*.json` |
| 7 | Data Chunking | `chunking.py` (4 strategies) |
| 8 | Embeddings | `gemini-embedding-2` (768-dim) |
| 9 | Vector Database | FAISS (open-source) |
| 10 | RAG Knowledge Base | Persisted FAISS index |
| 11 | User Query | Interactive / CLI |
| 12 | Input Guardrails | Prompt injection + abuse detection |
| 13 | Query Routing | Simple → fast model, Complex → capable model |
| 14 | Retrieval Strategy | Hybrid (semantic + keyword) |
| 15 | Reranking | Hybrid score reranking |
| 16 | LLM Generation | Gemini with context + system prompt |
| 17 | Data Postprocessing | Format answer + add references |
| 18 | Output Guardrails | PII masking + harmful content filter |
| 19 | Evaluator | LLM-as-a-judge (relevance, faithfulness, coherence) |

---

## 🔌 Supported File Types

| Format | Parser |
|--------|--------|
| `.pdf` | PyPDF |
| `.docx` | python-docx |
| `.txt` / `.md` | Built-in |
| `.json` | Built-in |

---

## 🧠 Embedding Model

- **Model**: `gemini-embedding-2` (multimodal)
- **Dimensions**: 768 (recommended truncation from 3072)
- **Task types**: `RETRIEVAL_DOCUMENT` (ingestion), `RETRIEVAL_QUERY` (query time)
- **Similarity**: Cosine (via normalized inner product in FAISS)

---

## 📊 Evaluation Metrics

The LLM-as-a-judge evaluator scores on:
1. **Relevance** – Does the answer address the question?
2. **Faithfulness** – Is it grounded in retrieved context?
3. **Coherence** – Is it well-structured?

---

## 🛡️ Guardrails

**Input Guardrails:**
- Blocks prompt injections (`ignore instructions`, `jailbreak`, etc.)
- Rejects queries that are too short/long

**Output Guardrails:**
- Masks PII (email, phone, SSN, credit cards)
- Blocks harmful content

---

## 🚀 Deployment Notes

- Wrap into a FastAPI/Flask API for production
- Test endpoints with Postman
- Deploy on Cloud (Azure / GCP) with auto-scaling
- Monitor with LangSmith-equivalent logging (token usage, latency, cost)
- Set up alerts (AWS SNS / Azure Monitor)
