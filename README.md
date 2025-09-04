# Battery RAG Pipeline

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline for querying battery modeling and degradation research papers.  
Built with **LangChain**, **FAISS**, and **HuggingFace embeddings**, with both a **CLI** and a **FastAPI server** for Q&A.

---

## Features
- **Ingest PDFs** from `data/battery_papers/`
- **Chunking** with `RecursiveCharacterTextSplitter`
- **Semantic search** using FAISS vector index
- **Embeddings** via HuggingFace (MiniLM by default)
- **Reranker** (MMR + battery-term keyword boost) for better retrieval
- **CLI tool** (`python -m src.query "your question"`)
- **API server** (`uvicorn src.serve:app --reload`) with `/query` endpoint

---

## 🛠️ Quickstart

### 1. Setup environment
```bash
git clone https://github.com/alebertuccirepo/battery-rag-pipeline.git
cd battery-rag-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # on Windows
pip install -r requirements.txt
