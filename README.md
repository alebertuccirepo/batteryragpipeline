# Battery RAG Pipeline

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline for querying battery modeling and degradation research papers.  
Built with **LangChain**, **FAISS**, and **HuggingFace embeddings**, with both a **CLI** and a **FastAPI server** for Q&A.

---

## Features
- Ingest PDFs from `data/battery_papers/`
- Chunking with `RecursiveCharacterTextSplitter`
- Semantic search using FAISS vector index
- Embeddings via HuggingFace (MiniLM by default)
- Reranker (MMR + battery-term keyword boost) for better retrieval
- CLI tool (`python -m src.query "your question"`)
- API server (`uvicorn src.serve:app --reload`) with `/query` endpoint

---

## Quickstart

### 1. Setup environment
```bash
git clone https://github.com/alebertuccirepo/battery-rag-pipeline.git
cd battery-rag-pipeline
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # on Windows
pip install -r requirements.txt
2. Run CLI
bash
Copy code
python -m src.query "Explain SEI formation"
3. Run API server
bash
Copy code
uvicorn src.serve:app --reload
```
## Roadmap (Future Additions)

UI/UX: Add a lightweight UI for interactive querying

Hybrid retrieval: Combine FAISS (dense) with BM25 (sparse) for better recall

Configurable ingestion: YAML/JSON configs to set chunk size, embeddings model, filters

Evaluation harness: Add unit tests with sample Q/A pairs for quality tracking

Docker support: Containerize pipeline for easy setup and sharing

Multi-modal ingestion: Support tables, figures, and CSV data alongside text

Domain-specific finetuning: Experiment with finetuned LLMs on electrochemistry/battery
