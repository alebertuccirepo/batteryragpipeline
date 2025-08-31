from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from langchain_community.vectorstores import FAISS
from .utils import load_config
from .models import EmbeddingFactory, simple_local_answer

app = FastAPI(title="Battery RAG", version="0.1.0")
_cfg = load_config()
_emb = EmbeddingFactory.get(_cfg["embedding_model"])
_index_dir = Path(_cfg["index_dir"])
_vs = FAISS.load_local(str(_index_dir), _emb, allow_dangerous_deserialization=True)
_retriever = _vs.as_retriever(search_kwargs={"k": _cfg["k"]})

class QueryIn(BaseModel):
    question: str

class QueryOut(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryOut)
def query(payload: QueryIn):
    if not payload.question.strip():
        raise HTTPException(400, "Empty question")
    docs = _retriever.get_relevant_documents(payload.question)
    ctxs = [d.page_content for d in docs]
    ans = simple_local_answer(payload.question, ctxs, _cfg["max_context_chars"])
    return QueryOut(answer=ans, sources=[d.metadata.get("source","?") for d in docs])

@app.get("/health")
def health():
    return {"status": "ok"}
