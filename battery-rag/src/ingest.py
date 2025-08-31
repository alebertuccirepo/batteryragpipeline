import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from .utils import load_config, ensure_dir
from .models import EmbeddingFactory

def load_pdfs(pdf_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        loader = PyMuPDFLoader(str(pdf))  # better extraction than PyPDFLoader
        page_docs = loader.load()
        for d in page_docs:
            # --- light text cleanup to reduce OCR/encoding noise ---
            txt = d.page_content
            # remove soft-hyphen and common ligature glyphs
            txt = txt.replace("\u00ad", "")
            txt = txt.replace("\ufb01", "fi").replace("\ufb02", "fl")
            # drop other weird non-ASCII control characters (keep whitespace)
            txt = "".join(ch if (ord(ch) >= 32 or ch in "\n\t ") else " " for ch in txt)
            # collapse whitespace
            d.page_content = " ".join(txt.split())
            # add source metadata for traceability
            d.metadata = d.metadata or {}
            d.metadata["source"] = pdf.name
        docs.extend(page_docs)
    return docs

def chunk_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

def build_index():
    cfg = load_config()
    data_dir = Path(cfg["data_dir"])
    index_dir = Path(cfg["index_dir"])
    ensure_dir(index_dir)

    print(f"[*] Loading PDFs from {data_dir} ...")
    docs = load_pdfs(data_dir)
    if not docs:
        raise SystemExit(f"No PDFs found in {data_dir}. Put files there and re-run.")

    print(f"[*] Chunking {len(docs)} docs ...")
    chunks = chunk_docs(docs, cfg["chunk_size"], cfg["chunk_overlap"])
    print(f"[+] Created {len(chunks)} chunks")

    emb = EmbeddingFactory.get(cfg["embedding_model"])

    print("[*] Building FAISS index (this may take a moment)...")
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(str(index_dir))
    print(f"[+] Saved FAISS index to {index_dir}")

if __name__ == "__main__":
    print("[INGEST] starting...")
    try:
        build_index()
        print("[INGEST] done.")
    except Exception as e:
        import traceback, sys
        print("[INGEST] ERROR:", e, file=sys.stderr)
        traceback.print_exc()