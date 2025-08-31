from pathlib import Path
from .utils import load_config
from .models import EmbeddingFactory
from langchain_community.vectorstores import FAISS

def main():
    cfg = load_config()
    emb = EmbeddingFactory.get(cfg["embedding_model"])
    vs = FAISS.load_local(str(Path(cfg["index_dir"])), emb, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": cfg["k"]})

    questions = [
        "Define SEI and its role in Li-ion batteries",
        "What limits rate capability in graphite anodes?",
        "Summarize diffusion vs. migration in electrolytes",
    ]
    for q in questions:
        docs = retriever.get_relevant_documents(q)
        print(f"\nQ: {q}\nTop sources:", [d.metadata.get("source","?") for d in docs])

if __name__ == "__main__":
    main()
