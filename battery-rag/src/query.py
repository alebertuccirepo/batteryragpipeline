import sys
from pathlib import Path
from rich import print
from langchain_community.vectorstores import FAISS
from .utils import load_config
from .models import EmbeddingFactory, simple_local_answer


def main():
    if len(sys.argv) < 2:
        print("[red]Usage: python -m src.query \"your question here\"[/red]")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"[bold]Question:[/bold] {question}")

    cfg = load_config()
    index_dir = Path(cfg["index_dir"])
    emb = EmbeddingFactory.get(cfg["embedding_model"])

    # Load FAISS index
    vs = FAISS.load_local(str(index_dir), emb, allow_dangerous_deserialization=True)

    # Retriever with MMR
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": cfg["k"], "fetch_k": 30, "lambda_mult": 0.5},
    )
    docs = retriever.get_relevant_documents(question)

    # Keyword rerank
    BATTERY_TERMS = [
        "SEI", "solid electrolyte interphase", "lithium", "capacity fade",
        "anode", "cathode", "plating", "electrolyte", "graphite", "impedance"
    ]

    def rerank_by_keywords(docs):
        def score(text: str) -> int:
            lo = text.lower()
            return sum(1 for t in BATTERY_TERMS if t.lower() in lo)
        return sorted(docs, key=lambda d: score(d.page_content), reverse=True)

    docs = rerank_by_keywords(docs)[: cfg["k"]]

    if not docs:
        print("[red]No documents retrieved. Did you run ingest?[/red]")
        return

    ctxs = [d.page_content for d in docs]
    answer = simple_local_answer(question, ctxs, max_chars=cfg["max_context_chars"])

    print("\n[bold cyan]Answer[/bold cyan]\n")
    print(answer)

    print("\n[bold yellow]Sources[/bold yellow]")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source','?')}")


if __name__ == "__main__":
    main()
