from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

# NOTE: Keep it simple and local by default. We can wire OpenAI later if needed.
class EmbeddingFactory:
    _cache = {}

    @staticmethod
    def get(model_name: str):
        if model_name in EmbeddingFactory._cache:
            return EmbeddingFactory._cache[model_name]
        emb = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},  # cosine-ready
        )
        EmbeddingFactory._cache[model_name] = emb
        return emb

def simple_local_answer(question: str, contexts: List[str], max_chars: int = 3500) -> str:
    """
    Super-lightweight 'answer' that just stitches top-k retrieved chunks into a
    grounded scaffold. This keeps the project fully local and CPU-only.
    """
    joined = "\n\n---\n\n".join(contexts)
    ctx = joined[:max_chars]
    return (
        f"[Grounded Answer]\nQ: {question}\n\n"
        f"Context (truncated):\n{ctx}\n\n"
        "Draft: Based on the retrieved excerpts above, ..."
    )
