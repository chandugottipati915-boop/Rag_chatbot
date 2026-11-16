"""Cross-encoder reranking.

The retriever compares the question and each chunk as separate vectors: fast, but coarse.
A cross-encoder reads the question and the chunk together and scores how well the chunk
answers the question: slower, but much better at putting the right chunk first.
So we retrieve more candidates than we need, rerank them, and keep the best top_k.
"""

from functools import lru_cache

from rag.config import settings


@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder   # heavy import, only when reranking is on
    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank_scores(question: str, texts: list[str]) -> list[float]:
    """Relevance score for each text (higher = more relevant)."""
    if not texts:
        return []
    scores = get_reranker().predict([(question, t) for t in texts], show_progress_bar=False)
    return [float(s) for s in scores]
