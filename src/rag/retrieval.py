from dataclasses import dataclass

from rag.config import settings
from rag.embeddings import get_embedder
from rag import vectorstore as vs


@dataclass
class RetrievedChunk:
    text: str
    score: float
    arxiv_id: str
    title: str
    section: str
    page_start: int
    page_end: int

    @property
    def citation(self) -> str:
        pages = (f"p. {self.page_start}" if self.page_start == self.page_end
                 else f"pp. {self.page_start}-{self.page_end}")
        return f"{self.title} ({self.arxiv_id}), {self.section}, {pages}"


def _to_chunk(p) -> RetrievedChunk:
    return RetrievedChunk(
        text=p.payload["text"],
        score=p.score,
        arxiv_id=p.payload["arxiv_id"],
        title=p.payload["title"],
        section=p.payload["section"],
        page_start=p.payload["page_start"],
        page_end=p.payload["page_end"],
    )


def retrieve(
    question: str,
    top_k: int = settings.top_k,
    arxiv_id: str | None = None,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """Top-k chunks for a question.

    Score meaning depends on the settings: cosine similarity (dense), an RRF rank score
    (hybrid), or the cross-encoder's relevance score (rerank). Only compare scores, and
    set min_score, within one setup.
    """
    # with reranking, fetch a wider pool first and let the cross-encoder pick the best
    k = max(top_k, settings.rerank_candidates) if settings.rerank else top_k
    qvec = get_embedder().embed_query(question)
    chunks = [_to_chunk(p) for p in vs.search(qvec, top_k=k, arxiv_id=arxiv_id, query_text=question)]

    if settings.rerank and chunks:
        from rag.rerank import rerank_scores
        for c, s in zip(chunks, rerank_scores(question, [c.text for c in chunks])):
            c.score = s
        chunks.sort(key=lambda c: c.score, reverse=True)

    return [c for c in chunks[:top_k] if c.score >= min_score]
