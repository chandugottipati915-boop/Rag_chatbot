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


def retrieve(
    question: str,
    top_k: int = settings.top_k,
    arxiv_id: str | None = None,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    qvec = get_embedder().embed_query(question)
    points = vs.search(qvec, top_k=top_k, arxiv_id=arxiv_id)
    return [
        RetrievedChunk(
            text=p.payload["text"],
            score=p.score,
            arxiv_id=p.payload["arxiv_id"],
            title=p.payload["title"],
            section=p.payload["section"],
            page_start=p.payload["page_start"],
            page_end=p.payload["page_end"],
        )
        for p in points
        if p.score >= min_score
    ]