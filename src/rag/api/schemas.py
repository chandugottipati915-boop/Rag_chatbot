from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from rag.retrieval import RetrievedChunk


class Message(BaseModel):
    """One earlier turn of the conversation."""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000,
                          examples=["How does Self-RAG decide when to retrieve?"])
    top_k: int = Field(default=5, ge=1, le=20)
    arxiv_id: str | None = Field(default=None, description="Only search this paper",
                                 examples=[None])
    history: list[Message] = Field(
        default_factory=list, max_length=20,
        description="Earlier turns, oldest first. Lets follow-ups like 'How is it trained?' work.",
        examples=[[]],
    )


class Source(BaseModel):
    n: int                      # the citation number used in the answer: [1], [2], ...
    arxiv_id: str
    title: str
    section: str
    page_start: int
    page_end: int
    score: float
    citation: str
    text: str | None = None     # filled by /search, left out of /chat to keep responses small

    @classmethod
    def from_chunk(cls, n: int, c: RetrievedChunk, include_text: bool = False) -> "Source":
        return cls(
            n=n, arxiv_id=c.arxiv_id, title=c.title, section=c.section,
            page_start=c.page_start, page_end=c.page_end,
            score=round(c.score, 4), citation=c.citation,
            text=c.text if include_text else None,
        )


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    latency_ms: int
    rewritten_query: str | None = None   # set when a follow-up was rewritten or a retry reformulated it
    retries: int = 0                     # reformulate-and-search attempts used


class SearchResponse(BaseModel):
    question: str
    sources: list[Source]


class Paper(BaseModel):
    arxiv_id: str
    title: str
    published: str | None = None
    url: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: bool
    collection: str
    points: int | None


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Re-index every PDF, even unchanged ones")


class IngestStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
