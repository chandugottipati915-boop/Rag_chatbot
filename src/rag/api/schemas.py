import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

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
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="Save this turn to a stored conversation. Its history is loaded "
                    "from the database and `history` is ignored.",
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
    database: bool = False      # Postgres reachable (chat history can be saved)


class IngestRequest(BaseModel):
    force: bool = Field(default=False, description="Re-index every PDF, even unchanged ones")


class IngestStatus(BaseModel):
    state: Literal["idle", "running", "done", "failed"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


# ------------------------------------------------------------------ auth

class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                       examples=["you@example.com"])
    password: str = Field(min_length=8, max_length=128, examples=["a-long-passphrase"])

    @field_validator("email", mode="before")
    @classmethod
    def _normalize_email(cls, v):
        # pasted emails often carry spaces; case doesn't matter in practice
        return v.strip().lower() if isinstance(v, str) else v


class UserOut(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    email: str
    created_at: datetime


# ------------------------------------------------------------------ chat history

class ConversationCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationSummary(BaseModel):
    model_config = {"from_attributes": True}     # build from SQLAlchemy objects

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    sources: list[Source] | None = None
    meta: dict | None = None
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut]
