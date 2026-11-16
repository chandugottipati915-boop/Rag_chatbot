from typing import TypedDict

from rag.retrieval import RetrievedChunk


class RAGState(TypedDict, total=False):
    """Shared state that every node reads from and writes to."""

    # input
    question: str                # the user's latest message, as typed
    history: list[dict]          # earlier turns: [{"role": "user"|"assistant", "content": "..."}]

    # filled in by nodes
    query: str                   # standalone search query (from rewrite)
    paper_filter: str | None     # arXiv id to restrict retrieval to, or None
    chunks: list[RetrievedChunk] # from retrieval (graded chunks from C2 on)
    answer: str
    retrieved: list[RetrievedChunk]   # before grading (chunks = after grading)
    retries: int                      # reformulate-and-search attempts so far (C3)
