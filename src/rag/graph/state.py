from typing import TypedDict

from rag.retrieval import RetrievedChunk


class RAGState(TypedDict, total=False):
    """Shared state that every node reads from and writes to."""

    # input
    question: str                # the user's latest message, as typed
    history: list[dict]          # earlier turns: [{"role": "user"|"assistant", "content": "..."}]
    top_k: int                   # optional: chunks to retrieve (default: settings.top_k)
    arxiv_id: str | None         # optional: user-chosen paper; unlike paper_filter, never dropped

    # filled in by nodes
    route: str                   # "papers" | "chat" | "off_topic" (from rewrite)
    standalone: str              # the question in standalone form (from rewrite); never changed after
    query: str                   # search query: the standalone question, or a reformulation of it
    paper_filter: str | None     # arXiv id to restrict retrieval to, or None
    chunks: list[RetrievedChunk] # from retrieval (graded chunks from C2 on)
    answer: str
    retrieved: list[RetrievedChunk]   # before grading (chunks = after grading)
    retries: int                      # reformulate-and-search attempts so far (C3)
