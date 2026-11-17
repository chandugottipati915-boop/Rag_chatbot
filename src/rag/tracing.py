"""LangSmith tracing for chat turns. Everything here is a no-op unless LANGSMITH_TRACING=true.

LangGraph traces each graph node on its own, and the wrapped OpenAI client traces each LLM
call. These helpers group one whole chat turn under a single parent run:

    chat_stream                       <- start_turn()
    ├── LangGraph                     <- run inside under(turn)
    │   ├── rewrite ── ChatOpenAI     (only with history)
    │   ├── retrieve
    │   └── grade ──── ChatOpenAI
    └── generate ───── ChatOpenAI     <- api.main.stream_answer(..., langsmith_extra={"parent": turn})

The run's metadata carries `conversation_id`, which LangSmith uses to group a chat's
turns into one thread.
"""
from contextlib import contextmanager

import langsmith as ls
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled

from rag.config import settings


def turn_metadata(req) -> dict:
    meta = {
        "top_k": req.top_k,
        "arxiv_id": req.arxiv_id,
        "retrieval_mode": settings.retrieval_mode,
        "rerank": settings.rerank,
        "llm_model": settings.llm_model,
    }
    if req.conversation_id:
        meta["conversation_id"] = str(req.conversation_id)   # groups the chat as a thread
    return meta


def start_turn(name: str, req) -> RunTree | None:
    """Open the parent run for one chat turn, or return None when tracing is off."""
    if not tracing_is_enabled():
        return None
    run = RunTree(
        name=name,
        run_type="chain",
        inputs={"question": req.question, "history": [m.model_dump() for m in req.history]},
        extra={"metadata": turn_metadata(req)},
        tags=["api"],
    )
    run.post()
    return run


@contextmanager
def under(run: RunTree | None):
    """Make traced work inside this block (e.g. the LangGraph run) a child of `run`."""
    if run is None:
        yield
        return
    with ls.tracing_context(parent=run):
        yield


def end_turn(run: RunTree | None, outputs: dict | None = None, error: str | None = None) -> None:
    if run is None:
        return
    run.end(outputs=outputs, error=error)
    run.patch()

