"""FastAPI backend for the RAG chatbot.

Run:  uv run uvicorn rag.api.main:app --app-dir src --reload
Docs: http://localhost:8000/docs
"""

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from openai import OpenAIError

from rag import vectorstore as vs
from rag.config import settings
from rag.api.schemas import (
    ChatRequest, ChatResponse, HealthResponse, IngestRequest,
    IngestStatus, SearchResponse, Source,
)
from rag.embeddings import get_embedder
from rag.generation import generate_stream
from rag.graph.build import get_graph, get_retrieval_graph
from rag.graph.nodes import answer_question
from rag.retrieval import RetrievedChunk, retrieve

log = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the embedding model once at startup instead of on the first request
    try:
        get_embedder()
        if settings.rerank:
            from rag.rerank import get_reranker
            get_reranker()      # load the cross-encoder at startup too
    except Exception:
        log.exception("Could not load the embedding/reranker model at startup")
    yield


app = FastAPI(title="RAG Chatbot API", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # Next.js dev server (Phase 4)
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ helpers

def get_chunks(req: ChatRequest) -> list[RetrievedChunk]:
    """Retrieval with one consistent error: 503 if Qdrant or the embedder fails."""
    try:
        return retrieve(req.question, top_k=req.top_k, arxiv_id=req.arxiv_id)
    except Exception as e:
        log.exception("Retrieval failed")
        raise HTTPException(status_code=503, detail="Search backend unavailable") from e


def to_sources(chunks: list[RetrievedChunk], include_text: bool = False) -> list[Source]:
    return [Source.from_chunk(i, c, include_text) for i, c in enumerate(chunks, 1)]


def sse(event: str, data) -> str:
    """Format one Server-Sent Event. JSON-encode data so newlines in tokens don't break it."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def graph_input(req: ChatRequest) -> dict:
    return {
        "question": req.question,
        "history": [m.model_dump() for m in req.history],
        "top_k": req.top_k,
        "arxiv_id": req.arxiv_id,
    }


def run_graph(graph, req: ChatRequest) -> dict:
    """Run a compiled graph with consistent errors.

    Rewrite, grade and reformulate already fall back on their own when the LLM fails,
    so an OpenAIError reaching here comes from generation (502). Anything else is
    retrieval: Qdrant, the embedder or the reranker (503).
    """
    try:
        return graph.invoke(graph_input(req))
    except OpenAIError as e:
        log.exception("LLM call failed")
        raise HTTPException(status_code=502, detail="LLM provider error") from e
    except Exception as e:
        log.exception("Graph failed (retrieval)")
        raise HTTPException(status_code=503, detail="Search backend unavailable") from e


def graph_info(req: ChatRequest, state: dict) -> dict:
    query = state.get("query")
    return {"rewritten_query": query if query and query != req.question else None,
            "retries": state.get("retries", 0)}


# ------------------------------------------------------------------ routes
# Routes are plain `def`: retrieval and OpenAI calls block, so FastAPI runs
# them in a thread pool and one slow request doesn't freeze the server.

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Is Qdrant reachable, and how many chunks are indexed?"""
    try:
        exists = vs.client.collection_exists(vs.COLL)
        points = vs.client.count(vs.COLL).count if exists else 0
    except Exception:
        log.exception("Health check: Qdrant unreachable")
        return HealthResponse(status="degraded", qdrant=False, collection=vs.COLL, points=None)
    return HealthResponse(status="ok" if points > 0 else "degraded",
                          qdrant=True, collection=vs.COLL, points=points)


@app.post("/search", response_model=SearchResponse, tags=["rag"])
def search(req: ChatRequest) -> SearchResponse:
    """Retrieved chunks only, no LLM call. Useful for debugging retrieval."""
    chunks = get_chunks(req)
    return SearchResponse(question=req.question, sources=to_sources(chunks, include_text=True))


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(req: ChatRequest) -> ChatResponse:
    """Full cited answer in one JSON response.

    Runs the graph: rewrite (with history) → retrieve → grade → (reformulate once)
    → generate, or abstain when nothing relevant is found.
    """
    t0 = time.perf_counter()
    state = run_graph(get_graph(), req)
    return ChatResponse(
        question=req.question,
        answer=state["answer"],
        sources=to_sources(state.get("chunks", [])),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        **graph_info(req, state),
    )


@app.post("/chat/stream", tags=["rag"])
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Streamed answer as Server-Sent Events, in this order:

    - `sources`: graded sources (sent first so a UI can show them right away; empty if abstaining)
    - `token`:   a piece of answer text (many of these)
    - `done`:    {"latency_ms", "rewritten_query", "retries"}
    - `error`:   {"message": ...} if the LLM fails mid-stream
    """
    t0 = time.perf_counter()
    # rewrite → retrieve → grade (→ reformulate) runs before streaming starts,
    # so retrieval errors still return a normal 503
    state = run_graph(get_retrieval_graph(), req)
    chunks = state.get("chunks", [])

    def events() -> Iterator[str]:
        yield sse("sources", [s.model_dump() for s in to_sources(chunks)])
        if "answer" in state:                       # the graph abstained: nothing to generate
            yield sse("token", state["answer"])
        else:
            try:
                for piece in generate_stream(answer_question(state), chunks):
                    yield sse("token", piece)
            except OpenAIError:
                # the 200 status is already sent, so report the failure as an event
                log.exception("LLM stream failed")
                yield sse("error", {"message": "LLM provider error"})
                return
        yield sse("done", {"latency_ms": int((time.perf_counter() - t0) * 1000),
                           **graph_info(req, state)})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ ingestion

_ingest = IngestStatus(state="idle")
_ingest_lock = threading.Lock()


def _run_ingest(force: bool) -> None:
    global _ingest
    from rag.ingestion.pipeline import run   # heavy imports (unstructured) only when needed
    try:
        run(force=force)
        _ingest = _ingest.model_copy(update={
            "state": "done", "finished_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        log.exception("Ingestion failed")
        _ingest = _ingest.model_copy(update={
            "state": "failed", "finished_at": datetime.now(timezone.utc), "error": str(e),
        })


@app.post("/ingest", response_model=IngestStatus, status_code=202, tags=["admin"])
def start_ingest(req: IngestRequest) -> IngestStatus:
    """Start ingesting data/raw in the background. Poll GET /ingest for progress."""
    global _ingest
    with _ingest_lock:
        if _ingest.state == "running":
            raise HTTPException(status_code=409, detail="Ingestion is already running")
        _ingest = IngestStatus(state="running", started_at=datetime.now(timezone.utc))
    threading.Thread(target=_run_ingest, args=(req.force,), daemon=True).start()
    return _ingest


@app.get("/ingest", response_model=IngestStatus, tags=["admin"])
def ingest_status() -> IngestStatus:
    """Current ingestion state: idle, running, done or failed."""
    return _ingest
