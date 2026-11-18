"""FastAPI backend for the RAG chatbot.

Run:  uv run uvicorn rag.api.main:app --app-dir src --reload
Docs: http://localhost:8000/docs
"""

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
import langsmith as ls
from openai import OpenAIError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from rag import vectorstore as vs
from rag.config import settings
from rag.api.schemas import (
    ChatRequest, ChatResponse, ConversationCreate, Credentials, UserOut, ConversationDetail, ConversationRename,
    ConversationSummary, HealthResponse, IngestRequest, IngestStatus, Paper,
    SearchResponse, Source,
)
from rag.auth import (
    authenticate, clear_session_cookie, current_user, hash_password, login_limiter,
    set_session_cookie,
)
from rag.db import repo
from rag.db.models import User
from rag.db.session import SessionLocal, get_db
from rag.embeddings import get_embedder
from rag.generation import generate_stream
from rag.graph.build import get_graph, get_retrieval_graph
from rag.graph.nodes import answer_question
from rag.retrieval import RetrievedChunk, retrieve
from rag.tracing import end_turn, start_turn, under

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


app = FastAPI(title="RAG Chatbot API", version="0.6.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,    # the Next.js UI; must be exact when cookies are sent
    allow_credentials=True,                 # let the browser send the session cookie
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


def with_db_history(req: ChatRequest, db: Session) -> ChatRequest:
    """For a stored conversation the database is the source of truth: replace the
    request's history with the saved turns (validated, so they become Message objects)."""
    return ChatRequest.model_validate(
        {**req.model_dump(), "history": repo.recent_history(db, req.conversation_id)})


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


@ls.traceable(name="generate", run_type="chain",
              process_inputs=lambda i: {"question": i["question"], "passages": len(i["chunks"])})
def stream_answer(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    """generate_stream as a LangSmith child run (call with langsmith_extra={"parent": turn}).

    A traceable generator keeps its tracing context across yields; a `with` block around
    the streaming loop can't, because each piece may be produced in a different thread."""
    yield from generate_stream(question, chunks)


# ------------------------------------------------------------------ routes
# Routes are plain `def`: retrieval and OpenAI calls block, so FastAPI runs
# them in a thread pool and one slow request doesn't freeze the server.

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


def database_ok() -> bool:
    try:
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
        return True
    except Exception:
        log.warning("Health check: database unreachable")
        return False


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Is Qdrant reachable, how many chunks are indexed, and can chats be saved?

    `status` reflects search only: answering still works without the database.
    """
    database = database_ok()
    try:
        exists = vs.client.collection_exists(vs.COLL)
        points = vs.client.count(vs.COLL).count if exists else 0
    except Exception:
        log.exception("Health check: Qdrant unreachable")
        return HealthResponse(status="degraded", qdrant=False, collection=vs.COLL, points=None,
                              database=database)
    return HealthResponse(status="ok" if points > 0 else "degraded",
                          qdrant=True, collection=vs.COLL, points=points, database=database)


@app.get("/papers", response_model=list[Paper], tags=["rag"])
def papers() -> list[Paper]:
    """The indexed papers (from data/raw/catalog.json), for the UI's paper filter."""
    try:
        data = json.loads((settings.raw_dir / "catalog.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return sorted(
        (Paper(arxiv_id=pid, title=m.get("title", pid), published=m.get("published"),
               url=f"https://arxiv.org/abs/{pid}") for pid, m in data.items()),
        key=lambda p: p.title.lower(),
    )


@app.post("/search", response_model=SearchResponse, tags=["rag"])
def search(req: ChatRequest, user: User = Depends(current_user)) -> SearchResponse:
    """Retrieved chunks only, no LLM call. Useful for debugging retrieval."""
    chunks = get_chunks(req)
    return SearchResponse(question=req.question, sources=to_sources(chunks, include_text=True))


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(req: ChatRequest, user: User = Depends(current_user),
         db: Session = Depends(get_db)) -> ChatResponse:
    """Full cited answer in one JSON response.

    Runs the graph: rewrite (with history) → retrieve → grade → (reformulate once)
    → generate, or abstain when nothing relevant is found.
    """
    t0 = time.perf_counter()
    cid = req.conversation_id
    if cid:
        conversation_or_404(db, cid, user)
        req = with_db_history(req, db)
    turn = start_turn("chat", req, user.id)               # LangSmith parent run (None when tracing is off)
    try:
        with under(turn):
            state = run_graph(get_graph(), req)
    except HTTPException as e:
        end_turn(turn, error=f"{e.status_code}: {e.detail}")
        raise
    response = ChatResponse(
        question=req.question,
        answer=state["answer"],
        sources=to_sources(state.get("chunks", [])),
        latency_ms=int((time.perf_counter() - t0) * 1000),
        **graph_info(req, state),
    )
    if cid:
        repo.add_message(db, cid, "user", req.question)
        repo.add_message(db, cid, "assistant", response.answer,
                         sources=[s.model_dump() for s in to_sources(state.get("chunks", []), include_text=True)],
                         meta={"latency_ms": response.latency_ms, "rewritten_query": response.rewritten_query,
                               "retries": response.retries})
    end_turn(turn, outputs={"answer": response.answer, **graph_info(req, state)})
    return response


@app.post("/chat/stream", tags=["rag"])
def chat_stream(req: ChatRequest, user: User = Depends(current_user),
                db: Session = Depends(get_db)) -> StreamingResponse:
    """Streamed answer as Server-Sent Events, in this order:

    - `sources`: graded sources (sent first so a UI can show them right away; empty if abstaining)
    - `token`:   a piece of answer text (many of these)
    - `done`:    {"latency_ms", "rewritten_query", "retries"}
    - `error`:   {"message": ...} if the LLM fails mid-stream
    """
    t0 = time.perf_counter()
    cid = req.conversation_id
    if cid:
        conversation_or_404(db, cid, user)
        req = with_db_history(req, db)

    # rewrite → retrieve → grade (→ reformulate) runs before streaming starts,
    # so retrieval errors still return a normal 503
    turn = start_turn("chat_stream", req, user.id)        # LangSmith parent run (None when tracing is off)
    try:
        with under(turn):
            state = run_graph(get_retrieval_graph(), req)
    except HTTPException as e:
        end_turn(turn, error=f"{e.status_code}: {e.detail}")
        raise
    chunks = state.get("chunks", [])
    sources = [s.model_dump() for s in to_sources(chunks, include_text=True)]
    if cid:
        repo.add_message(db, cid, "user", req.question)   # saved once retrieval has worked

    def save_answer(text: str, meta: dict) -> None:
        # A new session: the request's session (`db`) is closed once this function
        # returns, but the stream keeps running after that.
        if not cid or not text:
            return
        with SessionLocal() as s:
            repo.add_message(s, cid, "assistant", text, sources=sources, meta=meta)

    def events() -> Iterator[str]:
        answer: list[str] = []
        saved = False
        try:
            # passage text included so the UI can show what each citation says
            yield sse("sources", sources)
            if "answer" in state:                       # the graph abstained: nothing to generate
                answer.append(state["answer"])
                yield sse("token", state["answer"])
            else:
                try:
                    for piece in stream_answer(answer_question(state), chunks,
                                               langsmith_extra={"parent": turn}):
                        answer.append(piece)
                        yield sse("token", piece)
                except OpenAIError:
                    # the 200 status is already sent, so report the failure as an event
                    log.exception("LLM stream failed")
                    end_turn(turn, error="LLM provider error")
                    yield sse("error", {"message": "LLM provider error"})
                    return
            done = {"latency_ms": int((time.perf_counter() - t0) * 1000), **graph_info(req, state)}
            save_answer("".join(answer), done)      # save before `done`, so a reload right after finds it
            saved = True
            end_turn(turn, outputs={"answer": "".join(answer), **done})
            yield sse("done", done)
        finally:
            # the client pressed Stop (disconnected) or the LLM failed: keep the partial answer
            if not saved:
                save_answer("".join(answer), {"stopped": True})
                if turn is not None and turn.end_time is None:
                    end_turn(turn, outputs={"answer": "".join(answer)}, error="stopped by the client")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------ auth

def _login_key(request: Request, email: str) -> str:
    return f"{request.client.host if request.client else '?'}:{repo.normalize_email(email)}"


@app.post("/auth/signup", response_model=UserOut, status_code=201, tags=["auth"])
def signup(body: Credentials, response: Response, db: Session = Depends(get_db)):
    """Create an account and sign in (sets the session cookie)."""
    if repo.get_user_by_email(db, body.email):
        raise HTTPException(status_code=409, detail="An account with this email already exists. Sign in instead.")
    try:
        user = repo.create_user(db, body.email, hash_password(body.password))
    except IntegrityError:                      # two sign-ups with the same email at once
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists. Sign in instead.")
    set_session_cookie(response, user.id)
    return user


@app.post("/auth/login", response_model=UserOut, tags=["auth"])
def login(body: Credentials, request: Request, response: Response, db: Session = Depends(get_db)):
    """Sign in (sets the session cookie). Same error for an unknown email and a wrong
    password, and 429 after 5 failures in 15 minutes."""
    key = _login_key(request, body.email)
    login_limiter.check(key)
    user = authenticate(db, body.email, body.password)
    if user is None:
        login_limiter.failed(key)
        raise HTTPException(status_code=401, detail="Wrong email or password.")
    login_limiter.succeeded(key)
    set_session_cookie(response, user.id)
    return user


@app.post("/auth/logout", status_code=204, tags=["auth"])
def logout():
    """Sign out: the browser drops the session cookie."""
    response = Response(status_code=204)
    clear_session_cookie(response)
    return response


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: User = Depends(current_user)):
    """Who is signed in (401 if nobody). The UI calls this on load."""
    return user


# ------------------------------------------------------------------ chat history

def conversation_or_404(db: Session, conversation_id: uuid.UUID, user: User):
    """The user's own chat, or 404: someone else's chat looks the same as a missing one."""
    conv = repo.get_conversation(db, conversation_id, user.id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@app.get("/conversations", response_model=list[ConversationSummary], tags=["history"])
def list_conversations(user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Your saved chats, most recently active first."""
    return repo.list_conversations(db, user.id)


@app.post("/conversations", response_model=ConversationSummary, status_code=201, tags=["history"])
def create_conversation(body: ConversationCreate, user: User = Depends(current_user),
                        db: Session = Depends(get_db)):
    """Start a new, empty chat. Send its id as `conversation_id` to /chat/stream."""
    return repo.create_conversation(db, body.title, user.id)


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["history"])
def get_conversation(conversation_id: uuid.UUID, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    """One of your chats with all its messages, oldest first."""
    return conversation_or_404(db, conversation_id, user)


@app.patch("/conversations/{conversation_id}", response_model=ConversationSummary, tags=["history"])
def rename_conversation(conversation_id: uuid.UUID, body: ConversationRename,
                        user: User = Depends(current_user), db: Session = Depends(get_db)):
    return repo.rename_conversation(db, conversation_or_404(db, conversation_id, user), body.title)


@app.delete("/conversations/{conversation_id}", status_code=204, tags=["history"])
def delete_conversation(conversation_id: uuid.UUID, user: User = Depends(current_user),
                        db: Session = Depends(get_db)):
    """Delete one of your chats and its messages."""
    repo.delete_conversation(db, conversation_or_404(db, conversation_id, user))
    return Response(status_code=204)


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
def start_ingest(req: IngestRequest, user: User = Depends(current_user)) -> IngestStatus:
    """Start ingesting data/raw in the background. Poll GET /ingest for progress."""
    global _ingest
    with _ingest_lock:
        if _ingest.state == "running":
            raise HTTPException(status_code=409, detail="Ingestion is already running")
        _ingest = IngestStatus(state="running", started_at=datetime.now(timezone.utc))
    threading.Thread(target=_run_ingest, args=(req.force,), daemon=True).start()
    return _ingest


@app.get("/ingest", response_model=IngestStatus, tags=["admin"])
def ingest_status(user: User = Depends(current_user)) -> IngestStatus:
    """Current ingestion state: idle, running, done or failed."""
    return _ingest
