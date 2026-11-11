"""API tests. Retrieval, the LLM and ingestion are faked, so no Qdrant,
embedding model or OpenAI key is needed.

Run: uv run pytest -q
"""

import json
import time

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

import rag.api.main as api
from rag.api.schemas import IngestStatus
from rag.retrieval import RetrievedChunk

CHUNK = RetrievedChunk(
    text="RAG-Token can use a different document for each token.", score=0.81,
    arxiv_id="2005.11401", title="Retrieval-Augmented Generation", section="2.1 Models",
    page_start=3, page_end=3,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "retrieve", lambda q, top_k, arxiv_id=None: [CHUNK])
    monkeypatch.setattr(api, "generate", lambda q, chunks: "It uses per-token documents [1].")
    monkeypatch.setattr(api, "generate_stream", lambda q, chunks: iter(["It uses ", "per-token documents [1]."]))
    monkeypatch.setattr(api, "_ingest", IngestStatus(state="idle"))
    return TestClient(api.app)   # no `with`: skips startup, so the embedding model isn't loaded


def parse_sse(body: str) -> list[tuple[str, object]]:
    """Turn an SSE response body into [(event, data), ...]."""
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


# ---------------------------------------------------------------- /chat

def test_chat_returns_answer_and_numbered_sources(client):
    r = client.post("/chat", json={"question": "What is RAG-Token?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "It uses per-token documents [1]."
    assert body["sources"][0]["n"] == 1
    assert body["sources"][0]["arxiv_id"] == "2005.11401"
    assert body["sources"][0]["text"] is None          # chat responses stay small
    assert isinstance(body["latency_ms"], int)


@pytest.mark.parametrize("payload", [
    {},                                  # missing question
    {"question": ""},                    # empty question
    {"question": "x" * 2001},            # too long
    {"question": "x", "top_k": 0},       # top_k too small
    {"question": "x", "top_k": 21},      # top_k too large
])
def test_chat_rejects_bad_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_llm_failure_is_502(client, monkeypatch):
    def fail(q, chunks):
        raise OpenAIError("provider down")
    monkeypatch.setattr(api, "generate", fail)
    r = client.post("/chat", json={"question": "x"})
    assert r.status_code == 502
    assert "provider down" not in r.text               # internal details stay in the logs


def test_chat_retrieval_failure_is_503(client, monkeypatch):
    def down(q, top_k, arxiv_id=None):
        raise ConnectionError("qdrant unreachable")
    monkeypatch.setattr(api, "retrieve", down)
    assert client.post("/chat", json={"question": "x"}).status_code == 503


# ---------------------------------------------------------------- /search

def test_search_returns_chunk_text_and_no_answer(client):
    r = client.post("/search", json={"question": "RAG-Token", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" not in body
    assert body["sources"][0]["text"] == CHUNK.text


# ---------------------------------------------------------------- /chat/stream

def test_stream_sends_sources_then_tokens_then_done(client):
    r = client.post("/chat/stream", json={"question": "What is RAG-Token?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [name for name, _ in events] == ["sources", "token", "token", "done"]
    assert events[0][1][0]["arxiv_id"] == "2005.11401"
    assert "".join(d for name, d in events if name == "token") == "It uses per-token documents [1]."


def test_stream_reports_llm_error_as_event(client, monkeypatch):
    def broken(q, chunks):
        yield "partial "
        raise OpenAIError("rate limited")
    monkeypatch.setattr(api, "generate_stream", broken)
    events = parse_sse(client.post("/chat/stream", json={"question": "x"}).text)
    assert events[-1][0] == "error"


# ---------------------------------------------------------------- /health

def test_health_degraded_when_qdrant_down(client, monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError
    monkeypatch.setattr(api.vs.client, "collection_exists", down)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "degraded", "qdrant": False,
                        "collection": api.vs.COLL, "points": None}


# ---------------------------------------------------------------- /ingest

def test_ingest_runs_in_background_and_blocks_second_start(client, monkeypatch):
    def fake_ingest(force):
        time.sleep(0.2)
        api._ingest = api._ingest.model_copy(update={"state": "done"})
    monkeypatch.setattr(api, "_run_ingest", fake_ingest)

    first = client.post("/ingest", json={"force": False})
    assert first.status_code == 202 and first.json()["state"] == "running"
    assert client.post("/ingest", json={}).status_code == 409

    time.sleep(0.4)
    assert client.get("/ingest").json()["state"] == "done"
