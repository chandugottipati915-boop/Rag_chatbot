"""API tests. Retrieval, the LLM and ingestion are faked, so no Qdrant,
embedding model or OpenAI key is needed.

/chat and /chat/stream run the LangGraph pipeline, so the fakes sit at the graph level:
retrieval and generation inside the nodes, and one fake LLM that answers the rewrite,
grade and reformulate calls by recognising their prompts.

Run: uv run pytest -q
"""

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

import rag.api.main as api
import rag.graph.nodes as nodes
from rag.api.schemas import IngestStatus
from rag.generation import NO_ANSWER
from rag.retrieval import RetrievedChunk

CHUNK = RetrievedChunk(
    text="RAG-Token can use a different document for each token.", score=0.81,
    arxiv_id="2005.11401", title="Retrieval-Augmented Generation", section="2.1 Models",
    page_start=3, page_end=3,
)
HISTORY = [{"role": "user", "content": "What is RAG?"},
           {"role": "assistant", "content": "RAG combines a retriever with a generator [1]."}]


class FakeLLM:
    """Answers each graph LLM call based on which prompt it receives."""

    def __init__(self):
        self.relevant = [1]                                     # grader keeps chunk 1
        self.rewrite = {"query": "How does RAG-Token work?", "paper": "2005.11401"}
        self.calls = []
        self.chat = self.completions = self

    def create(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "standalone search query" in prompt:
            kind, reply = "rewrite", self.rewrite
        elif "found nothing relevant" in prompt:
            kind, reply = "reformulate", {"query": "RAG-Token marginalization"}
        else:
            kind, reply = "grade", {"relevant": self.relevant}
        self.calls.append(kind)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])


@pytest.fixture
def llm(monkeypatch):
    fake = FakeLLM()
    monkeypatch.setattr(nodes, "get_client", lambda: fake)
    # the rewrite only accepts paper ids that exist in the catalog (data/raw/catalog.json)
    monkeypatch.setattr(nodes, "paper_catalog", lambda: {"2005.11401": "RAG", "2310.11511": "Self-RAG"})
    return fake


@pytest.fixture
def retrieved(monkeypatch):
    calls = []

    def fake_retrieve(q, top_k=5, arxiv_id=None):
        calls.append({"query": q, "top_k": top_k, "arxiv_id": arxiv_id})
        return [CHUNK]

    monkeypatch.setattr(nodes, "do_retrieve", fake_retrieve)     # used by the graph
    monkeypatch.setattr(api, "retrieve", fake_retrieve)          # used by /search
    return calls


@pytest.fixture
def client(monkeypatch, llm, retrieved):
    monkeypatch.setattr(nodes, "llm_generate", lambda q, chunks: "It uses per-token documents [1].")
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
    {"question": "x", "history": [{"role": "system", "content": "hi"}]},     # unknown role
    {"question": "x", "history": [{"role": "user", "content": ""}]},         # empty turn
    {"question": "x", "history": [{"role": "user", "content": "q"}] * 21},   # too many turns
])
def test_chat_rejects_bad_input(client, payload):
    assert client.post("/chat", json=payload).status_code == 422


def test_chat_llm_failure_is_502(client, monkeypatch):
    def fail(q, chunks):
        raise OpenAIError("provider down")
    monkeypatch.setattr(nodes, "llm_generate", fail)
    r = client.post("/chat", json={"question": "x"})
    assert r.status_code == 502
    assert "provider down" not in r.text               # internal details stay in the logs


def test_chat_retrieval_failure_is_503(client, monkeypatch):
    def down(q, top_k=5, arxiv_id=None):
        raise ConnectionError("qdrant unreachable")
    monkeypatch.setattr(nodes, "do_retrieve", down)
    assert client.post("/chat", json={"question": "x"}).status_code == 503
    assert client.post("/chat/stream", json={"question": "x"}).status_code == 503   # before streaming


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


# ---------------------------------------------------------------- graph behaviour through the API

def test_chat_without_history_skips_rewrite(client, llm, retrieved):
    body = client.post("/chat", json={"question": "What is RAG-Token?", "top_k": 3}).json()
    assert "rewrite" not in llm.calls
    assert body["rewritten_query"] is None and body["retries"] == 0
    assert retrieved[0] == {"query": "What is RAG-Token?", "top_k": 3, "arxiv_id": None}


def test_chat_with_history_rewrites_the_follow_up(client, llm, retrieved, monkeypatch):
    monkeypatch.setattr(nodes.settings, "rewrite_paper_filter", True)
    body = client.post("/chat", json={"question": "How does it work?", "history": HISTORY}).json()
    assert llm.calls[0] == "rewrite"
    assert body["rewritten_query"] == "How does RAG-Token work?"
    assert retrieved[0]["query"] == "How does RAG-Token work?"
    assert retrieved[0]["arxiv_id"] == "2005.11401"            # paper filter from the rewrite


def test_chat_abstains_after_one_retry_when_nothing_is_relevant(client, llm, retrieved):
    llm.relevant = []
    body = client.post("/chat", json={"question": "How many minutes are in an hour?"}).json()
    assert body["answer"] == NO_ANSWER
    assert body["sources"] == [] and body["retries"] == 1
    assert llm.calls == ["grade", "reformulate", "grade"] and len(retrieved) == 2


def test_user_chosen_paper_is_never_dropped(client, llm, retrieved):
    llm.relevant = []                                           # forces a retry
    client.post("/chat", json={"question": "x", "arxiv_id": "2310.11511"})
    assert [c["arxiv_id"] for c in retrieved] == ["2310.11511", "2310.11511"]


def test_stream_reports_graph_info_in_done(client):
    events = parse_sse(client.post("/chat/stream",
                                   json={"question": "How does it work?", "history": HISTORY}).text)
    assert [n for n, _ in events][0] == "sources" and events[-1][0] == "done"
    assert events[-1][1]["rewritten_query"] == "How does RAG-Token work?"
    assert events[-1][1]["retries"] == 0


def test_stream_abstains_without_calling_the_generator(client, llm, monkeypatch):
    llm.relevant = []
    def must_not_run(q, chunks):
        raise AssertionError("generator called while abstaining")
    monkeypatch.setattr(api, "generate_stream", must_not_run)
    events = parse_sse(client.post("/chat/stream", json={"question": "Off-topic?"}).text)
    assert [n for n, _ in events] == ["sources", "token", "done"]
    assert events[0][1] == [] and events[1][1] == NO_ANSWER and events[2][1]["retries"] == 1


def test_stream_generator_gets_the_interpreted_question(client, monkeypatch):
    seen = {}
    def capture(q, chunks):
        seen["q"] = q
        return iter(["ok"])
    monkeypatch.setattr(api, "generate_stream", capture)
    client.post("/chat/stream", json={"question": "How does it work?", "history": HISTORY})
    assert seen["q"] == "How does it work?\n(Interpreted as: How does RAG-Token work?)"
