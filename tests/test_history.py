"""Chat-history tests: conversations CRUD and /chat/stream saving to the database.

Uses an in-memory SQLite database instead of Postgres, so nothing needs to be running.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import rag.api.main as api
from rag.db.models import Base, User
from rag.db.session import get_db
from tests.test_api import HISTORY, TEST_USER, client, llm, parse_sse, retrieved  # noqa: F401  (fixtures)


@pytest.fixture
def db_client(client, monkeypatch):   # noqa: F811
    # StaticPool: every session shares one connection, so the in-memory database survives
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(engine, expire_on_commit=False)

    def override_get_db():
        with TestSession() as s:
            yield s

    with TestSession() as s:                                 # the signed-in test user owns its chats
        s.add(User(id=TEST_USER.id, email=TEST_USER.email, password_hash="x"))
        s.commit()
    api.app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(api, "SessionLocal", TestSession)   # used inside the stream
    yield client
    api.app.dependency_overrides.clear()


def new_chat(c, title="RAG questions") -> str:
    r = c.post("/conversations", json={"title": title})
    assert r.status_code == 201
    return r.json()["id"]


def test_create_list_rename_delete(db_client):
    a = new_chat(db_client, "first")
    b = new_chat(db_client, "second")
    assert [c["id"] for c in db_client.get("/conversations").json()] == [b, a]   # newest first

    assert db_client.patch(f"/conversations/{a}", json={"title": "renamed"}).json()["title"] == "renamed"
    assert db_client.delete(f"/conversations/{a}").status_code == 204
    assert db_client.get(f"/conversations/{a}").status_code == 404
    assert [c["id"] for c in db_client.get("/conversations").json()] == [b]


def test_stream_saves_question_and_answer(db_client):
    cid = new_chat(db_client)
    events = parse_sse(db_client.post("/chat/stream",
                                      json={"question": "What is RAG-Token?", "conversation_id": cid}).text)
    assert events[-1][0] == "done"

    msgs = db_client.get(f"/conversations/{cid}").json()["messages"]
    assert [(m["role"], m["content"]) for m in msgs] == [
        ("user", "What is RAG-Token?"),
        ("assistant", "It uses per-token documents [1]."),
    ]
    assert msgs[1]["sources"][0]["arxiv_id"] == "2005.11401"
    assert msgs[1]["meta"]["retries"] == 0


def test_follow_up_uses_history_from_the_database(db_client):
    cid = new_chat(db_client)
    db_client.post("/chat/stream", json={"question": "What is RAG?", "conversation_id": cid})
    # no `history` in the request: the server loads it, so the follow-up gets rewritten
    events = parse_sse(db_client.post("/chat/stream",
                                      json={"question": "How does it work?", "conversation_id": cid}).text)
    assert events[-1][1]["rewritten_query"] == "How does RAG-Token work?"
    assert len(db_client.get(f"/conversations/{cid}").json()["messages"]) == 4


def test_unknown_conversation_is_404(db_client):
    r = db_client.post("/chat/stream", json={"question": "x",
                                             "conversation_id": "00000000-0000-0000-0000-000000000000"})
    assert r.status_code == 404


def test_chat_without_conversation_id_still_works_and_saves_nothing(db_client):
    assert db_client.post("/chat/stream", json={"question": "x"}).status_code == 200
    assert db_client.get("/conversations").json() == []


def test_health_reports_the_database(db_client, monkeypatch):
    monkeypatch.setattr(api.vs.client, "collection_exists", lambda *a: False)
    assert db_client.get("/health").json()["database"] is True     # the SQLite test database answers


def test_stopped_stream_keeps_the_partial_answer(db_client, monkeypatch):
    cid = new_chat(db_client)

    def stream_then_stop(q, chunks):
        yield "Partial "
        raise GeneratorExit        # what happens when the browser aborts mid-answer

    monkeypatch.setattr(api, "generate_stream", stream_then_stop)
    try:
        db_client.post("/chat/stream", json={"question": "What is RAG-Token?", "conversation_id": cid})
    except BaseException:
        pass
    msgs = db_client.get(f"/conversations/{cid}").json()["messages"]
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "Partial "
    assert msgs[-1]["meta"] == {"stopped": True}
