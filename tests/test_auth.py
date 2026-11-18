"""Auth tests: sign-up, sign-in, sessions, brute-force limit, and that users only ever see
their own chats. Runs the real `current_user` dependency on an in-memory SQLite database.
"""
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import rag.api.main as api
import rag.auth as auth
import rag.graph.nodes as nodes
from rag.db.models import Base, User
from rag.db.session import get_db
from tests.test_api import llm, parse_sse, retrieved  # noqa: F401  (fixtures)

PASSWORD = "correct horse battery"


@pytest.fixture
def db(monkeypatch, llm, retrieved):   # noqa: F811
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(engine, expire_on_commit=False)

    def override_get_db():
        with TestSession() as s:
            yield s

    api.app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(api, "SessionLocal", TestSession)
    monkeypatch.setattr(nodes, "llm_generate", lambda q, chunks: "Answer [1].")
    monkeypatch.setattr(api, "generate_stream", lambda q, chunks: iter(["Answer ", "[1]."]))
    monkeypatch.setattr(auth, "login_limiter", auth.LoginLimiter())
    monkeypatch.setattr(api, "login_limiter", auth.login_limiter)
    yield TestSession
    api.app.dependency_overrides.clear()


def browser(db) -> TestClient:
    """A separate client = a separate browser with its own cookies."""
    return TestClient(api.app)


def signed_up(db, email="ada@example.com") -> TestClient:
    c = browser(db)
    assert c.post("/auth/signup", json={"email": email, "password": PASSWORD}).status_code == 201
    return c


# ------------------------------------------------------------------ sign-up / sign-in

def test_signup_signs_you_in_with_an_httponly_cookie(db):
    c = browser(db)
    r = c.post("/auth/signup", json={"email": "  Ada@Example.com ", "password": PASSWORD})
    assert r.status_code == 201
    assert r.json()["email"] == "ada@example.com"                      # normalized
    cookie = r.headers["set-cookie"].lower()
    assert "session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert c.get("/auth/me").json()["email"] == "ada@example.com"


def test_password_is_stored_hashed(db):
    signed_up(db)
    with db() as s:
        stored = s.scalar(select(User.password_hash))
    assert PASSWORD not in stored and stored.startswith("$argon2")


def test_duplicate_email_is_rejected(db):
    signed_up(db)
    r = browser(db).post("/auth/signup", json={"email": "ADA@example.com", "password": PASSWORD})
    assert r.status_code == 409


@pytest.mark.parametrize("body", [
    {"email": "not-an-email", "password": PASSWORD},
    {"email": "ada@example.com", "password": "short"},
])
def test_invalid_signup_input_is_422(db, body):
    assert browser(db).post("/auth/signup", json=body).status_code == 422


def test_login_and_logout(db):
    signed_up(db)
    c = browser(db)
    assert c.get("/auth/me").status_code == 401
    assert c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD}).status_code == 200
    assert c.get("/auth/me").status_code == 200
    assert c.post("/auth/logout").status_code == 204
    assert c.get("/auth/me").status_code == 401


def test_wrong_password_and_unknown_email_get_the_same_answer(db):
    signed_up(db)
    c = browser(db)
    wrong = c.post("/auth/login", json={"email": "ada@example.com", "password": "wrong password"})
    unknown = c.post("/auth/login", json={"email": "nobody@example.com", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()                    # doesn't reveal which emails exist


def test_too_many_failed_logins_are_blocked(db):
    signed_up(db)
    c = browser(db)
    for _ in range(5):
        assert c.post("/auth/login", json={"email": "ada@example.com", "password": "nope-nope"}).status_code == 401
    r = c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD})
    assert r.status_code == 429 and "Retry-After" in r.headers   # blocked even with the right password


def test_a_successful_login_resets_the_failure_count(db):
    signed_up(db)
    c = browser(db)
    for _ in range(4):
        c.post("/auth/login", json={"email": "ada@example.com", "password": "nope-nope"})
    assert c.post("/auth/login", json={"email": "ada@example.com", "password": PASSWORD}).status_code == 200
    for _ in range(4):
        assert c.post("/auth/login", json={"email": "ada@example.com", "password": "nope-nope"}).status_code == 401


# ------------------------------------------------------------------ tokens

def test_tampered_or_expired_tokens_are_rejected(db):
    c = signed_up(db)
    user_id = c.get("/auth/me").json()["id"]
    forged = jwt.encode({"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=1)},
                        "someone-elses-secret", algorithm="HS256")
    expired = jwt.encode({"sub": user_id, "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
                         auth._SECRET, algorithm="HS256")
    for token in (forged, expired, "garbage"):
        assert browser(db).get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_bearer_token_works_for_scripts(db):
    c = signed_up(db)
    user_id = uuid.UUID(c.get("/auth/me").json()["id"])
    token = auth.create_token(user_id)
    assert browser(db).get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200


# ------------------------------------------------------------------ protected routes

@pytest.mark.parametrize("method,path,body", [
    ("get", "/conversations", None),
    ("post", "/conversations", {"title": "x"}),
    ("post", "/chat", {"question": "What is RAG?"}),
    ("post", "/chat/stream", {"question": "What is RAG?"}),
    ("post", "/search", {"question": "What is RAG?"}),
    ("get", "/ingest", None),
])
def test_routes_need_a_signed_in_user(db, method, path, body):
    c = browser(db)
    r = c.request(method.upper(), path, json=body)
    assert r.status_code == 401


def test_health_and_papers_stay_public(db, monkeypatch):
    monkeypatch.setattr(api.vs.client, "collection_exists", lambda *a: False)
    assert browser(db).get("/health").status_code == 200
    assert browser(db).get("/papers").status_code == 200


def test_signed_in_user_can_chat(db):
    c = signed_up(db)
    cid = c.post("/conversations", json={"title": "RAG"}).json()["id"]
    events = parse_sse(c.post("/chat/stream", json={"question": "What is RAG-Token?", "conversation_id": cid}).text)
    assert events[-1][0] == "done"
    assert len(c.get(f"/conversations/{cid}").json()["messages"]) == 2


# ------------------------------------------------------------------ isolation

def test_users_only_see_their_own_chats(db):
    ada, bob = signed_up(db, "ada@example.com"), signed_up(db, "bob@example.com")
    cid = ada.post("/conversations", json={"title": "Ada's chat"}).json()["id"]

    assert bob.get("/conversations").json() == []
    # someone else's chat looks exactly like a missing one: 404, never 403
    assert bob.get(f"/conversations/{cid}").status_code == 404
    assert bob.patch(f"/conversations/{cid}", json={"title": "mine now"}).status_code == 404
    assert bob.delete(f"/conversations/{cid}").status_code == 404
    assert bob.post("/chat/stream", json={"question": "x", "conversation_id": cid}).status_code == 404
    assert bob.post("/chat", json={"question": "x", "conversation_id": cid}).status_code == 404

    assert ada.get(f"/conversations/{cid}").json()["title"] == "Ada's chat"   # untouched
    assert [c["id"] for c in ada.get("/conversations").json()] == [cid]
