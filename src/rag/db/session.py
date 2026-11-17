"""Database connection: one engine for the app, one session per request."""
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from rag.config import settings

# pool_pre_ping: test a pooled connection before using it, so a restarted Postgres doesn't break requests
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: a session that is closed when the request ends."""
    with SessionLocal() as db:
        yield db
