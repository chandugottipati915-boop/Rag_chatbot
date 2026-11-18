"""Email + password auth.

- Passwords are stored as argon2 hashes (pwdlib), never as text.
- Signing in returns a JWT in an httpOnly cookie: JavaScript can't read it, the browser
  sends it with every request, and its signature means nobody can change the user id in it.
- `current_user` is the FastAPI dependency that protects a route: no valid token → 401.
"""
import logging
import secrets
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, Response
from pwdlib import PasswordHash
from sqlalchemy.orm import Session

from rag.config import settings
from rag.db import repo
from rag.db.models import User
from rag.db.session import get_db

log = logging.getLogger("rag.auth")

COOKIE_NAME = "session"
ALGORITHM = "HS256"

_hasher = PasswordHash.recommended()            # argon2id with safe default parameters
# Checked when an email doesn't exist, so that request takes as long as a wrong password:
# response times can't reveal which emails have accounts.
_DUMMY_HASH = _hasher.hash("not-a-real-password")

if settings.jwt_secret:
    _SECRET = settings.jwt_secret
else:
    _SECRET = secrets.token_urlsafe(32)
    log.warning("JWT_SECRET is not set: using a random secret, so everyone is signed out "
                "when the server restarts. Set JWT_SECRET in .env.")


# ------------------------------------------------------------------ passwords

def hash_password(password: str) -> str:
    return _hasher.hash(password)


def authenticate(db: Session, email: str, password: str) -> User | None:
    """The user if the email and password match, else None (same work either way)."""
    user = repo.get_user_by_email(db, email)
    if user is None:
        _hasher.verify(password, _DUMMY_HASH)
        return None
    return user if _hasher.verify(password, user.password_hash) else None


# ------------------------------------------------------------------ session tokens

def create_token(user_id: uuid.UUID) -> str:
    now = datetime.now(timezone.utc)
    claims = {"sub": str(user_id), "iat": now, "exp": now + timedelta(days=settings.jwt_expire_days)}
    return jwt.encode(claims, _SECRET, algorithm=ALGORITHM)


def set_session_cookie(response: Response, user_id: uuid.UUID) -> None:
    response.set_cookie(
        COOKIE_NAME,
        create_token(user_id),
        max_age=settings.jwt_expire_days * 24 * 3600,
        httponly=True,                   # not readable by JavaScript (protects against XSS theft)
        secure=settings.cookie_secure,   # HTTPS only, in production
        samesite="lax",                  # not sent on cross-site POSTs (protects against CSRF)
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True,
                           secure=settings.cookie_secure, samesite="lax")


def _token_from(request: Request) -> str | None:
    # the browser sends the cookie; scripts and curl can send "Authorization: Bearer <token>"
    if token := request.cookies.get(COOKIE_NAME):
        return token
    scheme, _, credentials = request.headers.get("Authorization", "").partition(" ")
    return credentials if scheme.lower() == "bearer" and credentials else None


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """FastAPI dependency: the signed-in user, or 401. Add `user: User = Depends(current_user)`
    to a route to protect it."""
    token = _token_from(request)
    if not token:
        raise _unauthorized("Not signed in")
    try:
        claims = jwt.decode(token, _SECRET, algorithms=[ALGORITHM])   # checks signature and expiry
        user_id = uuid.UUID(claims["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise _unauthorized("Your session has expired. Sign in again.")
    user = db.get(User, user_id)
    if user is None:                                  # account deleted since the token was issued
        raise _unauthorized("Your session has expired. Sign in again.")
    return user


# ------------------------------------------------------------------ brute-force protection

class LoginLimiter:
    """Blocks an email + IP pair after `max_failures` wrong passwords within `window_s`.

    Kept in memory, so it is per process and resets on restart: enough for one server.
    With several API processes, keep the counts in Redis instead.
    """

    def __init__(self, max_failures: int = 5, window_s: int = 15 * 60):
        self.max_failures, self.window_s = max_failures, window_s
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        times = [t for t in self._failures.get(key, []) if now - t < self.window_s]
        self._failures[key] = times
        return times

    def check(self, key: str) -> None:
        with self._lock:
            times = self._recent(key, time.monotonic())
            if len(times) >= self.max_failures:
                retry_after = int(self.window_s - (time.monotonic() - times[0])) + 1
                raise HTTPException(status_code=429, headers={"Retry-After": str(retry_after)},
                                    detail="Too many failed sign-ins. Try again in a few minutes.")

    def failed(self, key: str) -> None:
        with self._lock:
            self._recent(key, time.monotonic()).append(time.monotonic())

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


login_limiter = LoginLimiter()
