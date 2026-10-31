"""Local accounts: password hashing, session tokens and login throttling.

Passwords are hashed with scrypt (standard library) and a random salt. Session
tokens are random; only their SHA-256 is stored, so a copied database does not
contain usable sessions.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import threading
import time

from .store import Store, User

SESSION_COOKIE = "carag_session"
SESSION_TTL_S = 7 * 24 * 3600

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_MIN_PASSWORD = 8
_MAX_PASSWORD = 256
_SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}

# Failed logins per username before a temporary lock, and the lock duration.
MAX_FAILURES = 5
LOCK_S = 60


class AuthError(ValueError):
    """A login or registration problem to show to the user."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt${n}${r}${p}${salt}${digest}".format(
        **_SCRYPT, salt=base64.b64encode(salt).decode(), digest=base64.b64encode(digest).decode())


def check_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        actual = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), dklen=32,
                                n=int(n), r=int(r), p=int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, base64.b64decode(digest))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def validate_new_account(username: str, password: str) -> None:
    if not _USERNAME_RE.match(username or ""):
        raise AuthError("Usernames are 3-32 characters: letters, digits, '.', '_' or '-'.")
    if not password or len(password) < _MIN_PASSWORD:
        raise AuthError(f"Passwords need at least {_MIN_PASSWORD} characters.")
    if len(password) > _MAX_PASSWORD:
        raise AuthError(f"Passwords can be at most {_MAX_PASSWORD} characters.")


class Accounts:
    def __init__(self, store: Store):
        self.store = store
        self._failures: dict[str, tuple[int, float]] = {}   # username -> (count, locked_until)
        self._lock = threading.Lock()
        # Compared against when the username does not exist, so a login takes about
        # as long whether or not the account exists.
        self._dummy_hash = hash_password(secrets.token_urlsafe(16))

    def register(self, username: str, password: str) -> User:
        username = (username or "").strip()
        validate_new_account(username, password)
        try:
            return self.store.create_user(username, hash_password(password))
        except ValueError as exc:
            raise AuthError(str(exc)) from exc

    def login(self, username: str, password: str) -> User:
        key = (username or "").strip().lower()
        with self._lock:
            count, locked_until = self._failures.get(key, (0, 0.0))
            if locked_until > time.time():
                wait = int(locked_until - time.time()) + 1
                raise AuthError(f"Too many failed attempts. Try again in {wait} seconds.")
        found = self.store.user_credentials((username or "").strip()) if key else None
        valid = check_password(password or "", found[1] if found else self._dummy_hash) and found is not None
        with self._lock:
            if not valid:
                count = self._failures.get(key, (0, 0.0))[0] + 1
                # The counter restarts once the lock is set, so each lock needs MAX_FAILURES new failures.
                self._failures[key] = (0, time.time() + LOCK_S) if count >= MAX_FAILURES else (count, 0.0)
                raise AuthError("Incorrect username or password.")
            self._failures.pop(key, None)
        return found[0]

    def start_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        self.store.create_session(token_hash(token), user.id, SESSION_TTL_S)
        return token

    def user_for(self, token: str | None) -> User | None:
        return self.store.session_user(token_hash(token)) if token else None

    def end_session(self, token: str | None) -> None:
        if token:
            self.store.delete_session(token_hash(token))
