"""SQLite storage for accounts, sessions, documents and question history.

One short-lived connection per operation keeps the store safe to use from the
server's worker threads without sharing a connection.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    size INTEGER NOT NULL,
    sentences INTEGER NOT NULL,
    warnings TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    UNIQUE (user_id, name)
);
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    outcome TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS history_user ON history(user_id, id);
"""


@dataclass
class User:
    id: int
    username: str


@dataclass
class Document:
    id: str
    user_id: int
    name: str
    size: int
    sentences: int
    warnings: list[str]
    created_at: float

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "size": self.size, "sentences": self.sentences,
                "warnings": self.warnings, "created_at": self.created_at}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # -- users --------------------------------------------------------------------------
    def create_user(self, username: str, password_hash: str) -> User:
        """Raises ValueError when the username is taken (case-insensitive)."""
        try:
            with self._connect() as db:
                cur = db.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                                 (username, password_hash, time.time()))
                return User(cur.lastrowid, username)
        except sqlite3.IntegrityError as exc:
            raise ValueError("That username is already taken.") from exc

    def user_credentials(self, username: str) -> tuple[User, str] | None:
        with self._connect() as db:
            row = db.execute("SELECT id, username, password_hash FROM users WHERE username = ?",
                             (username,)).fetchone()
        return (User(row["id"], row["username"]), row["password_hash"]) if row else None

    def user_count(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    # -- sessions -----------------------------------------------------------------------
    def create_session(self, token_hash: str, user_id: int, ttl_s: float) -> None:
        now = time.time()
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            db.execute("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                       (token_hash, user_id, now, now + ttl_s))

    def session_user(self, token_hash: str) -> User | None:
        with self._connect() as db:
            row = db.execute("SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                             "WHERE s.token_hash = ? AND s.expires_at > ?", (token_hash, time.time())).fetchone()
        return User(row["id"], row["username"]) if row else None

    def delete_session(self, token_hash: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    # -- documents ----------------------------------------------------------------------
    def add_document(self, doc: Document) -> None:
        try:
            with self._connect() as db:
                db.execute("INSERT INTO documents (id, user_id, name, size, sentences, warnings, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (doc.id, doc.user_id, doc.name, doc.size, doc.sentences,
                            json.dumps(doc.warnings), doc.created_at))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A document named '{doc.name}' already exists. Delete it first to replace it.") from exc

    def documents(self, user_id: int) -> list[Document]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM documents WHERE user_id = ? ORDER BY created_at, name",
                              (user_id,)).fetchall()
        return [Document(r["id"], r["user_id"], r["name"], r["size"], r["sentences"], json.loads(r["warnings"]),
                         r["created_at"]) for r in rows]

    def document(self, user_id: int, doc_id: str) -> Document | None:
        return next((d for d in self.documents(user_id) if d.id == doc_id), None)

    def has_document_named(self, user_id: int, name: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM documents WHERE user_id = ? AND name = ?",
                              (user_id, name)).fetchone() is not None

    def delete_document(self, user_id: int, doc_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM documents WHERE user_id = ? AND id = ?", (user_id, doc_id))

    # -- history ------------------------------------------------------------------------
    def add_history(self, user_id: int, question: str, outcome: str, result: dict) -> int:
        with self._connect() as db:
            cur = db.execute("INSERT INTO history (user_id, question, outcome, result, created_at) "
                             "VALUES (?, ?, ?, ?, ?)",
                             (user_id, question, outcome, json.dumps(result), time.time()))
            return cur.lastrowid

    def history(self, user_id: int, limit: int = 50) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT id, question, outcome, result, created_at FROM history "
                              "WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
        return [{"id": r["id"], "question": r["question"], "outcome": r["outcome"],
                 "result": json.loads(r["result"]), "created_at": r["created_at"]} for r in rows]

    def clear_history(self, user_id: int) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
