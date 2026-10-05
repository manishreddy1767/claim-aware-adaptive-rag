"""Document workspaces on top of the shared models, in two storage modes.

* **local** (the installed application): each user's documents, their search
  index source and question history are kept on this computer and survive
  sign-out and restarts. Files on this computer can also be added by path; they
  are read where they are, never copied or deleted.
* **web** (a hosted website): nothing a visitor adds is written to disk.
  Documents, the index and history live in memory for one sign-in session and
  are dropped when the user signs out, after ``idle_timeout_s`` without
  activity, or when the server stops. Only the account itself is stored.

The models are loaded once per process and shared (carag.models). Model calls
are serialized with one lock, since the GPU is shared and the pipelines are not
designed for concurrent use.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from ..answering import AnswerResult
from ..config import RAGConfig
from ..ingestion import SUPPORTED_EXTENSIONS, IngestionError, load_bytes, load_url
from ..pipeline import ClaimAwareRAG
from ..schema import EvidenceUnit, SourceDocument
from .store import Document, Store

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_QUESTION_CHARS = 1000
MAX_VERIFY_CHARS = 5000
MAX_FOLDER_FILES = 200
MAX_SESSION_DOCUMENTS = 50         # web mode: per sign-in session, to bound memory
HISTORY_LIMIT = 50

# How the UI should present each kind of answer.
OUTCOME_LABELS = {
    "answer": "Answered from your documents",
    "yes": "Confirmed by your documents",
    "reject_premise": "The question's assumption is wrong",
    "not_specified": "Not specified in your documents",
    "conflict": "Your documents disagree",
    "abstain": "No answer in your documents",
}


class WorkspaceError(ValueError):
    """A problem with the user's request (bad file, empty question...)."""


def clean_name(filename: str) -> str:
    """Display name for an upload: the base name only, without control characters."""
    name = Path((filename or "").replace("\\", "/")).name
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name:
        raise WorkspaceError("The file has no name.")
    if len(name) > 150:
        stem, ext = Path(name).stem, Path(name).suffix
        name = stem[: 150 - len(ext)] + ext
    return name


def public_url_only(url: str) -> None:
    """Refuse URLs whose host resolves to a private, loopback or otherwise internal address,
    so website visitors cannot make the server fetch pages from its own network."""
    host = urlsplit(url).hostname or ""
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except socket.gaierror as exc:
        raise IngestionError(f"Could not find the website '{host}'.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if not ip.is_global:
            raise IngestionError(f"'{host}' is on a private network; only public websites can be added.")


def answer_view(result: AnswerResult) -> dict:
    """The parts of an answer the UI shows."""
    budget = result.budget
    return {
        "question": result.question,
        "outcome": result.outcome,
        "outcome_label": OUTCOME_LABELS[result.outcome],
        "answer": result.answer,
        "answer_span": result.answer_span,
        "citations": [{"marker": c.marker, "source": c.unit.source, "section": c.unit.section,
                       "page": c.unit.page, "url": c.unit.url, "text": c.unit.text} for c in result.citations],
        "claims": [{"claim": c.claim, "status": c.status.value, "explanation": c.explanation,
                    "self_supported": c.self_supported} for c in result.claims],
        "removed_claims": [{"claim": c.claim, "status": c.status.value} for c in result.removed_claims],
        "notes": result.notes,
        "timings": result.timings,
        "budget": None if budget is None else {"used": budget.used_total, "budget": budget.budget,
                                               "retrieval_rounds": budget.retrieval_rounds},
    }


@dataclass
class _Memory:
    """Web mode: everything one sign-in session added, held only in memory."""
    rag: ClaimAwareRAG
    documents: list[Document] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    last_used: float = field(default_factory=time.time)


class Workspaces:
    def __init__(self, store: Store, data_dir: Path, config: RAGConfig | None = None, mode: str = "local",
                 idle_timeout_s: float = 2 * 3600):
        if mode not in ("local", "web"):
            raise ValueError(f"Unknown mode '{mode}'.")
        self.store = store
        self.mode = mode
        self.files = Path(data_dir) / "files"
        self.config = config or RAGConfig()
        if mode == "web":
            # A headless browser would fetch every resource a page links to, including
            # internal addresses; hosted mode reads the page's own HTML only.
            self.config.ingestion.render_js = False
        self.idle_timeout_s = idle_timeout_s
        self._pipelines: dict[int, ClaimAwareRAG] = {}     # local mode, by user id
        self._memory: dict[str, _Memory] = {}               # web mode, by session key
        self._lock = threading.RLock()
        self.models_ready = threading.Event()
        self.model_error: str | None = None

    @property
    def persistent(self) -> bool:
        return self.mode == "local"

    # -- models ----------------------------------------------------------------------
    def warm_up(self) -> None:
        """Load every model once (in a background thread at start-up)."""
        try:
            with self._lock:
                rag = ClaimAwareRAG(self.config)
                rag.answerer      # QA model
                rag.verifier      # NLI model
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.exception("Model loading failed")
            self.model_error = str(exc)
        finally:
            self.models_ready.set()

    # -- owners ------------------------------------------------------------------------
    # An owner is the user id in local mode and the sign-in session key in web mode.
    def _memory_for(self, key: str) -> _Memory:
        self._expire_idle()
        memory = self._memory.get(key)
        if memory is None:
            memory = self._memory[key] = _Memory(ClaimAwareRAG(self.config))
        memory.last_used = time.time()
        return memory

    def _expire_idle(self) -> None:
        cutoff = time.time() - self.idle_timeout_s
        with self._lock:
            for key in [k for k, m in self._memory.items() if m.last_used < cutoff]:
                del self._memory[key]

    def _documents(self, owner) -> list[Document]:
        if self.persistent:
            return self.store.documents(owner)
        self._expire_idle()
        memory = self._memory.get(owner)
        return list(memory.documents) if memory else []

    # -- local mode storage --------------------------------------------------------------
    def _path(self, doc: Document) -> Path:
        """Where a local-mode document's content lives on disk."""
        if doc.origin == "path":
            return Path(doc.location)
        suffix = ".json" if doc.origin == "url" else Path(doc.name).suffix.lower()
        return self.files / str(doc.user_id) / f"{doc.id}{suffix}"

    def _reload(self, doc: Document) -> SourceDocument:
        path = self._path(doc)
        if doc.origin == "url":
            data = json.loads(path.read_text(encoding="utf-8"))
            data["units"] = [EvidenceUnit(**u) for u in data["units"]]
            return SourceDocument(**data)
        return load_bytes(path.read_bytes(), doc.name, self.config.ingestion)

    def _pipeline(self, owner) -> ClaimAwareRAG:
        """The owner's pipeline; in local mode rebuilt from the stored documents on first use."""
        if not self.persistent:
            return self._memory_for(owner).rag
        rag = self._pipelines.get(owner)
        if rag is None:
            rag = ClaimAwareRAG(self.config)
            for doc in self.store.documents(owner):
                try:
                    rag.add_document(self._reload(doc))
                except (OSError, ValueError, TypeError, IngestionError) as exc:
                    logger.warning("Could not reload %s for user %s: %s", doc.name, owner, exc)
            self._pipelines[owner] = rag
        return rag

    # -- documents ---------------------------------------------------------------------
    def documents(self, owner) -> list[dict]:
        docs = []
        for doc in self._documents(owner):
            item = doc.to_dict()
            if self.persistent and not self._path(doc).exists():
                hint = ("The file was moved or deleted on this computer" if doc.origin == "path"
                        else "The stored file is missing")
                item["warnings"] = item["warnings"] + [f"{hint}, so this document is not used. "
                                                       "Delete it and add it again."]
            docs.append(item)
        return docs

    def _check_room(self, owner, name: str) -> None:
        docs = self._documents(owner)
        if any(d.name == name for d in docs):
            raise WorkspaceError(f"A document named '{name}' already exists. Delete it first to replace it.")
        if not self.persistent and len(docs) >= MAX_SESSION_DOCUMENTS:
            raise WorkspaceError(f"At most {MAX_SESSION_DOCUMENTS} documents can be added per session.")

    def _register(self, owner, source: SourceDocument, name: str, size: int, origin: str,
                  location: str | None, content: bytes | None) -> dict:
        """Index an ingested document and record it (on disk only in local mode)."""
        if not source.units:
            raise WorkspaceError(f"No readable text was found in '{name}'.")
        user_id = owner if self.persistent else 0
        doc = Document(uuid.uuid4().hex, user_id, name, size, len(source.units), list(source.warnings),
                       time.time(), origin, location)
        rag = self._pipeline(owner)
        if self.persistent:
            path = None
            if origin != "path":
                path = self._path(doc)
                path.parent.mkdir(parents=True, exist_ok=True)
                if origin == "url":
                    path.write_text(json.dumps(asdict(source)), encoding="utf-8")
                else:
                    path.write_bytes(content)
            try:
                self.store.add_document(doc)
            except ValueError as exc:
                if path is not None:
                    path.unlink(missing_ok=True)
                raise WorkspaceError(str(exc)) from exc
        else:
            self._memory_for(owner).documents.append(doc)
        rag.add_document(source)
        return doc.to_dict()

    def add_document(self, owner, filename: str, data: bytes) -> dict:
        """An uploaded file (both modes)."""
        name = clean_name(filename)
        self._check_file(name, len(data))
        with self._lock:
            self._check_room(owner, name)
            try:
                source = load_bytes(data, name, self.config.ingestion)
            except IngestionError as exc:
                raise WorkspaceError(str(exc)) from exc
            return self._register(owner, source, name, len(data), "upload", None, data)

    def add_url(self, owner, url: str) -> dict:
        """A webpage or online PDF. Web mode refuses private-network addresses."""
        url = (url or "").strip()
        if not url:
            raise WorkspaceError("Enter the address of a webpage.")
        if len(url) > 2000:
            raise WorkspaceError("That address is too long.")
        guard = None if self.persistent else public_url_only
        with self._lock:
            self._check_room(owner, url)
            try:
                source = load_url(url, self.config.ingestion, url_guard=guard)
            except IngestionError as exc:
                raise WorkspaceError(str(exc)) from exc
            name = source.source
            if name != url:
                self._check_room(owner, name)
            return self._register(owner, source, name, sum(len(u.text) for u in source.units), "url", url, None)

    def add_path(self, owner, path_text: str) -> dict:
        """Local mode only: a file, or every supported file in a folder (recursively), read
        in place on this computer. Returns {"added": [...], "errors": [...]}."""
        if not self.persistent:
            raise WorkspaceError("Files on the server's computer cannot be added on the website.")
        raw = (path_text or "").strip().strip('"')
        if not raw:
            raise WorkspaceError("Enter the path of a file or folder on this computer.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise WorkspaceError("Enter the full path, for example C:\\Users\\you\\Documents\\policy.pdf")
        if not path.exists():
            raise WorkspaceError(f"'{path}' does not exist.")
        if path.is_file():
            candidates = [path]
        else:
            candidates = sorted(p for p in path.rglob("*")
                                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                                and not any(part.startswith(".") for part in p.relative_to(path).parts))
            if not candidates:
                raise WorkspaceError(f"No PDF, TXT, Markdown or Word files were found in '{path}'.")
            if len(candidates) > MAX_FOLDER_FILES:
                raise WorkspaceError(f"'{path}' contains {len(candidates)} documents; add at most "
                                     f"{MAX_FOLDER_FILES} at a time (choose a smaller folder).")
        added, errors = [], []
        for file in candidates:
            try:
                name = clean_name(file.name)
                self._check_file(name, file.stat().st_size)
                with self._lock:
                    self._check_room(owner, name)
                    try:
                        source = load_bytes(file.read_bytes(), name, self.config.ingestion)
                    except IngestionError as exc:
                        raise WorkspaceError(str(exc)) from exc
                    added.append(self._register(owner, source, name, file.stat().st_size, "path",
                                                str(file.resolve()), None))
            except (WorkspaceError, OSError) as exc:
                errors.append({"name": file.name, "error": str(exc)})
        if path.is_file() and errors:
            raise WorkspaceError(errors[0]["error"])
        return {"added": added, "errors": errors}

    @staticmethod
    def _check_file(name: str, size: int) -> None:
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise WorkspaceError(f"'{name}' is not a supported file type. "
                                 f"Use {', '.join(e.lstrip('.').upper() for e in SUPPORTED_EXTENSIONS)}.")
        if not size:
            raise WorkspaceError(f"'{name}' is empty.")
        if size > MAX_UPLOAD_BYTES:
            raise WorkspaceError(f"'{name}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    def delete_document(self, owner, doc_id: str) -> None:
        """Remove a document from the workspace. A file added by path is only forgotten,
        never deleted from the computer."""
        doc = next((d for d in self._documents(owner) if d.id == doc_id), None)
        if doc is None:
            raise KeyError(doc_id)
        with self._lock:
            if self.persistent:
                self.store.delete_document(owner, doc_id)
                if doc.origin != "path":
                    self._path(doc).unlink(missing_ok=True)
                rag = self._pipelines.get(owner)
            else:
                memory = self._memory_for(owner)
                memory.documents = [d for d in memory.documents if d.id != doc_id]
                rag = memory.rag
            if rag is not None and doc.name in rag.index.sources:
                rag.index.remove_source(doc.name)

    # -- questions ---------------------------------------------------------------------
    def ask(self, owner, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            raise WorkspaceError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise WorkspaceError(f"Questions can be at most {MAX_QUESTION_CHARS} characters.")
        if not self._documents(owner):
            raise WorkspaceError("Add at least one document before asking a question.")
        with self._lock:
            view = answer_view(self._pipeline(owner).ask(question))
        if self.persistent:
            view["id"] = self.store.add_history(owner, question, view["outcome"], view)
        else:
            memory = self._memory_for(owner)
            view["id"] = len(memory.history) + 1
            memory.history.insert(0, {"id": view["id"], "question": question, "outcome": view["outcome"],
                                      "result": view, "created_at": time.time()})
            del memory.history[HISTORY_LIMIT:]
        return view

    def verify(self, owner, text: str) -> dict:
        """Check every claim in a piece of text (e.g. a chatbot answer) against the documents."""
        text = (text or "").strip()
        if not text:
            raise WorkspaceError("Paste some text to check.")
        if len(text) > MAX_VERIFY_CHARS:
            raise WorkspaceError(f"Text to check can be at most {MAX_VERIFY_CHARS} characters.")
        if not self._documents(owner):
            raise WorkspaceError("Add at least one document before checking text.")
        with self._lock:
            check = self._pipeline(owner).check_answer(text)
        return {
            "revised": check.revised.text,
            "abstained": check.revised.abstained,
            "claims": [{"claim": c.claim, "status": c.status.value, "explanation": c.explanation,
                        "evidence": [j.premise for j in (c.supporting or c.contradicting)[:1]]}
                       for c in check.claims],
        }

    def history(self, owner) -> list[dict]:
        if self.persistent:
            return self.store.history(owner, HISTORY_LIMIT)
        self._expire_idle()
        memory = self._memory.get(owner)
        return list(memory.history) if memory else []

    def clear_history(self, owner) -> None:
        if self.persistent:
            self.store.clear_history(owner)
        elif owner in self._memory:
            self._memory[owner].history.clear()

    def forget(self, owner) -> None:
        """On sign-out. Local mode drops only the in-memory index (rebuilt from disk later);
        web mode discards everything the session added."""
        with self._lock:
            if self.persistent:
                self._pipelines.pop(owner, None)
            else:
                self._memory.pop(owner, None)
