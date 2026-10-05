"""Per-user document workspaces on top of the shared models.

Each user has an own ``ClaimAwareRAG`` index built from their stored files; the
models themselves are loaded once per process and shared (carag.models). Model
calls are serialized with one lock, since the GPU is shared and the pipelines
are not designed for concurrent use.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

from ..answering import AnswerResult
from ..config import RAGConfig
from ..ingestion import SUPPORTED_EXTENSIONS, IngestionError, load_bytes
from ..pipeline import ClaimAwareRAG
from .store import Document, Store

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_QUESTION_CHARS = 1000
MAX_VERIFY_CHARS = 5000

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
                       "page": c.unit.page, "text": c.unit.text} for c in result.citations],
        "claims": [{"claim": c.claim, "status": c.status.value, "explanation": c.explanation,
                    "self_supported": c.self_supported} for c in result.claims],
        "removed_claims": [{"claim": c.claim, "status": c.status.value} for c in result.removed_claims],
        "notes": result.notes,
        "timings": result.timings,
        "budget": None if budget is None else {"used": budget.used_total, "budget": budget.budget,
                                               "retrieval_rounds": budget.retrieval_rounds},
    }


class Workspaces:
    def __init__(self, store: Store, data_dir: Path, config: RAGConfig | None = None):
        self.store = store
        self.files = Path(data_dir) / "files"
        self.config = config or RAGConfig()
        self._pipelines: dict[int, ClaimAwareRAG] = {}
        self._lock = threading.RLock()
        self.models_ready = threading.Event()
        self.model_error: str | None = None

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

    # -- pipelines ---------------------------------------------------------------------
    def _path(self, doc: Document) -> Path:
        return self.files / str(doc.user_id) / f"{doc.id}{Path(doc.name).suffix.lower()}"

    def _pipeline(self, user_id: int) -> ClaimAwareRAG:
        """The user's pipeline, rebuilt from their stored files on first use."""
        rag = self._pipelines.get(user_id)
        if rag is None:
            rag = ClaimAwareRAG(self.config)
            for doc in self.store.documents(user_id):
                try:
                    rag.add_bytes(self._path(doc).read_bytes(), doc.name)
                except (OSError, IngestionError) as exc:
                    logger.warning("Could not reload %s for user %s: %s", doc.name, user_id, exc)
            self._pipelines[user_id] = rag
        return rag

    # -- documents ---------------------------------------------------------------------
    def documents(self, user_id: int) -> list[dict]:
        return [d.to_dict() for d in self.store.documents(user_id)]

    def add_document(self, user_id: int, filename: str, data: bytes) -> dict:
        name = clean_name(filename)
        extension = Path(name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise WorkspaceError(f"'{name}' is not a supported file type. "
                                 f"Use {', '.join(e.lstrip('.').upper() for e in SUPPORTED_EXTENSIONS)}.")
        if not data:
            raise WorkspaceError(f"'{name}' is empty.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise WorkspaceError(f"'{name}' is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        if self.store.has_document_named(user_id, name):
            raise WorkspaceError(f"A document named '{name}' already exists. Delete it first to replace it.")
        with self._lock:
            rag = self._pipeline(user_id)
            try:
                source = load_bytes(data, name, self.config.ingestion)
            except IngestionError as exc:
                raise WorkspaceError(str(exc)) from exc
            if not source.units:
                raise WorkspaceError(f"No readable text was found in '{name}'.")
            doc = Document(uuid.uuid4().hex, user_id, name, len(data), len(source.units),
                           list(source.warnings), time.time())
            path = self._path(doc)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            try:
                self.store.add_document(doc)
            except ValueError as exc:
                path.unlink(missing_ok=True)
                raise WorkspaceError(str(exc)) from exc
            rag.add_document(source)
        return doc.to_dict()

    def delete_document(self, user_id: int, doc_id: str) -> None:
        doc = self.store.document(user_id, doc_id)
        if doc is None:
            raise KeyError(doc_id)
        with self._lock:
            self.store.delete_document(user_id, doc_id)
            self._path(doc).unlink(missing_ok=True)
            rag = self._pipelines.get(user_id)
            if rag is not None and doc.name in rag.index.sources:
                rag.index.remove_source(doc.name)

    # -- questions ---------------------------------------------------------------------
    def ask(self, user_id: int, question: str) -> dict:
        question = (question or "").strip()
        if not question:
            raise WorkspaceError("Please enter a question.")
        if len(question) > MAX_QUESTION_CHARS:
            raise WorkspaceError(f"Questions can be at most {MAX_QUESTION_CHARS} characters.")
        if not self.store.documents(user_id):
            raise WorkspaceError("Upload at least one document before asking a question.")
        with self._lock:
            view = answer_view(self._pipeline(user_id).ask(question))
        view["id"] = self.store.add_history(user_id, question, view["outcome"], view)
        return view

    def verify(self, user_id: int, text: str) -> dict:
        """Check every claim in a piece of text (e.g. a chatbot answer) against the documents."""
        text = (text or "").strip()
        if not text:
            raise WorkspaceError("Paste some text to check.")
        if len(text) > MAX_VERIFY_CHARS:
            raise WorkspaceError(f"Text to check can be at most {MAX_VERIFY_CHARS} characters.")
        if not self.store.documents(user_id):
            raise WorkspaceError("Upload at least one document before checking text.")
        with self._lock:
            check = self._pipeline(user_id).check_answer(text)
        return {
            "revised": check.revised.text,
            "abstained": check.revised.abstained,
            "claims": [{"claim": c.claim, "status": c.status.value, "explanation": c.explanation,
                        "evidence": [j.premise for j in (c.supporting or c.contradicting)[:1]]}
                       for c in check.claims],
        }

    def forget(self, user_id: int) -> None:
        """Drop the in-memory index (e.g. on logout); it is rebuilt from disk when needed."""
        with self._lock:
            self._pipelines.pop(user_id, None)
