"""Lazy, shared model loading.

Models are loaded once per process (per name/device) and reused by the
retriever, verifier, CLI and UI, so no duplicate instances occupy memory.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .config import ModelConfig

logger = logging.getLogger(__name__)

_CACHE: dict[tuple[str, str, str], object] = {}
_LOCK = threading.Lock()


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded (missing download, no network, OOM...)."""


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _load(kind: str, name: str, device: str, factory):
    """Return a cached model instance, loading it on first use.

    ``factory(device)`` builds the model. A CUDA out-of-memory error falls
    back to CPU so the pipeline stays usable on small GPUs.
    """
    key = (kind, name, device)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key], device
        logger.info("Loading %s model %s on %s", kind, name, device)
        try:
            _CACHE[key] = factory(device)
            return _CACHE[key], device
        except Exception as exc:
            if device == "cpu" or "out of memory" not in str(exc).lower():
                raise ModelLoadError(
                    f"Could not load model '{name}': {exc}. On first run an internet "
                    "connection is needed once to download it into the Hugging Face cache."
                ) from exc
            logger.warning("GPU out of memory while loading %s; using CPU", name)
    return _load(kind, name, "cpu", factory)


class Embedder:
    """Wrapper around a SentenceTransformer producing L2-normalized numpy vectors."""

    def __init__(self, config: ModelConfig | None = None):
        config = config or ModelConfig()
        self.config = config
        self.device = resolve_device(config.device)

        def factory(device):
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(config.embedding_model, device=device)

        self.model, self.device = _load("embed", config.embedding_model, self.device, factory)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


class NLIModel:
    """Cross-encoder NLI returning full probability distributions.

    premise = evidence text, hypothesis = claim. The label order is read from
    the model config rather than assumed.
    """

    def __init__(self, config: ModelConfig | None = None):
        config = config or ModelConfig()
        self.config = config
        self.device = resolve_device(config.device)

        def factory(device):
            from sentence_transformers import CrossEncoder
            return CrossEncoder(config.nli_model, device=device)

        self.model, self.device = _load("nli", config.nli_model, self.device, factory)
        id2label = self.model.model.config.id2label
        self.label_index = {label.lower(): int(i) for i, label in id2label.items()}
        missing = {"entailment", "contradiction", "neutral"} - set(self.label_index)
        if missing:
            raise ModelLoadError(f"NLI model labels {id2label} lack {missing}.")

    def predict(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        """Return [{'entailment': p, 'contradiction': p, 'neutral': p}, ...]."""
        if not pairs:
            return []
        logits = self.model.predict(
            pairs, batch_size=self.config.batch_size, show_progress_bar=False,
            convert_to_numpy=True,
        )
        logits = np.asarray(logits, dtype=np.float64).reshape(len(pairs), -1)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return [
            {label: float(row[index]) for label, index in self.label_index.items()}
            for row in probs
        ]


def get_embedder(config: ModelConfig | None = None) -> Embedder:
    return Embedder(config)


def get_nli(config: ModelConfig | None = None) -> NLIModel:
    return NLIModel(config)
