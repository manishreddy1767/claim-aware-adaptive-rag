"""In-memory evidence index: sentence embeddings + BM25 lexical statistics.

Embeddings are computed once when a source is added (not on every query),
which the original prototype did.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .models import Embedder
from .schema import EvidenceUnit, SourceDocument
from .text_utils import (CAUSAL_MARKERS, COMPARISON_MARKERS, LIMITATION_MARKERS,
                         contains_marker, extract_numbers, tokenize)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def add(self, token_lists: list[list[str]]) -> None:
        for tokens in token_lists:
            doc_id = len(self.doc_lengths)
            self.doc_lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_id, tf))

    def idf(self, term: str) -> float:
        n = len(self.doc_lengths)
        df = len(self.postings.get(term, ()))
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query_terms: list[str]) -> np.ndarray:
        n = len(self.doc_lengths)
        result = np.zeros(n, dtype=np.float32)
        if n == 0:
            return result
        avg_len = (sum(self.doc_lengths) / n) or 1.0
        lengths = np.asarray(self.doc_lengths, dtype=np.float32)
        for term in set(query_terms):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            ids = np.fromiter((d for d, _ in postings), dtype=np.int64)
            tfs = np.fromiter((tf for _, tf in postings), dtype=np.float32)
            norm = self.k1 * (1 - self.b + self.b * lengths[ids] / avg_len)
            result[ids] += idf * tfs * (self.k1 + 1) / (tfs + norm)
        return result


def evidence_features(text: str) -> dict[str, bool]:
    """Evidence-type flags used by intent-aware ranking."""
    numeric = bool(extract_numbers(text))
    return {
        "causal": contains_marker(text, CAUSAL_MARKERS),
        "comparison": contains_marker(text, COMPARISON_MARKERS),
        "numeric": numeric,
        "limitation": contains_marker(text, LIMITATION_MARKERS),
    }


class EvidenceIndex:
    """Holds all ingested evidence units for the current session."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.units: list[EvidenceUnit] = []
        self.documents: list[SourceDocument] = []
        self.unit_terms: list[set[str]] = []
        self.unit_features: list[dict[str, bool]] = []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25()
        self._neighbors: dict[str, tuple[int | None, int | None]] = {}

    def __len__(self) -> int:
        return len(self.units)

    @property
    def sources(self) -> list[str]:
        return [d.source for d in self.documents]

    def add_document(self, document: SourceDocument) -> int:
        """Add a source; re-adding the same source name replaces it."""
        if document.source in self.sources:
            self.remove_source(document.source)
        start = len(self.units)
        vectors = self.embedder.encode([u.text for u in document.units])
        token_lists = [tokenize(u.text) for u in document.units]
        self.units.extend(document.units)
        self.documents.append(document)
        # Section headings give context ("5. Limitations") that the sentence itself
        # may omit; they count for key-term coverage but not for BM25.
        self.unit_terms.extend(set(t) | set(tokenize(u.section or ""))
                               for t, u in zip(token_lists, document.units))
        self.unit_features.extend(evidence_features(u.text) for u in document.units)
        self.embeddings = vectors if start == 0 else np.vstack([self.embeddings, vectors])
        self.bm25.add(token_lists)
        for offset, unit in enumerate(document.units):
            idx = start + offset
            prev_idx = idx - 1 if offset > 0 else None
            next_idx = idx + 1 if offset < len(document.units) - 1 else None
            self._neighbors[unit.evidence_id] = (prev_idx, next_idx)
        return len(document.units)

    def remove_source(self, source: str) -> None:
        remaining = [d for d in self.documents if d.source != source]
        self.clear()
        for document in remaining:
            self.add_document(document)

    def clear(self) -> None:
        self.units, self.documents, self.unit_terms, self.unit_features = [], [], [], []
        self.embeddings = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25()
        self._neighbors = {}

    def neighbors(self, unit_index: int) -> list[int]:
        prev_idx, next_idx = self._neighbors[self.units[unit_index].evidence_id]
        return [i for i in (prev_idx, next_idx) if i is not None]

    def semantic_scores(self, query_vector: np.ndarray) -> np.ndarray:
        if not self.units:
            return np.zeros(0, dtype=np.float32)
        return self.embeddings @ query_vector

    def lexical_scores(self, query_terms: list[str]) -> np.ndarray:
        """BM25 scores rescaled to [0, 1] by the best achievable score for the query."""
        raw = self.bm25.scores(query_terms)
        if not len(raw):
            return raw
        # Normalize by the score a unit containing every query term once would get,
        # so that the scale does not depend on the best match in this corpus.
        ideal = sum(self.bm25.idf(t) for t in set(query_terms)) * (self.bm25.k1 + 1) / (1 + self.bm25.k1)
        if ideal <= 0:
            return np.zeros_like(raw)
        return np.clip(raw / ideal, 0.0, 1.0)

    def term_coverage(self, unit_index: int, query_terms: list[str]) -> float:
        if not query_terms:
            return 0.0
        terms = self.unit_terms[unit_index]
        return sum(1 for t in query_terms if t in terms) / len(query_terms)
