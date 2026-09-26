"""Retrieval and classification metrics (no external dependencies)."""

from __future__ import annotations

import math
from collections import Counter


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for rank, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def set_precision_recall(retrieved: list[str], relevant: set[str]) -> tuple[float, float]:
    """Precision/recall of an unranked (variable-size) evidence set."""
    hits = sum(1 for d in retrieved if d in relevant)
    precision = hits / len(retrieved) if retrieved else 0.0
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def classification_report(gold: list[str], predicted: list[str], labels: list[str] | None = None) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix."""
    labels = labels or sorted(set(gold) | set(predicted))
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, predicted):
        confusion.setdefault(g, {q: 0 for q in labels})
        confusion[g][p] = confusion[g].get(p, 0) + 1
    per_class = {}
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion[g].get(label, 0) for g in confusion if g != label)
        fn = sum(v for p, v in confusion.get(label, {}).items() if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": Counter(gold)[label]}
    supported_labels = [l for l in labels if per_class[l]["support"] > 0]
    accuracy = sum(g == p for g, p in zip(gold, predicted)) / len(gold) if gold else 0.0
    macro_f1 = (sum(per_class[l]["f1"] for l in supported_labels) / len(supported_labels)
                if supported_labels else 0.0)
    return {"accuracy": round(accuracy, 4), "macro_f1": round(macro_f1, 4), "n": len(gold),
            "per_class": per_class, "confusion": confusion, "labels": labels}


def mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0
