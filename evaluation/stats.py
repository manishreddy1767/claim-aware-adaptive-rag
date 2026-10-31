"""Bootstrap confidence intervals and paired significance tests.

Resampling is done over independent units (clusters): responses for RAGTruth
(sentences within a response are correlated), claims for SciFact, questions for
SQuAD. ``metric`` maps a list of units to a number, so any metric (F1,
accuracy, macro-F1) can be bootstrapped.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def bootstrap_ci(units: Sequence[T], metric: Callable[[Sequence[T]], float], n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile (1 - alpha) confidence interval."""
    rng = random.Random(seed)
    point = metric(units)
    n = len(units)
    stats = sorted(metric([units[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = stats[int(alpha / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return point, lo, hi


def paired_bootstrap(units: Sequence[T], metric_a: Callable[[Sequence[T]], float],
                     metric_b: Callable[[Sequence[T]], float], n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    """Difference metric_a - metric_b on the same resamples.

    p_value is the two-sided bootstrap p-value for 'no difference'
    (2 x the smaller tail fraction of resampled differences crossing 0).
    """
    rng = random.Random(seed)
    n = len(units)
    point = metric_a(units) - metric_b(units)
    diffs = []
    for _ in range(n_boot):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        diffs.append(metric_a(sample) - metric_b(sample))
    diffs.sort()
    below = sum(d <= 0 for d in diffs) / n_boot
    above = sum(d >= 0 for d in diffs) / n_boot
    return {"diff": point, "ci_low": diffs[int(alpha / 2 * n_boot)],
            "ci_high": diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))],
            "p_value": min(1.0, 2 * min(below, above))}


def f1_from_pairs(pairs: Sequence[tuple[bool, bool]]) -> float:
    """F1 of the positive class from (gold, predicted) pairs."""
    tp = sum(g and p for g, p in pairs)
    fp = sum(p and not g for g, p in pairs)
    fn = sum(g and not p for g, p in pairs)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def accuracy_from_pairs(pairs: Sequence[tuple]) -> float:
    return sum(g == p for g, p in pairs) / len(pairs) if pairs else 0.0


def macro_f1_from_pairs(pairs: Sequence[tuple[str, str]], labels: Sequence[str]) -> float:
    scores = []
    for label in labels:
        if not any(g == label for g, _ in pairs):
            continue
        scores.append(f1_from_pairs([(g == label, p == label) for g, p in pairs]))
    return sum(scores) / len(scores) if scores else 0.0
