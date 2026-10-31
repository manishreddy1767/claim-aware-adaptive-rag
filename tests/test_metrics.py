import math

from evaluation.metrics import (classification_report, ndcg_at_k, precision_at_k, recall_at_k,
                                reciprocal_rank, set_precision_recall)


def test_ranking_metrics():
    ranked, relevant = ["a", "b", "c", "d"], {"b", "d"}
    assert precision_at_k(ranked, relevant, 2) == 0.5
    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert abs(ndcg_at_k(ranked, relevant, 4) - expected) < 1e-9
    assert set_precision_recall(["b"], relevant) == (1.0, 0.5)


def test_classification_report():
    report = classification_report(["S", "S", "C", "N"], ["S", "C", "C", "N"], ["S", "C", "N"])
    assert report["accuracy"] == 0.75
    assert report["per_class"]["S"] == {"precision": 1.0, "recall": 0.5, "f1": 0.6667, "support": 2}
    assert report["confusion"]["S"]["C"] == 1


def test_bootstrap_and_paired_test():
    from evaluation.stats import bootstrap_ci, f1_from_pairs, paired_bootstrap
    units = [(True, True)] * 40 + [(False, False)] * 40 + [(True, False)] * 10 + [(False, True)] * 10
    point, lo, hi = bootstrap_ci(units, f1_from_pairs, n_boot=500)
    assert lo <= point <= hi and 0.7 < point < 0.9
    # Identical systems: difference 0, not significant.
    same = paired_bootstrap([(u, u) for u in units], lambda s: f1_from_pairs([a for a, _ in s]),
                            lambda s: f1_from_pairs([b for _, b in s]), n_boot=300)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    # A perfect system vs a bad one: clearly significant.
    better = paired_bootstrap([(g, g, not g) for g, _ in units],
                              lambda s: f1_from_pairs([(g, a) for g, a, _ in s]),
                              lambda s: f1_from_pairs([(g, b) for g, _, b in s]), n_boot=300)
    assert better["diff"] > 0.9 and better["p_value"] < 0.01
