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
