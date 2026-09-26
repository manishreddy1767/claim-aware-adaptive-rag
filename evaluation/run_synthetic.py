"""Evaluate retrieval, claim verification and end-to-end answering on the
SYNTHETIC test document. All systems use the same models, documents and
questions.

    python -m evaluation.run_synthetic            # writes evaluation/results/synthetic_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.pipeline import ClaimAwareRAG
from carag.schema import ClaimStatus

from .metrics import (classification_report, mean, ndcg_at_k, precision_at_k, recall_at_k,
                      reciprocal_rank, set_precision_recall)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"

NOT_ESTABLISHED = "NOT_ESTABLISHED"


def collapse(label: str) -> str:
    """3-way view used to compare against baselines that cannot output partial/uncertain."""
    return label if label in ("SUPPORTED", "CONTRADICTED") else NOT_ESTABLISHED


def gold_ids(rag: ClaimAwareRAG, substrings: list[str]) -> set[str]:
    ids = set()
    for s in substrings:
        matches = [u.evidence_id for u in rag.index.units if s.lower() in u.text.lower()]
        if not matches:
            raise ValueError(f"Gold evidence substring not found in document: {s!r}")
        ids.update(matches)
    return ids


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def evaluate_retrieval(rag: ClaimAwareRAG, questions: list[dict], k: int = 5) -> dict:
    systems = {
        "fixed_top5_semantic": lambda q: [e.unit.evidence_id for e in rag.retriever.rank(q, k, "semantic")],
        "fixed_top5_bm25": lambda q: [e.unit.evidence_id for e in rag.retriever.rank(q, k, "lexical")],
        "fixed_top5_hybrid": lambda q: [e.unit.evidence_id for e in rag.retriever.rank(q, k, "hybrid")],
        "adaptive_hybrid": lambda q: [e.unit.evidence_id for e in rag.retrieve(q).evidence],
    }
    labelled = [q for q in questions if q["relevant"] and q["expected"] != "abstain"]
    report = {"n_questions": len(labelled), "k": k, "systems": {}}
    for name, run in systems.items():
        rows = []
        for q in labelled:
            relevant = gold_ids(rag, q["relevant"])
            start = time.perf_counter()
            ranked = run(q["question"])
            latency = time.perf_counter() - start
            set_p, set_r = set_precision_recall(ranked, relevant)
            rows.append({
                "p@1": precision_at_k(ranked, relevant, 1), "p@k": precision_at_k(ranked, relevant, k),
                "r@k": recall_at_k(ranked, relevant, k), "mrr": reciprocal_rank(ranked, relevant),
                "ndcg@k": ndcg_at_k(ranked, relevant, k), "set_precision": set_p, "set_recall": set_r,
                "n_retrieved": len(ranked), "latency_s": latency,
            })
        report["systems"][name] = {m: mean(r[m] for r in rows) for m in rows[0]}
    return report


# ---------------------------------------------------------------------------
# Claim verification
# ---------------------------------------------------------------------------

def baseline_similarity(rag: ClaimAwareRAG, claim: str, threshold: float = 0.5) -> str:
    """Original prototype logic: treat high cosine similarity as support (no NLI)."""
    top = rag.retriever.rank(claim, 1, "semantic")
    return "SUPPORTED" if top and top[0].semantic >= threshold else NOT_ESTABLISHED


def baseline_nli_top1(rag: ClaimAwareRAG, claim: str) -> str:
    """Original prototype logic: NLI on the single most similar passage, top label >= 0.7."""
    top = rag.retriever.rank(claim, 1, "semantic")
    if not top:
        return NOT_ESTABLISHED
    probs = rag.verifier.nli.predict([(top[0].unit.text, claim)])[0]
    label, p = max(probs.items(), key=lambda kv: kv[1])
    if p < 0.7 or label == "neutral":
        return NOT_ESTABLISHED
    return "SUPPORTED" if label == "entailment" else "CONTRADICTED"


def evaluate_verification(rag: ClaimAwareRAG, claims: list[dict]) -> dict:
    gold5 = [c["label"] for c in claims]
    gold3 = [collapse(g) for g in gold5]
    predictions = {"similarity_threshold": [], "nli_top1": [], "claim_aware_verifier": []}
    details = []
    latency = {k: [] for k in predictions}
    for c in claims:
        for name, fn in (("similarity_threshold", baseline_similarity), ("nli_top1", baseline_nli_top1)):
            start = time.perf_counter()
            predictions[name].append(fn(rag, c["claim"]))
            latency[name].append(time.perf_counter() - start)
        start = time.perf_counter()
        v = rag.verify_claim(c["claim"])
        latency["claim_aware_verifier"].append(time.perf_counter() - start)
        predictions["claim_aware_verifier"].append(v.status.value)
        details.append({"claim": c["claim"], "gold": c["label"], "predicted": v.status.value,
                        "explanation": v.explanation})
    three_labels = ["SUPPORTED", "CONTRADICTED", NOT_ESTABLISHED]
    report = {"n_claims": len(claims), "three_way": {}, "latency_s": {k: mean(v) for k, v in latency.items()}}
    for name, preds in predictions.items():
        report["three_way"][name] = classification_report(gold3, [collapse(p) for p in preds], three_labels)
    report["five_way_claim_aware_verifier"] = classification_report(
        gold5, predictions["claim_aware_verifier"], [s.value for s in ClaimStatus])
    report["details"] = details
    return report


# ---------------------------------------------------------------------------
# End-to-end answering
# ---------------------------------------------------------------------------

def naive_answer(rag: ClaimAwareRAG, question: str) -> dict:
    """Baseline: fixed top-3 semantic sentences, always answers, no verification."""
    top = rag.retriever.rank(question, 3, "semantic")
    return {"answer": " ".join(e.unit.text for e in top), "abstained": False,
            "cited_ids": [e.unit.evidence_id for e in top], "premise_status": None}


def pipeline_answer(rag: ClaimAwareRAG, question: str, verify: bool) -> dict:
    result = rag.ask(question, verify=verify, adaptive=True)
    return {"answer": result.answer, "abstained": result.abstained,
            "cited_ids": [c.unit.evidence_id for c in result.citations],
            "premise_status": result.premise_check.status.value if result.premise_check else None,
            "n_sentences": len(result.sentences),
            "cited_sentences": sum(1 for s in result.sentences if s.citation_markers)}


def judge(q: dict, out: dict) -> str:
    """Classify the outcome: correct / wrong / false_answer / over_abstain."""
    text = out["answer"].lower()
    expected = q["expected"]
    if expected == "answer":
        if out["abstained"]:
            return "over_abstain"
        return "correct" if any(s.lower() in text for s in q["answer_contains"]) else "wrong"
    if expected == "abstain":
        if out["abstained"]:
            return "correct"
        if q["answer_contains"] and any(s.lower() in text for s in q["answer_contains"]):
            return "correct"   # e.g. correctly reports "not recorded"
        return "false_answer"
    if expected == "correct_premise":
        flagged = out["premise_status"] == "CONTRADICTED" or text.startswith("no.") or "assumption" in text
        return "correct" if flagged else ("over_abstain" if out["abstained"] else "false_answer")
    if expected == "yes":
        if text.startswith("yes"):
            return "correct"
        return "over_abstain" if out["abstained"] else "wrong"
    raise ValueError(expected)


def evaluate_answers(rag: ClaimAwareRAG, questions: list[dict]) -> dict:
    systems = {
        "fixed_top3_no_verification": lambda q: naive_answer(rag, q),
        "adaptive_no_verification": lambda q: pipeline_answer(rag, q, verify=False),
        "adaptive_with_claim_verification": lambda q: pipeline_answer(rag, q, verify=True),
    }
    report = {"n_questions": len(questions), "systems": {}}
    for name, run in systems.items():
        rows = []
        for q in questions:
            start = time.perf_counter()
            out = run(q["question"])
            latency = time.perf_counter() - start
            outcome = judge(q, out)
            # Post-hoc: verify answer claims of every system with the same verifier.
            claims = [] if out["abstained"] else rag.verify_text(out["answer"])
            rows.append({"id": q["id"], "type": q["type"], "expected": q["expected"], "outcome": outcome,
                         "answer": out["answer"], "latency_s": round(latency, 3),
                         "claim_statuses": [c.status.value for c in claims],
                         "citation_ok": all(any(u.evidence_id == cid for u in rag.index.units) for cid in out["cited_ids"])
                                        and (out["abstained"] or bool(out["cited_ids"]))})
        answerable = [r for r in rows if r["expected"] == "answer"]
        unanswerable = [r for r in rows if r["expected"] == "abstain"]
        premise = [r for r in rows if r["expected"] in ("correct_premise", "yes")]
        all_claims = [s for r in rows for s in r["claim_statuses"]]
        report["systems"][name] = {
            "overall_accuracy": mean(r["outcome"] == "correct" for r in rows),
            "answerable_accuracy": mean(r["outcome"] == "correct" for r in answerable),
            "over_abstention_rate": mean(r["outcome"] == "over_abstain" for r in answerable),
            "correct_abstention_rate": mean(r["outcome"] == "correct" for r in unanswerable),
            "false_answer_rate_on_unanswerable": mean(r["outcome"] == "false_answer" for r in unanswerable),
            "premise_question_accuracy": mean(r["outcome"] == "correct" for r in premise),
            "supported_claim_rate": mean(s == "SUPPORTED" for s in all_claims),
            "unsupported_claim_rate": mean(s in ("INSUFFICIENT_EVIDENCE", "CONTRADICTED") for s in all_claims),
            "citation_validity": mean(r["citation_ok"] for r in rows),
            "mean_latency_s": mean(r["latency_s"] for r in rows),
            "rows": rows,
        }
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default=str(ROOT / "claim_aware_rag_test_document.pdf"))
    parser.add_argument("--output", default=str(RESULTS / "synthetic_results.json"))
    args = parser.parse_args(argv)

    rag = ClaimAwareRAG(RAGConfig())
    rag.add_source(args.document)
    questions = json.loads((DATA / "synthetic_qa.json").read_text(encoding="utf-8"))["questions"]
    claims = json.loads((DATA / "synthetic_claims.json").read_text(encoding="utf-8"))["claims"]

    results = {
        "dataset": "SYNTHETIC (author-labelled; see evaluation/datasets)",
        "device": rag.embedder.device,
        "config": rag.config.to_dict(),
        "retrieval": evaluate_retrieval(rag, questions),
        "verification": evaluate_verification(rag, claims),
        "answering": evaluate_answers(rag, questions),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results)
    print(f"\nFull results: {args.output}")
    return results


def print_summary(results: dict) -> None:
    print("\n== Retrieval (synthetic) ==")
    for name, m in results["retrieval"]["systems"].items():
        print(f"  {name:22s} P@1={m['p@1']:.3f} P@5={m['p@k']:.3f} R@5={m['r@k']:.3f} MRR={m['mrr']:.3f} "
              f"nDCG@5={m['ndcg@k']:.3f} setP={m['set_precision']:.3f} setR={m['set_recall']:.3f} "
              f"k={m['n_retrieved']:.1f}")
    print("\n== Claim verification (synthetic, 3-way) ==")
    for name, m in results["verification"]["three_way"].items():
        print(f"  {name:22s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")
    five = results["verification"]["five_way_claim_aware_verifier"]
    print(f"  5-way claim-aware verifier: acc={five['accuracy']:.3f} macroF1={five['macro_f1']:.3f}")
    print("\n== End-to-end answering (synthetic) ==")
    for name, m in results["answering"]["systems"].items():
        print(f"  {name:34s} overall={m['overall_accuracy']:.3f} answerable={m['answerable_accuracy']:.3f} "
              f"abstain_ok={m['correct_abstention_rate']:.3f} false_ans={m['false_answer_rate_on_unanswerable']:.3f} "
              f"premise={m['premise_question_accuracy']:.3f} supported_claims={m['supported_claim_rate']:.3f} "
              f"latency={m['mean_latency_s']:.2f}s")


if __name__ == "__main__":
    main()
