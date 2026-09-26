"""Budget-aware verification: accuracy vs. evidence-check cost.

Claims are verified in groups (as if they were the claims of one answer) under
a shared budget of NLI evidence checks. Strategies:

* exhaustive   - every candidate passage is checked (no budget).
* fixed_split  - each claim gets budget/n checks, no early stopping
                 (a fixed top-k per claim).
* round_robin  - shared budget, claims take turns, resolved claims stop.
* priority     - shared budget, evidence-gain-aware priority scheduling
                 (carag/budget.py).

Cost is the number of NLI (premise, claim) checks actually run, counted by
the verifier, so all strategies are measured the same way.

Datasets:
* synthetic - the 30 author-labelled claims about the synthetic test PDF.
* scifact   - SciFact dev claims verified against a pooled index of all
              abstracts cited by the dev set (retrieve-then-verify setting).

    python -m evaluation.run_budget [--scifact-limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.pipeline import ClaimAwareRAG
from carag.schema import EvidenceUnit, SourceDocument

from .metrics import classification_report
from .run_scifact import GOLD_MAP, ensure_data, read_jsonl
from .run_synthetic import NOT_ESTABLISHED, collapse

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"
GROUP_SIZE = 5
BUDGETS_PER_CLAIM = (2, 3, 4, 6)
THREE = ["SUPPORTED", "CONTRADICTED", NOT_ESTABLISHED]


def groups_of(items: list, size: int, seed: int = 0) -> list[list]:
    """Deterministically shuffled groups so each group mixes labels."""
    items = list(items)
    random.Random(seed).shuffle(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


def run_strategy(rag: ClaimAwareRAG, groups: list[list[dict]], strategy: str,
                 per_claim: int | None) -> dict:
    verifier = rag.verifier
    start_checks = verifier.nli_checks
    start = time.perf_counter()
    gold, predicted = [], []
    for group in groups:
        claims = [c["claim"] for c in group]
        if strategy == "exhaustive":
            results = [verifier.verify(c) for c in claims]
        elif strategy == "fixed_split":
            results = [verifier.verify(c, max_premises=per_claim) for c in claims]
        else:
            rag.config.budget.strategy = strategy
            results, _ = rag.budgeted.verify_claims(claims, budget=per_claim * len(claims))
        gold += [c["label"] for c in group]
        predicted += [r.status.value for r in results]
    n = len(gold)
    report3 = classification_report([collapse(g) for g in gold], [collapse(p) for p in predicted], THREE)
    return {
        "strategy": strategy, "budget_per_claim": per_claim,
        "nli_checks_per_claim": round((verifier.nli_checks - start_checks) / n, 2),
        "latency_per_claim_s": round((time.perf_counter() - start) / n, 4),
        "accuracy_3way": report3["accuracy"], "macro_f1_3way": report3["macro_f1"],
        "accuracy_raw": round(sum(g == p for g, p in zip(gold, predicted)) / n, 4),
        "uncertain_labels": sum(p == "UNCERTAIN" for p in predicted),
    }


def sweep(rag: ClaimAwareRAG, groups: list[list[dict]]) -> list[dict]:
    rows = [run_strategy(rag, groups, "exhaustive", None)]
    for per_claim in BUDGETS_PER_CLAIM:
        for strategy in ("fixed_split", "round_robin", "priority"):
            rows.append(run_strategy(rag, groups, strategy, per_claim))
    rag.config.budget.strategy = "priority"
    return rows


def synthetic_claims() -> list[dict]:
    return json.loads((DATA / "synthetic_claims.json").read_text(encoding="utf-8"))["claims"]


def scifact_claims(limit: int | None) -> tuple[list[dict], SourceDocument]:
    data = ensure_data()
    corpus = {d["doc_id"]: d for d in read_jsonl(data / "corpus.jsonl")}
    claims, doc_ids = [], []
    for c in read_jsonl(data / "claims_dev.jsonl")[:limit]:
        if c["evidence"]:
            doc_id, entries = next(iter(c["evidence"].items()))
            claims.append({"claim": c["claim"], "label": GOLD_MAP[entries[0]["label"]]})
        elif c.get("cited_doc_ids"):
            claims.append({"claim": c["claim"], "label": NOT_ESTABLISHED})
        doc_ids += [int(d) for d in c.get("cited_doc_ids", [])] + [int(d) for d in c["evidence"]]
    units = []
    for doc_id in dict.fromkeys(doc_ids):
        doc = corpus.get(doc_id)
        if doc is None:
            continue
        for i, sentence in enumerate(doc["abstract"]):
            units.append(EvidenceUnit(evidence_id=f"{doc_id}.s{i}", text=sentence.strip(),
                                      source=str(doc_id), source_type="txt", position=len(units)))
    # Units of different abstracts must not be treated as neighbours for windows.
    return claims, SourceDocument("scifact-dev-pool", "txt", units)


def add_pooled(rag: ClaimAwareRAG, pooled: SourceDocument) -> None:
    by_doc: dict[str, list[EvidenceUnit]] = {}
    for u in pooled.units:
        by_doc.setdefault(u.source, []).append(u)
    for source, units in by_doc.items():
        for i, u in enumerate(units):
            u.position = i
        rag.add_document(SourceDocument(source, "txt", units))


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scifact-limit", type=int, default=None)
    parser.add_argument("--skip-scifact", action="store_true")
    parser.add_argument("--output", default=str(RESULTS / "budget_results.json"))
    args = parser.parse_args(argv)

    results = {"group_size": GROUP_SIZE, "budgets_per_claim": BUDGETS_PER_CLAIM}
    rag = ClaimAwareRAG(RAGConfig())
    rag.add_source(str(ROOT / "claim_aware_rag_test_document.pdf"))
    results["synthetic"] = sweep(rag, groups_of(synthetic_claims(), GROUP_SIZE))

    if not args.skip_scifact:
        claims, pooled = scifact_claims(args.scifact_limit)
        sci = ClaimAwareRAG(RAGConfig())
        t0 = time.perf_counter()
        add_pooled(sci, pooled)
        results["scifact_pool"] = {"claims": len(claims), "sentences": len(sci.index),
                                   "abstracts": len(sci.index.documents),
                                   "index_time_s": round(time.perf_counter() - t0, 2)}
        results["scifact"] = sweep(sci, groups_of(claims, GROUP_SIZE))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("synthetic", "scifact"):
        if name not in results:
            continue
        print(f"\n== {name}: accuracy vs. NLI evidence checks ==")
        print(f"  {'strategy':12s} {'budget/claim':>12s} {'checks/claim':>12s} {'acc3':>6s} {'macroF1':>8s} "
              f"{'raw acc':>8s} {'UNCERTAIN':>12s} {'s/claim':>8s}")
        for r in results[name]:
            print(f"  {r['strategy']:12s} {str(r['budget_per_claim'] or '-'):>12s} {r['nli_checks_per_claim']:12.2f} "
                  f"{r['accuracy_3way']:6.3f} {r['macro_f1_3way']:8.3f} {r['accuracy_raw']:8.3f} "
                  f"{r['uncertain_labels']:12d} {r['latency_per_claim_s']:8.3f}")
    print(f"\nFull results: {args.output}")
    return results


if __name__ == "__main__":
    main()
