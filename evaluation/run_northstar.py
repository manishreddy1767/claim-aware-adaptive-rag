"""Stage-by-stage regression trace on the Northstar Digital handbook questions.

For each question, prints the query analysis, retrieval scores, premise check,
QA answerability and final answer, then checks the observed behaviour and the
required facts against evaluation/datasets/northstar_qa.json.

    python -m evaluation.run_northstar            # writes evaluation/results/northstar_results.json
    python -m evaluation.run_northstar --quiet    # summary table only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from carag.pipeline import ClaimAwareRAG

DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"

# Phrases that mark an answer as "the sources do not establish this" rather than a definite answer.
_NOT_SPECIFIED_MARKERS = ("not specified", "not establish", "does not specify", "do not specify")


def observed_behavior(result) -> str:
    text = result.answer.lower()
    if result.abstained:
        return "not_specified" if any(m in text for m in _NOT_SPECIFIED_MARKERS) else "abstain"
    if any(s.kind == "correction" for s in result.sentences):
        if text.startswith("no.") or "assumption is not supported" in text:
            return "reject_premise"
    if any(m in text for m in _NOT_SPECIFIED_MARKERS):
        return "not_specified"
    return "answer"


def trace(result) -> dict:
    r = result.retrieval
    a = r.analysis
    return {
        "key_terms": a.key_terms, "intents": sorted(a.intents), "premise": a.premise,
        "retrieval_sufficient": r.sufficient, "retrieval_expanded": r.expanded,
        "retrieval_reasons": r.reasons,
        "evidence": [{"score": e.score, "semantic": e.semantic, "lexical": e.lexical,
                      "coverage": e.coverage, "source": Path(e.unit.source).name, "text": e.unit.text}
                     for e in r.evidence],
        "premise_status": result.premise_check.status.value if result.premise_check else None,
        "answerability": result.answerability, "answer_span": result.answer_span,
        "notes": result.notes, "claim_statuses": [c.status.value for c in result.claims],
    }


def print_trace(case: dict, result, t: dict) -> None:
    print(f"\n=== {case['id']}: {case['question']}")
    print(f"key_terms={t['key_terms']} intents={t['intents']} premise={t['premise']!r}")
    print(f"retrieval sufficient={t['retrieval_sufficient']} expanded={t['retrieval_expanded']} "
          f"reasons={t['retrieval_reasons']}")
    for e in t["evidence"][:6]:
        print(f"   {e['score']:.2f} (sem {e['semantic']:.2f} lex {e['lexical']:.2f} cov {e['coverage']:.2f}) "
              f"[{e['source'][:6]}] {e['text'][:90]}")
    if t["premise_status"]:
        print(f"premise -> {t['premise_status']}")
    print(f"QA answerability={t['answerability']} span={t['answer_span']!r}")
    print(f"notes={t['notes'][:2]}")
    print("ANSWER:", result.answer[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print only the summary table")
    parser.add_argument("--out", default=str(RESULTS / "northstar_results.json"))
    args = parser.parse_args()

    rag = ClaimAwareRAG()
    for path in sorted((DATA / "northstar").glob("*.txt")):
        rag.add_source(str(path))
    cases = json.loads((DATA / "northstar_qa.json").read_text(encoding="utf-8"))["questions"]

    rows = []
    for case in cases:
        result = rag.ask(case["question"])
        t = trace(result)
        if not args.quiet:
            print_trace(case, result, t)
        text = result.answer.lower()
        found = [f for f in case["required_facts"] if f in text]
        behavior = observed_behavior(result)
        rows.append({"id": case["id"], "category": case["category"], "question": case["question"],
                     "expected_behavior": case["expected_behavior"], "observed_behavior": behavior,
                     "behavior_ok": behavior == case["expected_behavior"],
                     "fact_recall": round(len(found) / len(case["required_facts"]), 3),
                     "missing_facts": [f for f in case["required_facts"] if f not in found],
                     "answer": result.answer, "timings": result.timings, "trace": t})

    print(f"\n{'id':<18}{'expected':<16}{'observed':<16}{'ok':<5}fact recall")
    for row in rows:
        print(f"{row['id']:<18}{row['expected_behavior']:<16}{row['observed_behavior']:<16}"
              f"{'yes' if row['behavior_ok'] else 'NO':<5}{row['fact_recall']:.2f}")
    ok = sum(r["behavior_ok"] for r in rows)
    recall = sum(r["fact_recall"] for r in rows) / len(rows)
    print(f"\nbehavior correct: {ok}/{len(rows)}   mean fact recall: {recall:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"dataset": "northstar (reconstructed)", "config": rag.config.to_dict(),
                                          "behavior_correct": ok, "mean_fact_recall": round(recall, 3),
                                          "cases": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
