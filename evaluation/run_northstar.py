"""Stage-by-stage regression evaluation on the Northstar Digital handbook questions.

For each question, prints the query analysis, retrieval scores, premise check,
QA answerability and final answer, then checks the observed behaviour and the
required facts against evaluation/datasets/northstar_qa.json, overall and per
question category.

    python -m evaluation.run_northstar                    # writes evaluation/results/northstar_results.json
    python -m evaluation.run_northstar --quiet            # summary tables only
    python -m evaluation.run_northstar --quiet --budgets 0,8,12,20
        # also sweep the total answer budget (0 = automatic) and report quality vs. cost
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from carag.pipeline import ClaimAwareRAG

DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"

def trace(result) -> dict:
    r = result.retrieval
    a = r.analysis
    budget = result.budget
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
        "self_supported": [c.self_supported for c in result.claims],
        "budget": None if budget is None else {"budget": budget.budget, "used_total": budget.used_total,
                                               "retrieval_rounds": budget.retrieval_rounds},
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


def evaluate(rag: ClaimAwareRAG, cases: list[dict], quiet: bool) -> list[dict]:
    rows = []
    for case in cases:
        result = rag.ask(case["question"])
        t = trace(result)
        if not quiet:
            print_trace(case, result, t)
        text = result.answer.lower()
        facts = case["required_facts"]
        found = [f for f in facts if f in text]
        behavior = result.outcome
        rows.append({"id": case["id"], "category": case["category"], "question": case["question"],
                     "expected_behavior": case["expected_behavior"], "observed_behavior": behavior,
                     "behavior_ok": behavior == case["expected_behavior"],
                     "fact_recall": round(len(found) / len(facts), 3) if facts else 1.0,
                     "missing_facts": [f for f in facts if f not in found],
                     "answer": result.answer, "timings": result.timings, "trace": t})
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = sum(r["behavior_ok"] for r in rows)
    recall = sum(r["fact_recall"] for r in rows) / len(rows)
    supported = [f for r in rows for s, f in zip(r["trace"]["claim_statuses"], r["trace"]["self_supported"])
                 if s == "SUPPORTED"]
    budgets = [r["trace"]["budget"] for r in rows if r["trace"]["budget"]]
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
    return {
        "behavior_correct": ok, "n": len(rows), "mean_fact_recall": round(recall, 3),
        "self_supported_rate": round(sum(supported) / len(supported), 3) if supported else 0.0,
        "mean_checks_used": round(sum(b["used_total"] for b in budgets) / len(budgets), 2) if budgets else 0.0,
        "mean_retrieval_rounds": round(sum(b["retrieval_rounds"] for b in budgets) / len(budgets), 2)
        if budgets else 0.0,
        "by_category": {c: {"n": len(rs), "behavior_correct": sum(r["behavior_ok"] for r in rs),
                            "mean_fact_recall": round(sum(r["fact_recall"] for r in rs) / len(rs), 3)}
                        for c, rs in sorted(by_category.items())},
    }


def print_summary(rows: list[dict], summary: dict) -> None:
    print(f"\n{'id':<20}{'category':<15}{'expected':<16}{'observed':<16}{'ok':<5}fact recall")
    for row in rows:
        print(f"{row['id']:<20}{row['category']:<15}{row['expected_behavior']:<16}{row['observed_behavior']:<16}"
              f"{'yes' if row['behavior_ok'] else 'NO':<5}{row['fact_recall']:.2f}")
    print(f"\n{'category':<15}{'n':>3}{'behavior ok':>13}{'fact recall':>13}")
    for c, s in summary["by_category"].items():
        print(f"{c:<15}{s['n']:>3}{s['behavior_correct']:>9}/{s['n']:<3}{s['mean_fact_recall']:>12.2f}")
    print(f"\nbehavior correct: {summary['behavior_correct']}/{summary['n']}   "
          f"mean fact recall: {summary['mean_fact_recall']:.2f}   "
          f"supported claims backed only by their own sentence: {summary['self_supported_rate']:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quiet", action="store_true", help="print only the summary tables")
    parser.add_argument("--budgets", default="",
                        help="comma-separated total answer budgets to sweep (0 = automatic)")
    parser.add_argument("--out", default=str(RESULTS / "northstar_results.json"))
    args = parser.parse_args()

    rag = ClaimAwareRAG()
    for path in sorted((DATA / "northstar").glob("*.txt")):
        rag.add_source(str(path))
    cases = json.loads((DATA / "northstar_qa.json").read_text(encoding="utf-8"))["questions"]

    rows = evaluate(rag, cases, args.quiet)
    summary = summarize(rows)
    print_summary(rows, summary)
    output = {"dataset": "northstar (reconstructed)", "config": rag.config.to_dict(), **summary, "cases": rows}

    if args.budgets:
        base = rag.config
        sweep = []
        print(f"\n{'answer budget':<15}{'behavior ok':>12}{'fact recall':>13}{'checks used':>13}{'retr. rounds':>14}")
        for value in [int(v) for v in args.budgets.split(",")]:
            config = copy.deepcopy(base)
            config.budget.answer_budget = value or None
            rag.apply_config(config)
            s = summarize(evaluate(rag, cases, quiet=True))
            sweep.append({"answer_budget": value or "automatic", **{k: v for k, v in s.items() if k != "by_category"}})
            print(f"{value or 'automatic'!s:<15}{s['behavior_correct']:>8}/{s['n']:<3}{s['mean_fact_recall']:>13.2f}"
                  f"{s['mean_checks_used']:>13.2f}{s['mean_retrieval_rounds']:>14.2f}")
        rag.apply_config(base)
        output["budget_sweep"] = sweep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
