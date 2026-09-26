"""Does claim verification prevent hallucinations of a local LLM?

For each question a small local LLM (Qwen2.5-0.5B-Instruct) drafts an answer from
the retrieved evidence; the draft is then verified and revised (check_answer).
Draft and revised answers are scored against *external gold labels*, not the
verifier's own judgement:

* SQuAD 2.0 test-split articles (articles 17-34; answerable + unanswerable).
* The synthetic question set (development set; author labels).

Metrics (per answer version: draft, revised; extractive full system for reference)
* answer accuracy (answerable): answered and contains a gold answer string.
* false-answer (hallucination) rate (unanswerable): answered at all.
* over-abstention (answerable): abstained / declined.

    python -m evaluation.run_generation [--per-article 4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.ingestion import load_bytes
from carag.pipeline import ClaimAwareRAG

from .metrics import mean
from .run_squad import DEV_ARTICLES, contains_gold, ensure_data, sample_questions

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"


def evaluate_question(rag: ClaimAwareRAG, question: str, golds: list[str], answerable: bool,
                      acceptable: list[str] | None = None, gate: bool = True) -> dict:
    t0 = time.perf_counter()
    gen = rag.generate_answer(question, gate=gate)
    gen_latency = time.perf_counter() - t0
    extractive = rag.ask(question)
    targets = golds or acceptable or []

    def score(text: str, abstained: bool) -> str:
        if answerable:
            if abstained:
                return "over_abstain"
            return "correct" if contains_gold(text, targets) else "wrong"
        if abstained or (targets and contains_gold(text, targets)):
            return "correct"
        return "false_answer"

    return {
        "question": question, "answerable": answerable, "gold": golds,
        "draft": gen.draft, "revised": gen.final_text, "extractive": extractive.answer,
        "draft_outcome": score(gen.draft, gen.draft_declined),
        "revised_outcome": score(gen.final_text, gen.abstained),
        "extractive_outcome": score(extractive.answer, extractive.abstained),
        "claim_statuses": [c.status.value for c in gen.check.claims] if gen.check else [],
        "generation_latency_s": round(gen_latency, 3),
    }


def summarize(rows: list[dict]) -> dict:
    out = {"n": len(rows), "n_answerable": sum(r["answerable"] for r in rows),
           "n_unanswerable": sum(not r["answerable"] for r in rows)}
    ans = [r for r in rows if r["answerable"]]
    imp = [r for r in rows if not r["answerable"]]
    for version in ("draft", "revised", "extractive"):
        key = f"{version}_outcome"
        out[version] = {
            "answer_accuracy": mean(r[key] == "correct" for r in ans),
            "over_abstention": mean(r[key] == "over_abstain" for r in ans),
            "false_answer_rate": mean(r[key] == "false_answer" for r in imp),
            "overall_accuracy": mean(r[key] == "correct" for r in rows),
        }
    statuses = [s for r in rows for s in r["claim_statuses"]]
    out["draft_claim_labels"] = {s: statuses.count(s) for s in sorted(set(statuses))}
    out["mean_generation_latency_s"] = mean(r["generation_latency_s"] for r in rows)
    return out


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-article", type=int, default=4)
    parser.add_argument("--no-gate", action="store_true",
                        help="Generate even when the QA answerability check finds no answer")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    gate = not args.no_gate
    if args.output is None:
        args.output = str(RESULTS / ("generation_results.json" if gate else "generation_results_ungated.json"))

    results: dict = {"generator": "Qwen/Qwen2.5-0.5B-Instruct", "answerability_gate": gate}

    # SQuAD 2.0 test split
    articles = json.loads(ensure_data().read_text(encoding="utf-8"))["data"]
    rows = []
    for a_index, article in enumerate(articles):
        if a_index in DEV_ARTICLES:
            continue
        rag = ClaimAwareRAG(RAGConfig())
        text = "\n\n".join(p["context"] for p in article["paragraphs"])
        rag.add_document(load_bytes(text.encode("utf-8"), f"{article['title']}.txt"))
        for qa in sample_questions(article, args.per_article, 101 + a_index):
            rows.append(evaluate_question(rag, qa["question"], [a["text"] for a in qa["answers"]],
                                          not qa["is_impossible"], gate=gate))
    results["squad_test"] = {"summary": summarize(rows), "rows": rows}

    # Synthetic questions (answer / abstain types only; premise questions scored elsewhere)
    rag = ClaimAwareRAG(RAGConfig())
    rag.add_source(str(ROOT / "claim_aware_rag_test_document.pdf"))
    questions = json.loads((DATA / "synthetic_qa.json").read_text(encoding="utf-8"))["questions"]
    rows = [evaluate_question(rag, q["question"], q["answer_contains"] if q["expected"] == "answer" else [],
                              q["expected"] == "answer",
                              acceptable=q["answer_contains"] if q["expected"] == "abstain" else None, gate=gate)
            for q in questions if q["expected"] in ("answer", "abstain")]
    results["synthetic"] = {"summary": summarize(rows), "rows": rows}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("squad_test", "synthetic"):
        s = results[name]["summary"]
        print(f"\n== {name} (n={s['n_answerable']} answerable + {s['n_unanswerable']} unanswerable) ==")
        for version in ("draft", "revised", "extractive"):
            m = s[version]
            print(f"  {version:10s} answer_acc={m['answer_accuracy']:.3f} over_abstain={m['over_abstention']:.3f} "
                  f"false_answer={m['false_answer_rate']:.3f} overall={m['overall_accuracy']:.3f}")
        print(f"  draft claim labels: {s['draft_claim_labels']}  gen latency {s['mean_generation_latency_s']:.2f}s")
    print(f"\nFull results: {args.output}")
    return results


if __name__ == "__main__":
    main()
