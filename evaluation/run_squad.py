"""End-to-end answering and abstention on SQuAD 2.0 dev (real data).

SQuAD 2.0 (Rajpurkar et al., 2018) pairs Wikipedia articles with crowd-written
questions; about half are *unanswerable* - written to look answerable from the
article (plausible distractor spans exist) but not actually answered by it.

Setup: each article is ingested as one document (all paragraphs), so retrieval
must find the right sentence among ~100-300. For each article a fixed random
sample of answerable and unanswerable questions is asked.

Split: articles 0-16 = dev (may be inspected when developing), articles 17-34 =
test (reported, never used for tuning).

Metrics
* answer-sentence accuracy (answerable): the system answers and the answer text
  contains a gold answer span (case/whitespace-insensitive). Sentence-level, so
  more lenient than SQuAD exact match.
* evidence recall (answerable): a retrieved evidence sentence contains a gold span.
* correct-abstention rate (unanswerable): the system abstains.
* false-answer rate (unanswerable) = 1 - correct-abstention rate.
* over-abstention rate (answerable): the system abstains.
* overall accuracy: answerable correct + unanswerable abstained, over all questions.

    python -m evaluation.run_squad [--per-article 6] [--systems ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.ingestion import load_bytes
from carag.pipeline import ClaimAwareRAG

from .metrics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "squad" / "dev-v2.0.json"
URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
RESULTS = Path(__file__).resolve().parent / "results"
DEV_ARTICLES = range(0, 17)


def ensure_data() -> Path:
    if not DATA.exists():
        import requests
        print(f"Downloading SQuAD 2.0 dev from {URL} ...")
        DATA.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        DATA.write_bytes(response.content)
    return DATA


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def contains_gold(text: str, golds: list[str]) -> bool:
    t = f" {norm(text)} "
    return any(f" {norm(g)} " in t for g in golds if norm(g))


def sample_questions(article: dict, per_kind: int, seed: int) -> list[dict]:
    qas = [qa for p in article["paragraphs"] for qa in p["qas"]]
    rng = random.Random(seed)
    answerable = [q for q in qas if not q["is_impossible"]]
    impossible = [q for q in qas if q["is_impossible"]]
    return rng.sample(answerable, min(per_kind, len(answerable))) + rng.sample(impossible, min(per_kind, len(impossible)))


# name -> (ask() kwargs, QA answerability check on?)
SYSTEMS = {
    "fixed_top3_no_verification": None,
    "adaptive_no_verification": (dict(verify=False), False),
    "adaptive_with_claim_verification": (dict(verify=True), False),
    "full_system_with_qa_answerability": (dict(verify=True), True),
}


def answer_fixed(rag: ClaimAwareRAG, question: str):
    top = rag.retriever.rank(question, 3, "semantic")
    return " ".join(e.unit.text for e in top), False, [e.unit.text for e in top]


def run(per_article: int, systems: list[str], config: RAGConfig, seed: int = 13) -> dict:
    articles = json.loads(ensure_data().read_text(encoding="utf-8"))["data"]
    rows = []
    for a_index, article in enumerate(articles):
        text = "\n\n".join(p["context"] for p in article["paragraphs"])
        rag = ClaimAwareRAG(config)
        rag.add_document(load_bytes(text.encode("utf-8"), f"{article['title']}.txt"))
        split = "dev" if a_index in DEV_ARTICLES else "test"
        for qa in sample_questions(article, per_article, seed + a_index):
            golds = [a["text"] for a in qa["answers"]]
            for name in systems:
                t0 = time.perf_counter()
                if SYSTEMS[name] is None:
                    answer, abstained, evidence = answer_fixed(rag, qa["question"])
                else:
                    kwargs, use_qa = SYSTEMS[name]
                    if rag.config.answer.relevance_check != use_qa:
                        cfg = copy.deepcopy(rag.config)
                        cfg.answer.relevance_check = use_qa
                        rag.apply_config(cfg)
                    result = rag.ask(qa["question"], **kwargs)
                    answer, abstained = result.answer, result.abstained
                    evidence = [e.unit.text for e in result.retrieval.evidence] if result.retrieval else []
                latency = time.perf_counter() - t0
                rows.append({
                    "system": name, "split": split, "article": article["title"], "id": qa["id"],
                    "question": qa["question"], "answerable": not qa["is_impossible"], "gold": golds,
                    "answer": answer, "abstained": abstained, "latency_s": round(latency, 4),
                    "correct": (not abstained and contains_gold(answer, golds)) if golds else abstained,
                    "evidence_hit": contains_gold(" ".join(evidence), golds) if golds else None,
                })
    return {"rows": rows, "summary": summarize(rows, systems)}


def summarize(rows: list[dict], systems: list[str]) -> dict:
    summary = {}
    for split in ("dev", "test", "all"):
        summary[split] = {}
        for name in systems:
            sel = [r for r in rows if r["system"] == name and (split == "all" or r["split"] == split)]
            ans = [r for r in sel if r["answerable"]]
            imp = [r for r in sel if not r["answerable"]]
            summary[split][name] = {
                "n_answerable": len(ans), "n_unanswerable": len(imp),
                "overall_accuracy": mean(r["correct"] for r in sel),
                "answer_sentence_accuracy": mean(r["correct"] for r in ans),
                "evidence_recall": mean(r["evidence_hit"] for r in ans),
                "over_abstention_rate": mean(r["abstained"] for r in ans),
                "correct_abstention_rate": mean(r["abstained"] for r in imp),
                "false_answer_rate": mean(not r["abstained"] for r in imp),
                "mean_latency_s": mean(r["latency_s"] for r in sel),
            }
    return summary


def print_summary(summary: dict) -> None:
    for split in ("dev", "test"):
        print(f"\n== SQuAD 2.0 {split} split ==")
        for name, m in summary[split].items():
            print(f"  {name:34s} overall={m['overall_accuracy']:.3f} answer_acc={m['answer_sentence_accuracy']:.3f} "
                  f"evid_recall={m['evidence_recall']:.3f} over_abstain={m['over_abstention_rate']:.3f} "
                  f"abstain_ok={m['correct_abstention_rate']:.3f} false_ans={m['false_answer_rate']:.3f} "
                  f"lat={m['mean_latency_s']:.2f}s (n={m['n_answerable']}+{m['n_unanswerable']})")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-article", type=int, default=6, help="answerable and unanswerable questions per article")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--output", default=str(RESULTS / "squad_results.json"))
    args = parser.parse_args(argv)
    results = run(args.per_article, args.systems, RAGConfig())
    results["dataset"] = "SQuAD 2.0 dev (real, crowd-annotated)"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print_summary(results["summary"])
    print(f"\nFull results: {args.output}")
    return results


if __name__ == "__main__":
    main()
