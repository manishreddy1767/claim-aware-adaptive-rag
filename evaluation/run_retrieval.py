"""Full evaluation of adaptive retrieval, against fixed top-k baselines and ablations.

Datasets (gold = the evidence sentences/abstracts that answer each question):
  northstar  handbook, 7 documents, 43 questions (gold: sentences containing each required fact)
  synthetic  claim_aware_rag_test_document.pdf, 27 questions (gold: annotated sentences)
  squad      SQuAD 2.0 dev, 35 Wikipedia articles (gold: the sentence holding the answer span)
  scifact    SciFact dev claims over ~5,000 abstracts (gold: cited abstracts)
Unanswerable questions (SQuAD impossible, absent topics, SciFact claims without evidence) are
used for the sufficiency check only.

Systems: adaptive retrieval; fixed top-k (hybrid k=3/5/10, semantic and BM25 k=5); hybrid with
the same number of results adaptive chose ("matched-k"); and ablations of each adaptive part.

Metrics per system: set recall/precision/F1, hit rate, MRR, nDCG@10, number of results,
latency. For adaptive retrieval also: recall before vs after expansion, expansion rate and
outcome, rounds, refinement rate, and sufficiency accuracy against answerability. Paired
bootstrap tests compare adaptive with fixed hybrid top-5.

    python -m evaluation.run_retrieval                     # all datasets
    python -m evaluation.run_retrieval --datasets northstar synthetic --squad-per-article 6
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.ingestion import load_bytes
from carag.pipeline import ClaimAwareRAG
from carag.text_utils import split_sentences

from .metrics import mean, ndcg_at_k, reciprocal_rank, set_precision_recall
from .stats import bootstrap_ci, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def _ablation(**changes):
    def apply(config: RAGConfig) -> RAGConfig:
        config = copy.deepcopy(config)
        for key, value in changes.items():
            setattr(config.retrieval, key, value)
        return config
    return apply


ABLATIONS = {
    "adaptive": _ablation(),
    "adaptive_no_expansion": _ablation(max_expansion_rounds=0),
    "adaptive_one_round": _ablation(max_expansion_rounds=1),
    "adaptive_no_neighbours": _ablation(expand_neighbors=False),
    "adaptive_no_missing_terms": _ablation(expand_missing_terms=False),
    "adaptive_no_refinement": _ablation(relative_cutoff=0.0),
    "adaptive_no_intent_bonus": _ablation(intent_bonus=0.0),
    "adaptive_semantic_only": _ablation(semantic_weight=1.0, lexical_weight=0.0),
    "adaptive_lexical_only": _ablation(semantic_weight=0.0, lexical_weight=1.0),
}
FIXED = {"fixed_hybrid_top3": ("hybrid", 3), "fixed_hybrid_top5": ("hybrid", 5),
         "fixed_hybrid_top10": ("hybrid", 10), "fixed_semantic_top5": ("semantic", 5),
         "fixed_bm25_top5": ("lexical", 5)}


# ---------------------------------------------------------------------------
# Datasets: lists of corpora, each {"build": () -> ClaimAwareRAG, "questions": [...]}
# question = {"id", "question", "answerable", "gold": [set of unit ids per required fact]}
# ---------------------------------------------------------------------------

def _units_with(rag: ClaimAwareRAG, text: str) -> set[str]:
    low = text.lower()
    return {u.evidence_id for u in rag.index.units if low in u.text.lower()}


def northstar(config: RAGConfig) -> list[dict]:
    rag = ClaimAwareRAG(config)
    for path in sorted((DATA / "northstar").glob("*.txt")):
        rag.add_source(str(path))
    cases = json.loads((DATA / "northstar_qa.json").read_text(encoding="utf-8"))["questions"]
    questions = []
    for c in cases:
        answerable = c["expected_behavior"] in ("answer", "yes", "reject_premise", "conflict")
        gold = [_units_with(rag, f) for f in c["required_facts"]] if answerable else []
        gold = [g for g in gold if g]
        questions.append({"id": c["id"], "category": c["category"], "question": c["question"],
                          "answerable": answerable and bool(gold), "gold": gold})
    return [{"rag": rag, "questions": questions}]


def synthetic(config: RAGConfig) -> list[dict]:
    rag = ClaimAwareRAG(config)
    rag.add_source(str(ROOT / "claim_aware_rag_test_document.pdf"))
    cases = json.loads((DATA / "synthetic_qa.json").read_text(encoding="utf-8"))["questions"]
    questions = []
    for c in cases:
        gold = [_units_with(rag, s) for s in c.get("relevant", [])]
        gold = [g for g in gold if g]
        answerable = c["expected"] != "abstain" and bool(gold)
        # All relevant sentences together are the evidence (one "fact" per sentence).
        questions.append({"id": c["id"], "category": c["type"], "question": c["question"],
                          "answerable": answerable, "gold": gold if answerable else []})
    return [{"rag": rag, "questions": questions}]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def squad(config: RAGConfig, per_article: int, seed: int = 13) -> list[dict]:
    from .run_squad import ensure_data
    articles = json.loads(ensure_data().read_text(encoding="utf-8"))["data"]
    corpora = []
    for a_index, article in enumerate(articles):
        text = "\n\n".join(p["context"] for p in article["paragraphs"])

        def build(text=text, title=article["title"]):
            rag = ClaimAwareRAG(config)
            rag.add_document(load_bytes(text.encode("utf-8"), f"{title}.txt"))
            return rag

        rng = random.Random(seed + a_index)
        answerable = [(p, q) for p in article["paragraphs"] for q in p["qas"] if not q["is_impossible"]]
        impossible = [(p, q) for p in article["paragraphs"] for q in p["qas"] if q["is_impossible"]]
        picked = (rng.sample(answerable, min(per_article, len(answerable)))
                  + rng.sample(impossible, min(max(1, per_article // 2), len(impossible))))
        corpora.append({"build": build, "picked": picked, "title": article["title"]})
    return corpora


def _squad_questions(rag: ClaimAwareRAG, corpus: dict) -> list[dict]:
    units = [(u.evidence_id, _norm(u.text)) for u in rag.index.units]
    questions = []
    for paragraph, qa in corpus["picked"]:
        gold = set()
        if not qa["is_impossible"]:
            answer = qa["answers"][0]
            # The sentence of the paragraph that holds the answer span.
            offset, sentence = 0, None
            for s in split_sentences(paragraph["context"]):
                start = paragraph["context"].find(s[:30], offset)
                if start <= answer["answer_start"] < start + len(s) + 1:
                    sentence = _norm(s)
                    break
                offset = max(offset, start)
            if sentence:
                gold = {uid for uid, text in units if text and (text == sentence or sentence in text
                                                               or (len(text) > 40 and text in sentence))}
            if not gold:
                gold = {uid for uid, text in units if _norm(answer["text"]) in text}
        questions.append({"id": qa["id"], "category": "impossible" if qa["is_impossible"] else "answerable",
                          "question": qa["question"], "answerable": bool(gold), "gold": [gold] if gold else []})
    return questions


def scifact(config: RAGConfig) -> list[dict]:
    from .run_scifact import abstract_document, ensure_data, read_jsonl
    folder = ensure_data()
    corpus = read_jsonl(folder / "corpus.jsonl")
    claims = read_jsonl(folder / "claims_dev.jsonl")
    config = copy.deepcopy(config)
    config.retrieval.expand_neighbors = False      # abstracts are independent units
    rag = ClaimAwareRAG(config)
    rag.add_document(abstract_document(corpus))
    questions = [{"id": str(c["id"]), "category": "evidence" if c["evidence"] else "no_evidence",
                  "question": c["claim"], "answerable": bool(c["evidence"]),
                  "gold": [{str(d) for d in c["evidence"]}] if c["evidence"] else []} for c in claims]
    return [{"rag": rag, "questions": questions, "neighbours_off": True}]


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _score(retrieved: list[str], gold: list[set[str]]) -> dict:
    relevant = set().union(*gold) if gold else set()
    precision, recall = set_precision_recall(retrieved, relevant)
    found = [any(g in retrieved for g in fact) for fact in gold]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"recall": recall, "precision": precision, "f1": f1,
            "fact_recall": sum(found) / len(found), "hit": float(any(found)),
            "mrr": reciprocal_rank(retrieved, relevant), "ndcg@10": ndcg_at_k(retrieved, relevant, 10)}


def evaluate_corpus(rag: ClaimAwareRAG, questions: list[dict], base: RAGConfig, systems: list[str]) -> dict:
    """Rows per system for one corpus."""
    rows: dict[str, list[dict]] = {name: [] for name in systems}
    adaptive_counts: dict[str, int] = {}
    for name in [s for s in systems if s in ABLATIONS]:
        config = ABLATIONS[name](rag.config)
        rag.retriever.config = rag.retriever.scorer.config = config.retrieval
        for q in questions:
            start = time.perf_counter()
            result = rag.retrieve(q["question"])
            latency = time.perf_counter() - start
            retrieved = [e.unit.evidence_id for e in result.evidence]
            row = {"id": q["id"], "category": q["category"], "answerable": q["answerable"],
                   "n": len(retrieved), "latency_s": latency, "sufficient": result.sufficient,
                   "expanded": result.expanded, "rounds": result.expansion_rounds,
                   "refined": result.refined}
            if q["answerable"]:
                row.update(_score(retrieved, q["gold"]))
                initial = result.trace[0]["kept"] if result.trace else []
                row["initial_recall"] = _score(initial, q["gold"])["recall"]
                row["initial_precision"] = _score(initial, q["gold"])["precision"]
                row["initial_n"] = len(initial)
                row["gold_complete"] = row["recall"] == 1.0
            rows[name].append(row)
            if name == "adaptive":
                adaptive_counts[q["id"]] = len(retrieved)
    rag.retriever.config = rag.retriever.scorer.config = rag.config.retrieval
    for name in [s for s in systems if s in FIXED or s == "matched_k_hybrid"]:
        for q in questions:
            if name == "matched_k_hybrid":
                mode, k = "hybrid", max(1, adaptive_counts.get(q["id"], 5))
            else:
                mode, k = FIXED[name]
            start = time.perf_counter()
            retrieved = [e.unit.evidence_id for e in rag.retriever.rank(q["question"], k, mode)]
            row = {"id": q["id"], "category": q["category"], "answerable": q["answerable"], "n": len(retrieved),
                   "latency_s": time.perf_counter() - start}
            if q["answerable"]:
                row.update(_score(retrieved, q["gold"]))
            rows[name].append(row)
    return rows


def _p95(values: list[float]) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)] if values else 0.0


def summarize(rows: list[dict], adaptive: bool) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    s = {"n_questions": len(rows), "n_answerable": len(answerable)}
    for m in ("recall", "fact_recall", "precision", "f1", "hit", "mrr", "ndcg@10"):
        s[m] = round(mean(r[m] for r in answerable), 4) if answerable else None
    if answerable:
        point, lo, hi = bootstrap_ci(answerable, lambda u: mean(r["recall"] for r in u), n_boot=1000)
        s["recall_ci95"] = [round(lo, 4), round(hi, 4)]
    s["mean_results"] = round(mean(r["n"] for r in answerable), 2) if answerable else None
    s["latency_ms_mean"] = round(1000 * mean(r["latency_s"] for r in rows), 1)
    s["latency_ms_p95"] = round(1000 * _p95([r["latency_s"] for r in rows]), 1)
    if not adaptive:
        return s
    expanded = [r for r in answerable if r["expanded"]]
    s["initial_recall"] = round(mean(r["initial_recall"] for r in answerable), 4) if answerable else None
    s["initial_precision"] = round(mean(r["initial_precision"] for r in answerable), 4) if answerable else None
    s["initial_mean_results"] = round(mean(r["initial_n"] for r in answerable), 2) if answerable else None
    s["expansion_rate"] = round(mean(r["expanded"] for r in rows), 4)
    s["expansion_rate_answerable"] = round(len(expanded) / len(answerable), 4) if answerable else None
    if expanded:
        gains = [r["recall"] - r["initial_recall"] for r in expanded]
        s["expanded_recall_gain_mean"] = round(statistics.fmean(gains), 4)
        s["expansion_success_rate"] = round(sum(g > 0 for g in gains) / len(gains), 4)
        s["expansion_no_gain_rate"] = round(sum(g == 0 for g in gains) / len(gains), 4)
        s["expanded_precision_change_mean"] = round(statistics.fmean(
            r["precision"] - r["initial_precision"] for r in expanded), 4)
    s["mean_rounds"] = round(mean(r["rounds"] for r in rows), 3)
    s["rounds_distribution"] = {str(k): sum(r["rounds"] == k for r in rows) for k in range(0, 3)}
    s["refinement_rate"] = round(mean(r["refined"] for r in rows), 4)
    unanswerable = [r for r in rows if not r["answerable"]]
    # Sufficiency as a classifier of answerability.
    tp = sum(r["sufficient"] for r in answerable)
    fp = sum(r["sufficient"] for r in unanswerable)
    s["sufficiency"] = {
        "accuracy": round((tp + len(unanswerable) - fp) / len(rows), 4) if rows else None,
        "flagged_sufficient_answerable": round(tp / len(answerable), 4) if answerable else None,
        "flagged_sufficient_unanswerable": round(fp / len(unanswerable), 4) if unanswerable else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        # Among answerable questions: was "sufficient" right about the gold being fully retrieved?
        "agrees_with_gold_complete": round(mean(r["sufficient"] == r["gold_complete"] for r in answerable), 4)
        if answerable else None,
    }
    return s


def compare(rows_a: list[dict], rows_b: list[dict], metric: str) -> dict:
    by_id = {r["id"]: r for r in rows_b if r["answerable"]}
    pairs = [(r, by_id[r["id"]]) for r in rows_a if r["answerable"] and r["id"] in by_id]
    result = paired_bootstrap(pairs, lambda u: mean(a[metric] for a, _ in u),
                              lambda u: mean(b[metric] for _, b in u), n_boot=2000)
    return {k: round(v, 4) for k, v in result.items()}


def run(datasets: list[str], squad_per_article: int) -> dict:
    base = RAGConfig()
    systems = list(ABLATIONS) + list(FIXED) + ["matched_k_hybrid"]
    report = {"config": base.retrieval.__dict__, "datasets": {}}
    for name in datasets:
        start = time.perf_counter()
        all_rows: dict[str, list[dict]] = {s: [] for s in systems}
        if name == "squad":
            for corpus in squad(base, squad_per_article):
                rag = corpus["build"]()
                rows = evaluate_corpus(rag, _squad_questions(rag, corpus), base, systems)
                for s in systems:
                    all_rows[s] += rows[s]
        else:
            for corpus in {"northstar": northstar, "synthetic": synthetic, "scifact": scifact}[name](base):
                rows = evaluate_corpus(corpus["rag"], corpus["questions"], base, systems)
                for s in systems:
                    all_rows[s] += rows[s]
        summary = {s: summarize(all_rows[s], s in ABLATIONS) for s in systems}
        tests = {f"adaptive_vs_{b}_{m}": compare(all_rows["adaptive"], all_rows[b], m)
                 for b in ("fixed_hybrid_top5", "matched_k_hybrid") for m in ("recall", "f1")}
        by_category = {}
        for row in all_rows["adaptive"]:
            by_category.setdefault(row["category"], []).append(row)
        report["datasets"][name] = {
            "seconds": round(time.perf_counter() - start, 1),
            "summary": summary, "significance": tests,
            "adaptive_by_category": {c: summarize(rs, True) for c, rs in sorted(by_category.items())},
            "rows": {s: all_rows[s] for s in ("adaptive", "fixed_hybrid_top5")},
        }
        print_dataset(name, report["datasets"][name])
    return report


def print_dataset(name: str, data: dict) -> None:
    s = data["summary"]
    a = s["adaptive"]
    print(f"\n=== {name}: {a['n_questions']} questions ({a['n_answerable']} answerable), {data['seconds']} s")
    print(f"{'system':<28}{'recall':>8}{'fact_r':>8}{'prec':>8}{'F1':>7}{'hit':>7}{'MRR':>7}{'nDCG':>7}"
          f"{'#res':>7}{'ms':>7}")
    for system, m in s.items():
        print(f"{system:<28}{m['recall'] or 0:>8.3f}{m['fact_recall'] or 0:>8.3f}{m['precision'] or 0:>8.3f}"
              f"{m['f1'] or 0:>7.3f}{m['hit'] or 0:>7.3f}{m['mrr'] or 0:>7.3f}{m['ndcg@10'] or 0:>7.3f}"
              f"{m['mean_results'] or 0:>7.2f}{m['latency_ms_mean']:>7.1f}")
    print(f"adaptive: recall {a['initial_recall']} -> {a['recall']} after expansion; expanded "
          f"{a['expansion_rate']:.0%} of questions; on expanded answerable ones recall +"
          f"{a.get('expanded_recall_gain_mean', 0)}, improved {a.get('expansion_success_rate', 0):.0%}, "
          f"precision {a.get('expanded_precision_change_mean', 0):+}; rounds {a['rounds_distribution']}; "
          f"refined {a['refinement_rate']:.0%}")
    print(f"sufficiency: {a['sufficiency']}")
    for test, r in data["significance"].items():
        print(f"  {test}: diff {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p_value']}")


NAMES = {
    "adaptive": "Adaptive (full)", "adaptive_no_expansion": "– no expansion", "adaptive_one_round": "– one round max",
    "adaptive_no_neighbours": "– no neighbour sentences", "adaptive_no_missing_terms": "– no missing-term search",
    "adaptive_no_refinement": "– no tail trimming", "adaptive_no_intent_bonus": "– no intent bonus",
    "adaptive_semantic_only": "– semantic score only", "adaptive_lexical_only": "– BM25 score only",
    "fixed_hybrid_top3": "Fixed hybrid top-3", "fixed_hybrid_top5": "Fixed hybrid top-5",
    "fixed_hybrid_top10": "Fixed hybrid top-10", "fixed_semantic_top5": "Fixed semantic top-5",
    "fixed_bm25_top5": "Fixed BM25 top-5", "matched_k_hybrid": "Hybrid, same count as adaptive",
}


def write_markdown(report: dict, path: Path) -> None:
    """The recorded metrics as readable tables."""
    out = ["# Adaptive retrieval evaluation", "",
           "Generated by `python -m evaluation.run_retrieval`. Recall, precision and F1 are set metrics over the "
           "retrieved evidence against the gold evidence; fact recall is the share of required facts with at least "
           "one gold sentence retrieved; nDCG@10 and MRR use the retrieval order. Answerable questions only, "
           "except the sufficiency and expansion-rate rows.", ""]
    for name, data in report["datasets"].items():
        s = data["summary"]
        a = s["adaptive"]
        out += [f"## {name}", "",
                f"{a['n_questions']} questions ({a['n_answerable']} answerable); run time {data['seconds']} s.", "",
                "| System | Recall (95% CI) | Fact recall | Precision | F1 | Hit | MRR | nDCG@10 | Results | ms |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for system, m in s.items():
            ci = m.get("recall_ci95") or [0, 0]
            out.append(f"| {NAMES.get(system, system)} | {m['recall']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | "
                       f"{m['fact_recall']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} | {m['hit']:.3f} | "
                       f"{m['mrr']:.3f} | {m['ndcg@10']:.3f} | {m['mean_results']:.2f} | {m['latency_ms_mean']:.1f} |")
        suf = a["sufficiency"]
        out += ["", "**Adaptive behaviour**", "",
                f"- Recall before expansion {a['initial_recall']:.3f} → after {a['recall']:.3f}; "
                f"results {a['initial_mean_results']:.2f} → {a['mean_results']:.2f}.",
                f"- Expanded on {a['expansion_rate']:.0%} of all questions ({a['expansion_rate_answerable']:.0%} "
                "of answerable ones); rounds used: " + ", ".join(f"{k}: {v}" for k, v in a["rounds_distribution"].items()) + ".",
                (f"- When it expanded (answerable): recall gain {a['expanded_recall_gain_mean']:+.3f}, recall improved "
                 f"in {a['expansion_success_rate']:.0%}, no gain in {a['expansion_no_gain_rate']:.0%}, precision "
                 f"change {a['expanded_precision_change_mean']:+.3f}.") if "expanded_recall_gain_mean" in a else
                "- No answerable question was expanded.",
                f"- Tail trimming removed weak results on {a['refinement_rate']:.0%} of questions.",
                f"- Sufficiency check: accuracy {suf['accuracy']:.3f} against answerability; flagged sufficient for "
                f"{suf['flagged_sufficient_answerable']:.0%} of answerable and "
                + (f"{suf['flagged_sufficient_unanswerable']:.0%}" if suf["flagged_sufficient_unanswerable"] is not None
                   else "n/a") + f" of unanswerable questions; agreed with 'all gold evidence retrieved' on "
                f"{suf['agrees_with_gold_complete']:.0%} of answerable questions.", "",
                "**Paired bootstrap (adaptive minus baseline, 2,000 resamples)**", "",
                "| Comparison | Difference | 95% CI | p |", "|---|---|---|---|"]
        for test, r in data["significance"].items():
            out.append(f"| {test.replace('adaptive_vs_', '').replace('_', ' ')} | {r['diff']:+.4f} | "
                       f"{r['ci_low']:+.4f} to {r['ci_high']:+.4f} | {r['p_value']:.3f} |")
        cats = data["adaptive_by_category"]
        out += ["", "**Adaptive by question category**", "",
                "| Category | n | Answerable | Recall | Precision | Expanded | Sufficient-flag accuracy |",
                "|---|---|---|---|---|---|---|"]
        for cat, c in cats.items():
            fmt = lambda v: "–" if v is None else f"{v:.3f}"
            out.append(f"| {cat} | {c['n_questions']} | {c['n_answerable']} | {fmt(c['recall'])} | "
                       f"{fmt(c['precision'])} | {c['expansion_rate']:.0%} | {fmt(c['sufficiency']['accuracy'])} |")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=["northstar", "synthetic", "squad", "scifact"],
                        choices=["northstar", "synthetic", "squad", "scifact"])
    parser.add_argument("--squad-per-article", type=int, default=12,
                        help="answerable questions per SQuAD article (plus half as many unanswerable)")
    parser.add_argument("--output", default=str(RESULTS / "retrieval_results.json"))
    parser.add_argument("--report-only", action="store_true", help="rewrite the Markdown report from --output")
    args = parser.parse_args(argv)
    if args.report_only:
        report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    else:
        report = run(args.datasets, args.squad_per_article)
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"\nwrote {args.output}")
    markdown = Path(args.output).with_name("retrieval_report.md")
    write_markdown(report, markdown)
    print(f"wrote {markdown}")
    return report


if __name__ == "__main__":
    main()
