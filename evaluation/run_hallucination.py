"""Hallucination prevention: does claim verification keep false statements out of answers?

Part A - planted hallucinations (known ground truth, no LLM needed)
    True sentences from the documents are corrupted automatically:
      number      a number changed            ("24 days" -> "48 days")
      swap        a number taken from another sentence of the same documents
      negation    the statement negated        ("must" -> "must not")
      invented    a plausible fact the documents never state (hand-written list)
      generalize  an over-general subject      ("Full-time employees" -> "All employees")
    Answers of three true sentences plus one corrupted sentence go through check_answer
    (claim extraction, budgeted NLI verification, revision), at several evidence budgets and
    both scheduling strategies. Measured: corrupted sentences kept as fact (hallucination
    passed), removed or corrected (prevented), true sentences kept (retention), checks used.

Part B - real local LLMs (Ollama) drafting answers
    Questions with external gold labels: the Northstar handbook (required facts) and SQuAD 2.0
    test-split articles (answerable: gold answer strings; unanswerable: any answer is a
    hallucination). Draft (the model's text) versus shown (after verification) for each
    model, with the extractive RAG answer as a reference.

    python -m evaluation.run_hallucination                       # both parts
    python -m evaluation.run_hallucination --parts injected
    python -m evaluation.run_hallucination --parts llm --models qwen3:8b
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

from carag.claims import extract_claims
from carag.config import RAGConfig
from carag.ingestion import load_bytes
from carag.pipeline import ClaimAwareRAG
from carag.schema import ClaimStatus
from carag.text_utils import split_sentences

from .metrics import mean
from .stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "datasets"
RESULTS = Path(__file__).resolve().parent / "results"

# Plausible statements the documents never make.
INVENTED = {
    "northstar": [
        "Employees receive a free annual gym membership in addition to the wellness benefit.",
        "The company provides a monthly internet allowance of ₹1,000 for remote work.",
        "Employees can take 12 weeks of paid paternity leave.",
        "Unused wellness benefit can be carried forward to the next year.",
        "Work-from-home requests are approved automatically after three days.",
        "Employees receive a bonus of one month's salary every December.",
        "Laptops can be replaced every year on request.",
        "Business class travel is allowed for all trips longer than 3 hours.",
        "The company reimburses taxi fares to the office every day.",
        "Employees may work remotely from abroad for up to 30 days a year.",
        "Personal leave can be converted into cash at the end of the year.",
        "The company provides free lunch on all working days.",
    ],
    "synthetic": [
        "Trial Three used a sampling frequency of 5 Hz.",
        "The sensor nodes were powered by solar panels.",
        "The gateway connected to the internet over a 4G cellular link.",
        "Each sensor node cost 45 dollars.",
        "The study ran at three different rooftop sites.",
        "Node A rebooted five times during the study.",
        "Data were transmitted every 30 seconds in Trial Two.",
        "The battery in Trial One lasted 120 days.",
        "Humidity readings were calibrated against a weather station every hour.",
        "The study was funded by a national science agency.",
    ],
}
_NEGATIONS = [(r"\bmust not\b", "must"), (r"\bcannot\b", "can"), (r"\bmust\b", "must not"), (r"\bcan\b", "cannot"),
              (r"\bis not\b", "is"), (r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bwas\b", "was not"),
              (r"\bwere\b", "were not"), (r"\brequires\b", "does not require"), (r"\brequire\b", "do not require"),
              (r"\breceive\b", "do not receive"), (r"\breceives\b", "does not receive"),
              (r"\bconsumed\b", "did not consume"), (r"\brecorded\b", "did not record")]
_GENERAL = [(r"^(Full-time|Part-time|Eligible|New) employees\b", "All employees"),
            (r"^Employees\b", "All staff and contractors"),
            (r"^Trial (One|Two)\b", "Both trials"), (r"^Node [AB]\b", "Every node")]
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


def _fmt(value: float, like: str) -> str:
    if "," in like:
        return f"{int(value):,}"
    return str(int(value)) if float(value).is_integer() and "." not in like else f"{value:.1f}"


def corruptions(sentence: str, numbers_elsewhere: list[str], rng: random.Random) -> dict[str, str]:
    out = {}
    match = _NUMBER.search(sentence)
    if match:
        raw = match.group(1)
        value = float(raw.replace(",", ""))
        changed = value * 2 if value > 1 else value + 3
        out["number"] = sentence[:match.start(1)] + _fmt(changed, raw) + sentence[match.end(1):]
        others = [n for n in numbers_elsewhere if n.replace(",", "") != raw.replace(",", "")]
        if others:
            out["swap"] = sentence[:match.start(1)] + rng.choice(others) + sentence[match.end(1):]
    for pattern, repl in _NEGATIONS:
        if re.search(pattern, sentence):
            out["negation"] = re.sub(pattern, repl, sentence, count=1)
            break
    for pattern, repl in _GENERAL:
        if re.search(pattern, sentence):
            out["generalize"] = re.sub(pattern, repl, sentence, count=1)
            break
    return out


def build_corpus(name: str, config: RAGConfig) -> ClaimAwareRAG:
    rag = ClaimAwareRAG(config)
    if name == "northstar":
        for path in sorted((DATA / "northstar").glob("*.txt")):
            rag.add_source(str(path))
    else:
        rag.add_source(str(ROOT / "claim_aware_rag_test_document.pdf"))
    return rag


def planted_items(rag: ClaimAwareRAG, corpus: str, per_type: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sentences = [u.text for u in rag.index.units if len(u.text.split()) >= 6 and not u.text.endswith(":")]
    numbers = sorted({m.group(1) for s in sentences for m in _NUMBER.finditer(s)})
    by_type: dict[str, list[tuple[str, str]]] = {}
    for s in sentences:
        for kind, bad in corruptions(s, numbers, rng).items():
            if bad != s:
                by_type.setdefault(kind, []).append((s, bad))
    by_type["invented"] = [("", s) for s in INVENTED[corpus]]
    items = []
    for kind, pairs in sorted(by_type.items()):
        for original, bad in rng.sample(pairs, min(per_type, len(pairs))):
            fillers = [s for s in rng.sample(sentences, 6) if s != original][:3]
            text = fillers + [bad]
            rng.shuffle(text)
            items.append({"corpus": corpus, "type": kind, "original": original, "corrupted": bad,
                          "true": fillers, "answer": " ".join(text)})
    return items


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def sentence_outcome(sentence: str, verdicts) -> str:
    """How a sentence of the answer fared: kept (presented as fact), qualified, corrected or removed."""
    mine = [v for v in verdicts if _norm(v.claim) and _norm(v.claim) in _norm(sentence)]
    if not mine:
        return "removed"
    statuses = {v.status for v in mine}
    if statuses <= {ClaimStatus.SUPPORTED}:
        return "kept"
    if ClaimStatus.CONTRADICTED in statuses:
        return "corrected"
    if statuses & {ClaimStatus.PARTIALLY_SUPPORTED, ClaimStatus.UNCERTAIN}:
        return "qualified"
    return "removed"


BUDGETS = [("cpc1_priority", 1.0, "priority"), ("cpc2_priority", 2.0, "priority"),
           ("cpc2_round_robin", 2.0, "round_robin"), ("cpc4_priority", 4.0, "priority"),
           ("cpc4_round_robin", 4.0, "round_robin"), ("cpc8_priority", 8.0, "priority"),
           ("exhaustive", None, None)]


def run_injected(per_type: int, seed: int = 7) -> dict:
    report = {"settings": {}, "n_items": 0}
    base = RAGConfig()
    rags = {c: build_corpus(c, base) for c in ("northstar", "synthetic")}
    items = [i for c, rag in rags.items() for i in planted_items(rag, c, per_type, seed)]
    report["n_items"] = len(items)
    report["items_by_type"] = {t: sum(i["type"] == t for i in items) for t in sorted({i["type"] for i in items})}
    for name, cpc, strategy in BUDGETS:
        rows = []
        for item in items:
            rag = rags[item["corpus"]]
            config = copy.deepcopy(base)
            if cpc is None:
                config.budget.enabled = False
            else:
                config.budget.checks_per_claim, config.budget.strategy = cpc, strategy
            rag.apply_config(config)
            before = rag.verifier.nli_checks
            start = time.perf_counter()
            check = rag.check_answer(item["answer"])
            latency = time.perf_counter() - start
            bad = sentence_outcome(item["corrupted"], check.claims)
            true = [sentence_outcome(s, check.claims) for s in item["true"]]
            rows.append({"type": item["type"], "corpus": item["corpus"], "corrupted_outcome": bad,
                         "true_kept": sum(t == "kept" for t in true) / len(true),
                         "clean": bad != "kept" and all(t == "kept" for t in true),
                         "checks": rag.verifier.nli_checks - before,
                         "claims": len(check.claims), "latency_s": latency})
        rag.apply_config(base)
        report["settings"][name] = summarize_injected(rows)
        s = report["settings"][name]
        print(f"{name:<18} hallucinations passed {s['passed_rate']:.3f}  prevented {s['prevented_rate']:.3f} "
              f"(corrected {s['corrected_rate']:.3f})  true kept {s['true_retention']:.3f}  clean answers "
              f"{s['clean_answer_rate']:.3f}  checks/answer {s['checks_per_answer']:.1f}")
    return report


def summarize_injected(rows: list[dict]) -> dict:
    def rates(rs):
        n = len(rs)
        return {"n": n,
                "passed_rate": round(sum(r["corrupted_outcome"] == "kept" for r in rs) / n, 4),
                "prevented_rate": round(sum(r["corrupted_outcome"] != "kept" for r in rs) / n, 4),
                "corrected_rate": round(sum(r["corrupted_outcome"] == "corrected" for r in rs) / n, 4),
                "qualified_rate": round(sum(r["corrupted_outcome"] == "qualified" for r in rs) / n, 4),
                "removed_rate": round(sum(r["corrupted_outcome"] == "removed" for r in rs) / n, 4)}
    s = rates(rows)
    point, lo, hi = bootstrap_ci(rows, lambda u: mean(r["corrupted_outcome"] != "kept" for r in u), n_boot=1000)
    s["prevented_ci95"] = [round(lo, 4), round(hi, 4)]
    s["true_retention"] = round(mean(r["true_kept"] for r in rows), 4)
    s["clean_answer_rate"] = round(mean(r["clean"] for r in rows), 4)
    s["checks_per_answer"] = round(mean(r["checks"] for r in rows), 2)
    s["checks_per_claim"] = round(sum(r["checks"] for r in rows) / max(1, sum(r["claims"] for r in rows)), 2)
    s["latency_ms"] = round(1000 * mean(r["latency_s"] for r in rows), 1)
    s["by_type"] = {t: rates([r for r in rows if r["type"] == t]) for t in sorted({r["type"] for r in rows})}
    s["by_corpus"] = {c: rates([r for r in rows if r["corpus"] == c]) for c in sorted({r["corpus"] for r in rows})}
    return s


# ---------------------------------------------------------------------------
# Part B: real local LLMs
# ---------------------------------------------------------------------------

def contains_gold(text: str, golds: list[str]) -> bool:
    t = f" {_norm(text)} "
    return any(f" {_norm(g)} " in t for g in golds if _norm(g))


def llm_questions(squad_articles: int, per_article: int, seed: int = 21) -> list[dict]:
    """Northstar (required facts) and SQuAD test-split articles (gold answers / unanswerable)."""
    qs = []
    cases = json.loads((DATA / "northstar_qa.json").read_text(encoding="utf-8"))["questions"]
    for c in cases:
        answerable = c["expected_behavior"] in ("answer", "yes", "reject_premise", "conflict")
        qs.append({"set": "northstar", "doc": "northstar", "id": c["id"], "question": c["question"],
                   "answerable": answerable, "facts": c["required_facts"],
                   "absent": c["category"] in ("absent", "vague")})
    from .run_squad import DEV_ARTICLES, ensure_data
    articles = json.loads(ensure_data().read_text(encoding="utf-8"))["data"]
    test_articles = [i for i in range(len(articles)) if i not in DEV_ARTICLES][:squad_articles]
    for a in test_articles:
        article = articles[a]
        rng = random.Random(seed + a)
        qas = [qa for p in article["paragraphs"] for qa in p["qas"]]
        pos = rng.sample([q for q in qas if not q["is_impossible"]], per_article)
        neg = rng.sample([q for q in qas if q["is_impossible"]], per_article)
        for qa in pos + neg:
            qs.append({"set": "squad", "doc": a, "id": qa["id"], "question": qa["question"],
                       "answerable": not qa["is_impossible"], "golds": [x["text"] for x in qa["answers"]],
                       "absent": qa["is_impossible"]})
    return qs


def run_llm(models: list[str], squad_articles: int, per_article: int) -> dict:
    from carag.generation import is_no_answer
    from carag.llm import LLMError, LLMGenerator, OllamaClient
    from .run_squad import ensure_data

    articles = json.loads(ensure_data().read_text(encoding="utf-8"))["data"]
    questions = llm_questions(squad_articles, per_article)
    base = RAGConfig()
    rags: dict = {"northstar": build_corpus("northstar", base)}
    for q in questions:
        if q["set"] == "squad" and q["doc"] not in rags:
            article = articles[q["doc"]]
            rag = ClaimAwareRAG(base)
            rag.add_document(load_bytes("\n\n".join(p["context"] for p in article["paragraphs"]).encode("utf-8"),
                                        f"{article['title']}.txt"))
            rags[q["doc"]] = rag

    def correct(q, text: str, abstained: bool) -> bool:
        if q["set"] == "northstar":
            return not abstained and all(f in text.lower() for f in q["facts"]) if q["facts"] else not abstained
        return not abstained and contains_gold(text, q["golds"])

    report = {"n_questions": len(questions), "models": {}}
    systems = ["rag"] + models
    for system in systems:
        rows = []
        for q in questions:
            rag = rags[q["doc"]]
            start = time.perf_counter()
            if system == "rag":
                result = rag.ask(q["question"])
                draft, shown, abstained_shown = None, result.answer, result.abstained
                draft_abstained, removed = None, 0
            else:
                generator = LLMGenerator(system, OllamaClient())
                try:
                    a = rag.ask_with_llm(q["question"], generator)
                except LLMError as exc:
                    print(f"  {system} failed on {q['id']}: {exc}")
                    continue
                draft = a.draft
                draft_abstained = not draft or is_no_answer(draft) and len(extract_claims(draft)) <= 1
                shown, abstained_shown = a.text, a.abstained
                removed = len(a.check.revised.removed) + len(a.check.revised.corrected) if a.check else 0
            latency = time.perf_counter() - start
            row = {"set": q["set"], "id": q["id"], "answerable": q["answerable"], "latency_s": latency,
                   "shown_abstained": abstained_shown, "shown_correct": correct(q, shown, abstained_shown),
                   "removed_claims": removed}
            if draft is not None:
                row["draft_abstained"] = draft_abstained
                row["draft_correct"] = correct(q, draft, draft_abstained)
            if q["set"] == "northstar" and q["facts"] and q["answerable"]:
                row["shown_fact_recall"] = sum(f in shown.lower() for f in q["facts"]) / len(q["facts"])
                if draft is not None:
                    row["draft_fact_recall"] = sum(f in draft.lower() for f in q["facts"]) / len(q["facts"])
            rows.append(row)
        report["models"][system] = summarize_llm(rows, system != "rag")
        print_llm(system, report["models"][system])
    return report


def summarize_llm(rows: list[dict], has_draft: bool) -> dict:
    out = {"n": len(rows), "latency_s_mean": round(mean(r["latency_s"] for r in rows), 2)}
    for name, subset in (("squad_unanswerable", [r for r in rows if r["set"] == "squad" and not r["answerable"]]),
                         ("squad_answerable", [r for r in rows if r["set"] == "squad" and r["answerable"]]),
                         ("northstar_answerable", [r for r in rows if r["set"] == "northstar" and r["answerable"]]),
                         ("northstar_unanswerable", [r for r in rows if r["set"] == "northstar" and not r["answerable"]])):
        if not subset:
            continue
        s = {"n": len(subset)}
        if name.endswith("unanswerable"):
            # Any answer to an unanswerable question is a hallucination.
            s["shown_hallucination_rate"] = round(mean(not r["shown_abstained"] for r in subset), 4)
            if has_draft:
                s["draft_hallucination_rate"] = round(mean(not r["draft_abstained"] for r in subset), 4)
        else:
            s["shown_accuracy"] = round(mean(r["shown_correct"] for r in subset), 4)
            s["shown_over_abstention"] = round(mean(r["shown_abstained"] for r in subset), 4)
            if has_draft:
                s["draft_accuracy"] = round(mean(r["draft_correct"] for r in subset), 4)
            recall = [r for r in subset if "shown_fact_recall" in r]
            if recall:
                s["shown_fact_recall"] = round(mean(r["shown_fact_recall"] for r in recall), 4)
                if has_draft:
                    s["draft_fact_recall"] = round(mean(r["draft_fact_recall"] for r in recall), 4)
        if has_draft:
            s["mean_removed_claims"] = round(mean(r["removed_claims"] for r in subset), 3)
        out[name] = s
    return out


def print_llm(system: str, s: dict) -> None:
    print(f"\n{system}  (n={s['n']}, {s['latency_s_mean']} s/question)")
    for part, m in s.items():
        if isinstance(m, dict):
            print(f"  {part:<24} " + "  ".join(f"{k}={v}" for k, v in m.items()))


def write_markdown(report: dict, path: Path) -> None:
    out = ["# Hallucination prevention evaluation", "", "Generated by `python -m evaluation.run_hallucination`.", ""]
    inj = report.get("injected")
    if inj:
        out += ["## A. Planted hallucinations", "",
                f"{inj['n_items']} answers, each with three true sentences and one corrupted sentence "
                f"({', '.join(f'{t}: {n}' for t, n in inj['items_by_type'].items())}). *Passed* = the corrupted "
                "sentence was kept as a supported fact; *prevented* = removed, corrected or qualified.", "",
                "| Budget | Passed | Prevented (95% CI) | Corrected | True kept | Clean answers | Checks/answer | ms |",
                "|---|---|---|---|---|---|---|---|"]
        for name, s in inj["settings"].items():
            ci = s["prevented_ci95"]
            out.append(f"| {name} | {s['passed_rate']:.3f} | {s['prevented_rate']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | "
                       f"{s['corrected_rate']:.3f} | {s['true_retention']:.3f} | {s['clean_answer_rate']:.3f} | "
                       f"{s['checks_per_answer']:.1f} | {s['latency_ms']:.0f} |")
        default = inj["settings"].get("cpc4_priority") or next(iter(inj["settings"].values()))
        out += ["", "By corruption type (default budget, 4 checks per claim, priority):", "",
                "| Type | n | Passed | Prevented | Corrected | Qualified | Removed |", "|---|---|---|---|---|---|---|"]
        for t, r in default["by_type"].items():
            out.append(f"| {t} | {r['n']} | {r['passed_rate']:.3f} | {r['prevented_rate']:.3f} | "
                       f"{r['corrected_rate']:.3f} | {r['qualified_rate']:.3f} | {r['removed_rate']:.3f} |")
        out.append("")
    llm = report.get("llm")
    if llm:
        out += ["## B. Real local LLMs", "", f"{llm['n_questions']} questions. Draft = the model's own text; "
                "shown = after claim verification. On unanswerable questions any answer counts as a hallucination.",
                "", "| System | Part | n | Draft | Shown | Notes |", "|---|---|---|---|---|---|"]
        for system, s in llm["models"].items():
            for part, m in s.items():
                if not isinstance(m, dict):
                    continue
                if part.endswith("unanswerable"):
                    draft, shown = m.get("draft_hallucination_rate"), m["shown_hallucination_rate"]
                    label = "hallucination rate"
                else:
                    draft, shown = m.get("draft_accuracy"), m["shown_accuracy"]
                    label = "accuracy"
                extra = []
                if "shown_fact_recall" in m:
                    extra.append(f"fact recall {m.get('draft_fact_recall', '–')} → {m['shown_fact_recall']}")
                if "mean_removed_claims" in m:
                    extra.append(f"{m['mean_removed_claims']} claims removed per answer")
                fmt = lambda v: "–" if v is None else f"{v:.3f}"
                out.append(f"| {system} | {part} ({label}) | {m['n']} | {fmt(draft)} | {fmt(shown)} | "
                           f"{'; '.join(extra)} |")
        out.append("")
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", default=["injected", "llm"], choices=["injected", "llm"])
    parser.add_argument("--per-type", type=int, default=24, help="planted hallucinations per type and corpus")
    parser.add_argument("--models", nargs="+", default=None, help="Ollama models (default: those installed)")
    parser.add_argument("--squad-articles", type=int, default=6)
    parser.add_argument("--squad-per-article", type=int, default=4)
    parser.add_argument("--output", default=str(RESULTS / "hallucination_results.json"))
    args = parser.parse_args(argv)
    # Parts not re-run keep their earlier results (e.g. LLMs no longer installed).
    try:
        report = json.loads(Path(args.output).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {}
    if "injected" in args.parts:
        print("== A. planted hallucinations")
        report["injected"] = run_injected(args.per_type)
    if "llm" in args.parts:
        from carag.llm import LLMError, OllamaClient
        try:
            installed = OllamaClient().models()
        except LLMError:
            installed = []
        models = args.models or [m for m in ("qwen3:8b", "llama3.2:3b") if m in installed]
        print(f"\n== B. local LLMs: {models or 'none installed'}")
        report["llm"] = run_llm(models, args.squad_articles, args.squad_per_article)
    Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_markdown(report, Path(args.output).with_name("hallucination_report.md"))
    print(f"\nwrote {args.output} and hallucination_report.md")
    return report


if __name__ == "__main__":
    main()
