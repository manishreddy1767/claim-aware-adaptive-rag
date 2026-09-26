"""Hallucination detection on RAGTruth (real LLM responses, human-annotated).

RAGTruth (Niu et al., ACL 2024) contains responses from GPT-3.5/4, Llama-2 and
Mistral models to retrieval-grounded tasks (QA, news summarization, and
data-to-text over structured business data). Annotators marked every
hallucinated span as *conflict* (contradicts the source) or *baseless* (not
supported by the source). Downloaded once (~37 MB) into data/ragtruth/.

Setup (test split only; nothing is tuned on RAGTruth):
* Each response is split into sentences; a sentence is gold-hallucinated if it
  overlaps an annotated span.
* Each sentence's claims are verified against that response's own source
  (passages / article / linearized business record).
* A sentence is predicted hallucinated if any of its claims is not SUPPORTED
  ("any-unsupported"); the "strict" variant flags only CONTRADICTED or
  INSUFFICIENT_EVIDENCE.
* Response-level: hallucinated if any sentence is.

Systems: similarity threshold and NLI-top-1 (the original prototype logic),
claim-aware verifier (exhaustive), claim-aware verifier under the shared budget.

    python -m evaluation.run_ragtruth [--per-task 300]
"""

from __future__ import annotations

import argparse
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

from .metrics import classification_report, mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ragtruth"
BASE_URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/"
RESULTS = Path(__file__).resolve().parent / "results"
TASKS = ("QA", "Summary", "Data2txt")
SYSTEMS = ("similarity_threshold", "nli_top1", "claim_aware", "claim_aware_strict", "claim_aware_budgeted")


def ensure_data() -> Path:
    import requests
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("response.jsonl", "source_info.jsonl"):
        if not (DATA / name).exists():
            print(f"Downloading RAGTruth {name} ...")
            r = requests.get(BASE_URL + name, timeout=300)
            r.raise_for_status()
            (DATA / name).write_bytes(r.content)
    return DATA


# ---------------------------------------------------------------------------
# Source text
# ---------------------------------------------------------------------------

def _linearize(name: str, value, path: str) -> list[str]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in _linearize(name, v, f"{path} {k}".strip())]
    if isinstance(value, list):
        return [s for v in value for s in _linearize(name, v, path)]
    label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", path).replace("_", " ").strip()
    if value is None:
        return [f"{name}'s {label} is not specified."]
    return [f"{name}'s {label} is {value}."]


def source_text(task: str, info) -> str:
    if task == "QA":
        return str(info["passages"])
    if task == "Summary":
        return str(info)
    # Data2txt: structured business record -> one sentence per field; reviews verbatim.
    name = info.get("name", "The business")
    lines = []
    for key, value in info.items():
        if key == "review_info":
            for review in value or []:
                lines.append(f"A {review.get('review_stars')}-star review from {review.get('review_date', '')[:10]} "
                             f"of {name} says:")
                lines.append(str(review.get("review_text", "")).replace("\n", " "))
        elif key == "hours" and isinstance(value, dict):
            lines += [f"{name} is open on {day} from {hours}." for day, hours in value.items()]
        elif key != "name":
            lines += _linearize(name, value, key)
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Sentence spans and gold labels
# ---------------------------------------------------------------------------

def sentence_spans(text: str) -> list[tuple[str, int, int]]:
    """Sentences with character offsets in the original response (whitespace-robust)."""
    keep = [(i, c.lower()) for i, c in enumerate(text) if c.isalnum()]
    squeezed = "".join(c for _, c in keep)
    spans, cursor = [], 0
    for sentence in split_sentences(text):
        key = "".join(c.lower() for c in sentence if c.isalnum())
        if not key:
            continue
        pos = squeezed.find(key, cursor)
        if pos < 0:
            continue
        start, end = keep[pos][0], keep[pos + len(key) - 1][0] + 1
        spans.append((sentence, start, end))
        cursor = pos + len(key)
    return spans


def gold_type(labels: list[dict], start: int, end: int) -> str | None:
    kinds = [l["label_type"] for l in labels if l["start"] < end and l["end"] > start]
    if not kinds:
        return None
    return "conflict" if any("Conflict" in k for k in kinds) else "baseless"


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

NOT_SUPPORTED = {s.value for s in ClaimStatus} - {"SUPPORTED"}
STRICT = {"CONTRADICTED", "INSUFFICIENT_EVIDENCE"}


def predict(rag: ClaimAwareRAG, claims_by_sentence: list[list[str]], systems=SYSTEMS) -> dict[str, list]:
    """Per system: one prediction per sentence (label string or None if no claim)."""
    flat = [c for cs in claims_by_sentence for c in cs]
    verifier = rag.verifier
    per_claim: dict[str, list] = {}
    counts: dict[str, int] = {}

    def run(name, fn):
        if name not in systems and not (name == "claim_aware" and "claim_aware_strict" in systems):
            per_claim[name] = ["SUPPORTED"] * len(flat)
            counts[name] = 0
            return
        before = verifier.nli_checks
        per_claim[name] = fn()
        counts[name] = verifier.nli_checks - before

    def sim():
        out = []
        for c in flat:
            top = rag.retriever.rank(c, 1, "semantic")
            out.append("SUPPORTED" if top and top[0].semantic >= 0.5 else "INSUFFICIENT_EVIDENCE")
        return out

    def nli1():
        out = []
        for c in flat:
            top = rag.retriever.rank(c, 1, "semantic")
            if not top:
                out.append("INSUFFICIENT_EVIDENCE")
                continue
            p = verifier.judge(c, [(top[0].unit.text, [top[0].unit.evidence_id], 1.0)])[0]
            label = max((("entailment", p.entailment), ("contradiction", p.contradiction),
                         ("neutral", p.neutral)), key=lambda kv: kv[1])
            out.append("SUPPORTED" if label[0] == "entailment" and label[1] >= 0.7 else
                       "CONTRADICTED" if label[0] == "contradiction" and label[1] >= 0.7 else
                       "INSUFFICIENT_EVIDENCE")
        return out

    run("similarity_threshold", sim)
    run("nli_top1", nli1)
    run("claim_aware", lambda: [verifier.verify(c).status.value for c in flat])
    run("claim_aware_budgeted", lambda: [v.status.value for v in rag.budgeted.verify_claims(flat)[0]])

    results: dict[str, list] = {"_checks": counts}
    for name, labels in list(per_claim.items()) + [("claim_aware_strict", per_claim["claim_aware"])]:
        flagged_set = STRICT if name == "claim_aware_strict" else NOT_SUPPORTED
        out, cursor = [], 0
        for cs in claims_by_sentence:
            labs = labels[cursor: cursor + len(cs)]
            cursor += len(cs)
            if not labs:
                out.append(None)
            elif any(l in flagged_set for l in labs):
                out.append("CONTRADICTED" if "CONTRADICTED" in labs else "UNSUPPORTED")
            else:
                out.append("SUPPORTED")
        results[name] = out
    return results


def prf(gold: list[bool], pred: list[bool]) -> dict:
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(p and not g for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "positives": sum(gold), "n": len(gold)}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=300)
    parser.add_argument("--split", default="test", choices=["test", "train"],
                        help="train = development split (used for design decisions); test = reported")
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    parser.add_argument("--output", default=str(RESULTS / "ragtruth_results.json"))
    args = parser.parse_args(argv)

    data = ensure_data()
    sources = {}
    for line in (data / "source_info.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        sources[s["source_id"]] = s
    responses = [json.loads(l) for l in (data / "response.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(0)
    sample = []
    for task in TASKS:
        pool = [r for r in responses if r["split"] == args.split and sources[r["source_id"]]["task_type"] == task]
        sample += rng.sample(pool, min(args.per_task, len(pool)))

    config = RAGConfig()
    config.answer.relevance_check = False   # only the verifier is evaluated here
    rags: dict[str, ClaimAwareRAG] = {}
    rows = []
    checks = {s: 0 for s in SYSTEMS}
    t0 = time.perf_counter()
    for n, r in enumerate(sample, start=1):
        src = sources[r["source_id"]]
        task = src["task_type"]
        if r["source_id"] not in rags:
            rag = ClaimAwareRAG(config)
            rag.add_document(load_bytes(source_text(task, src["source_info"]).encode("utf-8"),
                                        f"source-{r['source_id']}.txt"))
            rags = {r["source_id"]: rag}   # keep one index alive (responses are grouped loosely)
        rag = rags[r["source_id"]]
        spans = sentence_spans(r["response"])
        claims = [extract_claims(s) for s, _, _ in spans]
        preds = predict(rag, claims, args.systems)
        for name, c in preds.pop("_checks").items():
            checks[name] += c
        for i, (sentence, start, end) in enumerate(spans):
            rows.append({"response_id": r["id"], "task": task, "model": r["model"], "sentence": sentence,
                         "gold": gold_type(r["labels"], start, end), "n_claims": len(claims[i]),
                         **{name: preds[name][i] for name in SYSTEMS}})
        # Annotated spans that fall outside any sentence we extracted count as missed.
        covered = [(s, e) for _, s, e in spans]
        for label in r["labels"]:
            if not any(s < label["end"] and e > label["start"] for s, e in covered):
                rows.append({"response_id": r["id"], "task": task, "model": r["model"], "sentence": label["text"],
                             "gold": "conflict" if "Conflict" in label["label_type"] else "baseless",
                             "n_claims": 0, **{name: None for name in SYSTEMS}})
        if n % 100 == 0:
            print(f"  {n}/{len(sample)} responses ({time.perf_counter() - t0:.0f}s)")

    summary = {"n_responses": len(sample), "n_sentences": len(rows),
               "nli_checks_per_claim": {s: round(checks[s] / max(1, sum(r['n_claims'] for r in rows)), 2)
                                        for s in SYSTEMS if s != "claim_aware_strict"},
               "sentence_level": {}, "response_level": {}, "type_discrimination": {}}
    for task in TASKS + ("all",):
        sel = [r for r in rows if task == "all" or r["task"] == task]
        gold = [r["gold"] is not None for r in sel]
        summary["sentence_level"][task] = {s: prf(gold, [r[s] not in (None, "SUPPORTED") for r in sel])
                                           for s in SYSTEMS}
        by_resp: dict[str, dict] = {}
        for r in sel:
            entry = by_resp.setdefault(r["response_id"], {"gold": False, **{s: False for s in SYSTEMS}})
            entry["gold"] |= r["gold"] is not None
            for s in SYSTEMS:
                entry[s] |= r[s] not in (None, "SUPPORTED")
        summary["response_level"][task] = {
            s: prf([e["gold"] for e in by_resp.values()], [e[s] for e in by_resp.values()]) for s in SYSTEMS}
    # Among correctly flagged hallucinated sentences: does CONTRADICTED match gold 'conflict'?
    for s in ("nli_top1", "claim_aware"):
        flagged = [r for r in rows if r["gold"] and r[s] not in (None, "SUPPORTED")]
        summary["type_discrimination"][s] = classification_report(
            [r["gold"] for r in flagged],
            ["conflict" if r[s] == "CONTRADICTED" else "baseless" for r in flagged], ["conflict", "baseless"])
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"dataset": f"RAGTruth {args.split} split (sampled)", "per_task": args.per_task,
                                             "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    for level in ("sentence_level", "response_level"):
        print(f"\n== RAGTruth {level.replace('_', ' ')}: hallucination detection (P / R / F1) ==")
        for task in TASKS + ("all",):
            m = summary[level][task]
            print(f"  {task:8s} (pos {m['claim_aware']['positives']}/{m['claim_aware']['n']})  " + "  ".join(
                f"{s}={m[s]['precision']:.2f}/{m[s]['recall']:.2f}/{m[s]['f1']:.2f}" for s in SYSTEMS))
    print("\nNLI checks per claim:", summary["nli_checks_per_claim"])
    for s, rep in summary["type_discrimination"].items():
        print(f"conflict vs baseless among flagged ({s}): acc={rep['accuracy']:.3f} "
              f"F1[conflict]={rep['per_class']['conflict']['f1']:.3f} F1[baseless]={rep['per_class']['baseless']['f1']:.3f} "
              f"(n={rep['n']})")
    print(f"\nFull results: {args.output}  ({summary['runtime_s']}s)")
    return summary


if __name__ == "__main__":
    main()
