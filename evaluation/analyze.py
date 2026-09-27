"""Confidence intervals and paired significance tests for the headline results.

Reads the per-item outputs saved by the evaluation scripts and writes
evaluation/results/analysis.json. 95% percentile bootstrap CIs (2,000
resamples); paired bootstrap p-values. Resampling units: responses (RAGTruth),
claims (SciFact, budget), questions (SQuAD).

    python -m evaluation.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

from .stats import (accuracy_from_pairs, bootstrap_ci, f1_from_pairs, macro_f1_from_pairs,
                    paired_bootstrap)

RESULTS = Path(__file__).resolve().parent / "results"
THREE = ["SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED"]


def _load(name: str) -> dict | None:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ci(units, metric) -> dict:
    point, lo, hi = bootstrap_ci(units, metric)
    return {"value": round(point, 4), "ci": [round(lo, 4), round(hi, 4)]}


def _paired(units, a, b) -> dict:
    r = paired_bootstrap(units, a, b)
    return {k: round(v, 4) for k, v in r.items()}


def ragtruth(data: dict) -> dict:
    rows = data["rows"]
    systems = [k for k in rows[0] if k not in ("response_id", "task", "model", "sentence", "gold", "n_claims")]
    by_resp: dict[str, list] = {}
    for r in rows:
        by_resp.setdefault(r["response_id"], []).append(r)
    units = list(by_resp.values())      # cluster = response

    def flagged(r, s):
        return r[s] not in (None, "SUPPORTED")

    def sent_f1(s):
        return lambda us: f1_from_pairs([(r["gold"] is not None, flagged(r, s)) for u in us for r in u])

    def resp_f1(s):
        return lambda us: f1_from_pairs([(any(r["gold"] for r in u), any(flagged(r, s) for r in u)) for u in us])

    out = {"n_responses": len(units), "sentence_f1": {}, "response_f1": {}, "paired": {}}
    for s in systems:
        out["sentence_f1"][s] = _ci(units, sent_f1(s))
        out["response_f1"][s] = _ci(units, resp_f1(s))
    comparisons = [(a, b) for a in systems if a.startswith("claim_aware")
                   for b in ("similarity_threshold", "nli_top1", "minicheck", "hhem", "claim_aware") if b in systems and a != b]
    for a, b in comparisons:
        out["paired"][f"{a} vs {b}"] = {"sentence_f1": _paired(units, sent_f1(a), sent_f1(b)),
                                        "response_f1": _paired(units, resp_f1(a), resp_f1(b))}
    return out


def scifact(data: dict) -> dict:
    items = data["verification"].get("items")
    if not items:
        return {}
    systems3 = ["similarity_threshold", "nli_top1", "claim_aware_verifier"]
    out = {"n_claims": len(items), "three_way_macro_f1": {}, "binary_macro_f1": {}, "paired": {}}

    def m3(s):
        return lambda us: macro_f1_from_pairs([(u["gold"], u[s]) for u in us], THREE)

    def binary_pred(u, s):
        return u[s] >= 0.5 if isinstance(u[s], float) else u[s] == "SUPPORTED"

    def mb(s):
        return lambda us: macro_f1_from_pairs(
            [(str(u["gold"] == "SUPPORTED"), str(binary_pred(u, s))) for u in us], ["True", "False"])

    for s in systems3:
        out["three_way_macro_f1"][s] = _ci(items, m3(s))
    binary_systems = systems3 + [k for k in ("minicheck_deberta_large", "hhem_2_1_open") if k in items[0]]
    for s in binary_systems:
        out["binary_macro_f1"][s] = _ci(items, mb(s))
    for b in ("similarity_threshold", "nli_top1"):
        out["paired"][f"claim_aware_verifier vs {b} (3-way macro-F1)"] = _paired(
            items, m3("claim_aware_verifier"), m3(b))
    for b in binary_systems:
        if b != "claim_aware_verifier":
            out["paired"][f"claim_aware_verifier vs {b} (binary macro-F1)"] = _paired(
                items, mb("claim_aware_verifier"), mb(b))
    return out


def squad(data: dict) -> dict:
    rows = [r for r in data["rows"] if r["split"] == "test"]
    by_q: dict[str, dict] = {}
    for r in rows:
        by_q.setdefault(r["id"], {})[r["system"]] = r
    units = list(by_q.values())
    systems = list(units[0])

    def acc(s):
        return lambda us: sum(u[s]["correct"] for u in us) / len(us)

    def false_answer(s):
        return lambda us: (sum(not u[s]["abstained"] for u in us if not u[s]["answerable"])
                           / max(1, sum(not u[s]["answerable"] for u in us)))

    out = {"n_questions": len(units), "overall_accuracy": {}, "false_answer_rate": {}, "paired": {}}
    for s in systems:
        out["overall_accuracy"][s] = _ci(units, acc(s))
        out["false_answer_rate"][s] = _ci(units, false_answer(s))
    full = "full_system_with_qa_answerability"
    for b in systems:
        if b != full:
            out["paired"][f"{full} vs {b} (overall accuracy)"] = _paired(units, acc(full), acc(b))
    return out


def budget(data: dict) -> dict:
    out = {}
    for dataset in ("synthetic", "scifact"):
        runs = data.get(dataset) or []
        if not runs or "items" not in runs[0]:
            continue
        exhaustive = next(r for r in runs if r["strategy"] == "exhaustive")
        n = len(exhaustive["items"])
        res = {"exhaustive": {"checks_per_claim": exhaustive["nli_checks_per_claim"],
                              "macro_f1": _ci(exhaustive["items"], lambda us: macro_f1_from_pairs(
                                  [(u["gold"], _c(u["predicted"])) for u in us], THREE))}}
        for r in runs:
            if r["strategy"] == "exhaustive":
                continue
            key = f"{r['strategy']}@{r['budget_per_claim']}"
            units = [(e, x) for e, x in zip(exhaustive["items"], r["items"])]
            assert len(units) == n

            def mf(idx):
                return lambda us: macro_f1_from_pairs([(u[idx]["gold"], _c(u[idx]["predicted"])) for u in us], THREE)

            res[key] = {"checks_per_claim": r["nli_checks_per_claim"],
                        "macro_f1": _ci([x for _, x in units], lambda us: macro_f1_from_pairs(
                            [(u["gold"], _c(u["predicted"])) for u in us], THREE)),
                        "vs_exhaustive": _paired(units, mf(1), mf(0))}
        # priority vs round-robin at equal budget
        for b in sorted({r["budget_per_claim"] for r in runs if r["budget_per_claim"]}):
            pr = next(r for r in runs if r["strategy"] == "priority" and r["budget_per_claim"] == b)
            rr = next(r for r in runs if r["strategy"] == "round_robin" and r["budget_per_claim"] == b)
            units = list(zip(pr["items"], rr["items"]))
            res[f"priority vs round_robin @{b}"] = _paired(
                units, lambda us: macro_f1_from_pairs([(a["gold"], _c(a["predicted"])) for a, _ in us], THREE),
                lambda us: macro_f1_from_pairs([(b_["gold"], _c(b_["predicted"])) for _, b_ in us], THREE))
        out[dataset] = res
    return out


def ablations() -> dict:
    """RAGTruth ablations: each variant vs. the full fine-tuned verifier on the same responses."""
    files = sorted(RESULTS.glob("ragtruth_ablation_*.json"))
    if not files:
        return {}
    variants = {f.stem.removeprefix("ragtruth_ablation_"): json.loads(f.read_text(encoding="utf-8"))["rows"]
                for f in files}
    if "full" not in variants:
        return {}

    def clusters(rows):
        out: dict[str, list] = {}
        for r in rows:
            out.setdefault(r["response_id"], []).append(r)
        return list(out.values())

    def sent_f1(system, idx):
        return lambda us: f1_from_pairs([(r["gold"] is not None, r[system] not in (None, "SUPPORTED"))
                                         for u in us for r in u[idx]])

    def resp_f1(system, idx):
        return lambda us: f1_from_pairs([(any(r["gold"] for r in u[idx]),
                                          any(r[system] not in (None, "SUPPORTED") for r in u[idx])) for u in us])

    full = clusters(variants["full"])
    out = {}
    for name, rows in variants.items():
        units = list(zip(full, clusters(rows)))
        assert all(a[0]["response_id"] == b[0]["response_id"] for a, b in units)
        entry = {}
        for system in ("claim_aware", "claim_aware_budgeted"):
            entry[system] = {
                "sentence_f1": _ci(units, sent_f1(system, 1)),
                "response_f1": _ci(units, resp_f1(system, 1)),
            }
            if name != "full":
                entry[system]["vs_full_sentence_f1"] = _paired(units, sent_f1(system, 1), sent_f1(system, 0))
        out[name] = entry
    return out


def _c(label: str) -> str:
    return label if label in ("SUPPORTED", "CONTRADICTED") else "NOT_ESTABLISHED"


def main() -> dict:
    analysis = {}
    for name, fn, file in (("ragtruth", ragtruth, "ragtruth_results.json"),
                           ("scifact", scifact, "scifact_results.json"),
                           ("squad", squad, "squad_results.json"),
                           ("budget", budget, "budget_results.json")):
        data = _load(file)
        if data is not None:
            print(f"analyzing {file} ...", flush=True)
            analysis[name] = fn(data)
    abl = ablations()
    if abl:
        analysis["ragtruth_ablations"] = abl
    ft = _load("scifact_results_ft.json")
    if ft is not None and ft["verification"].get("items"):
        analysis["scifact_finetuned_verifier"] = scifact(ft)
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    for name, section in analysis.items():
        print(f"\n== {name} ==")
        for key, value in section.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {key:22s} {k:55s} {v}")
    return analysis


if __name__ == "__main__":
    main()
