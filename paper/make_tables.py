"""Generate the paper's LaTeX tables from evaluation/results/*.json (no hand-typed numbers).

    python paper/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
OUT = Path(__file__).resolve().parent / "tables"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ci(entry, digits=2):
    lo, hi = entry["ci"]
    return f"{entry['value']:.{digits}f} {{\\scriptsize[{lo:.{digits}f}, {hi:.{digits}f}]}}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name, body):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.tex").write_text(body, encoding="utf-8")
    print(f"wrote tables/{name}.tex")


NAMES = {
    "similarity_threshold": "Cosine similarity $\\geq$ 0.5",
    "nli_top1": "NLI on top-1 passage",
    "minicheck": "MiniCheck-DeBERTa-L",
    "hhem": "HHEM-2.1-Open",
    "claim_aware": "Ours, zero-shot NLI",
    "claim_aware_strict": "Ours, zero-shot, strict",
    "claim_aware_budgeted": "Ours, zero-shot, budgeted",
    "claim_aware_ft": "Ours, fine-tuned",
    "claim_aware_strict_ft": "Ours, fine-tuned, strict",
    "claim_aware_budgeted_ft": "Ours, fine-tuned, budgeted",
}


def ragtruth_table(analysis, results):
    a = analysis["ragtruth"]
    summ = results["summary"]
    rows = []
    order = ["similarity_threshold", "nli_top1", "minicheck", "hhem", "claim_aware", "claim_aware_budgeted",
             "claim_aware_ft", "claim_aware_budgeted_ft"]
    for s in [s for s in order if s in a["sentence_f1"]]:
        sp = summ["sentence_level"]["all"][s]
        calls = summ["nli_checks_per_claim"].get(s)
        calls = "--" if s in ("similarity_threshold", "minicheck", "hhem") else f"{calls:.1f}"
        rows.append(f"{NAMES[s]} & {sp['precision']:.2f} & {sp['recall']:.2f} & {ci(a['sentence_f1'][s])} & "
                    f"{ci(a['response_f1'][s])} & {calls} \\\\")
        if s in ("hhem", "claim_aware_budgeted"):
            rows.append("\\midrule")
    while rows and rows[-1] == "\\midrule":
        rows.pop()
    per_task = []
    for s in [s for s in order if s in a["sentence_f1"]]:
        per_task.append(f"{NAMES[s]} & " + " & ".join(
            f"{summ['sentence_level'][t][s]['f1']:.2f}" for t in ("QA", "Summary", "Data2txt")) + " \\\\")
    n = summ["n_responses"]
    body = f"""\\begin{{table*}}[t]
\\centering\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
 & \\multicolumn{{3}}{{c}}{{Sentence level}} & Response level & NLI calls \\\\
Detector & P & R & F1 [95\\% CI] & F1 [95\\% CI] & per claim \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Hallucination detection on RAGTruth test ({n} responses, 300 per task; {summ['n_sentences']} sentences).
A sentence is flagged if any of its claims is not \\textsc{{Supported}}; a response if any sentence is flagged.
Confidence intervals: bootstrap over responses. All detectors score the same extracted claims.}}
\\label{{tab:ragtruth}}
\\end{{table*}}
"""
    write("ragtruth", body)
    write("ragtruth_tasks", f"""\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lccc}}
\\toprule
Detector & QA & Summary & Data2txt \\\\
\\midrule
{chr(10).join(per_task)}
\\bottomrule
\\end{{tabular}}
\\caption{{Sentence-level F1 on RAGTruth test by task.}}
\\label{{tab:ragtruth-tasks}}
\\end{{table}}
""")


def budget_table(analysis, ragtruth_results):
    b = analysis.get("budget", {})
    rows = []
    for dataset, label in (("synthetic", "Synthetic claims"), ("scifact", "SciFact (pooled)")):
        if dataset not in b:
            continue
        d = b[dataset]
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\")
        rows.append(f"Exhaustive & {d['exhaustive']['checks_per_claim']:.1f} & {ci(d['exhaustive']['macro_f1'])} & -- \\\\")
        for key in ("fixed_split@3", "round_robin@3", "priority@3", "round_robin@6", "priority@6"):
            if key in d:
                e = d[key]
                vs = e["vs_exhaustive"]
                rows.append(f"{key.replace('_', '-').replace('@', ' @ ')} & {e['checks_per_claim']:.1f} & "
                            f"{ci(e['macro_f1'])} & {vs['diff']:+.3f} (p={pval(vs['p_value'])}) \\\\")
    body = f"""\\begin{{table}}[t]
\\centering\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lccc}}
\\toprule
Strategy @ budget/claim & Calls & Macro-F1 [95\\% CI] & $\\Delta$ vs.\\ exh. \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Budget-aware verification (3-way labels; claims verified in groups of five). Calls = NLI calls actually run per claim. $\\Delta$: paired bootstrap difference to exhaustive verification.}}
\\label{{tab:budget}}
\\end{{table}}
"""
    write("budget", body)
    # priority vs round robin
    rows = []
    for dataset in ("synthetic", "scifact"):
        if dataset not in b:
            continue
        for key, v in b[dataset].items():
            if key.startswith("priority vs round_robin"):
                rows.append(f"{dataset} & {key.split('@')[1]} & {v['diff']:+.3f} & "
                            f"[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}] & {pval(v['p_value'])} \\\\")
    write("priority_vs_rr", f"""\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
Dataset & Budget & $\\Delta$ macro-F1 & 95\\% CI & $p$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Priority (evidence-gain) scheduling minus round-robin scheduling at equal budget (paired bootstrap).}}
\\label{{tab:priority}}
\\end{{table}}
""")


def squad_table(analysis, squad):
    a = analysis["squad"]
    summ = squad["summary"]["test"]
    names = {"fixed_top3_no_verification": "Fixed top-3, always answer",
             "adaptive_no_verification": "Adaptive retrieval + sufficiency",
             "adaptive_with_claim_verification": "\\quad + claim \\& premise verification",
             "full_system_with_qa_answerability": "\\quad + answerability gate (full)"}
    rows = []
    for s, label in names.items():
        m = summ[s]
        rows.append(f"{label} & {ci(a['overall_accuracy'][s])} & {m['answer_sentence_accuracy']:.2f} & "
                    f"{m['over_abstention_rate']:.2f} & {ci(a['false_answer_rate'][s])} \\\\")
    n_ans = summ["full_system_with_qa_answerability"]["n_answerable"]
    n_un = summ["full_system_with_qa_answerability"]["n_unanswerable"]
    write("squad", f"""\\begin{{table*}}[t]
\\centering\\small
\\begin{{tabular}}{{lcccc}}
\\toprule
System & Overall acc.\\ [95\\% CI] & Answer acc. & Over-abstention & False-answer rate [95\\% CI] \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{End-to-end answering on SQuAD~2.0 dev, test split of articles ({n_ans} answerable + {n_un} unanswerable questions).
Answer accuracy is sentence-level (the answer contains a gold span). The answerability reader was trained on SQuAD~2.0 train, so its gain is in-distribution.}}
\\label{{tab:squad}}
\\end{{table*}}
""")


def scifact_table(analysis, analysis_ft=None):
    a = analysis.get("scifact") or {}
    if "three_way_macro_f1" not in a:
        print("skip scifact table (run evaluation.run_scifact --external first)")
        return
    rows = []
    names = {"similarity_threshold": NAMES["similarity_threshold"], "nli_top1": NAMES["nli_top1"],
             "minicheck_deberta_large": NAMES["minicheck"], "hhem_2_1_open": NAMES["hhem"],
             "claim_aware_verifier": "Ours (zero-shot NLI)"}
    for s, label in names.items():
        three = ci(a["three_way_macro_f1"][s]) if s in a["three_way_macro_f1"] else "--"
        binary = ci(a["binary_macro_f1"][s]) if s in a["binary_macro_f1"] else "--"
        rows.append(f"{label} & {three} & {binary} \\\\")
    if analysis_ft:
        rows.append(f"Ours (RAGTruth-fine-tuned) & {ci(analysis_ft['three_way_macro_f1']['claim_aware_verifier'])} & "
                    f"{ci(analysis_ft['binary_macro_f1']['claim_aware_verifier'])} \\\\")
    write("scifact", f"""\\begin{{table}}[t]
\\centering\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lcc}}
\\toprule
Verifier & 3-way macro-F1 & Binary macro-F1 \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Claim verification on SciFact dev ({a['n_claims']} claims, oracle abstract). 3-way: supported / contradicted / not established; binary: supported vs.\\ not (2-class detectors).}}
\\label{{tab:scifact}}
\\end{{table}}
""")


def ablation_table(analysis):
    abl = analysis.get("ragtruth_ablations")
    if not abl:
        return
    names = {"full": "Full (fine-tuned verifier)", "no_windows": "$-$ adjacent-sentence windows",
             "no_multi_sentence": "$-$ multi-sentence premise", "no_decomposition": "$-$ causal decomposition",
             "no_relevance_gate": "$-$ relevance gating", "top1_evidence_only": "top-1 evidence only",
             "budget_2": "budget 2 calls/claim", "budget_6": "budget 6 calls/claim"}
    rows = []
    for key, label in names.items():
        if key not in abl:
            continue
        e = abl[key]["claim_aware"] if not key.startswith("budget") else abl[key]["claim_aware_budgeted"]
        delta = e.get("vs_full_sentence_f1")
        d = f"{delta['diff']:+.3f} (p={pval(delta['p_value'])})" if delta else "--"
        rows.append(f"{label} & {ci(e['sentence_f1'])} & {ci(e['response_f1'])} & {d} \\\\")
    write("ablation", f"""\\begin{{table}}[t]
\\centering\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lccc}}
\\toprule
Variant & Sent.\\ F1 & Resp.\\ F1 & $\\Delta$ sent.\\ F1 \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Ablations on RAGTruth test (100 responses per task). Budget rows use the budgeted verifier (default 4 calls/claim in the full row's budgeted variant).}}
\\label{{tab:ablation}}
\\end{{table}}
""")


def retrieval_table():
    s = load("scifact_results.json")
    if not s or "retrieval" not in s:
        return
    rows = []
    for name, label in (("bm25", "BM25 top-10"), ("semantic", "MiniLM top-10"), ("hybrid", "Hybrid top-10"),
                        ("adaptive_hybrid", "Adaptive hybrid")):
        m = s["retrieval"]["systems"][name]
        rows.append(f"{label} & {m['p@1']:.3f} & {m['r@10']:.3f} & {m['mrr']:.3f} & {m['ndcg@10']:.3f} & "
                    f"{m['set_precision']:.3f} & {m['n_retrieved']:.1f} \\\\")
    write("retrieval", f"""\\begin{{table}}[t]
\\centering\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Retriever & P@1 & R@10 & MRR & nDCG@10 & Set P & $k$ \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Abstract retrieval on SciFact dev ({s['retrieval']['n_queries']} claims with evidence, {s['retrieval']['n_abstracts']} abstracts).}}
\\label{{tab:retrieval}}
\\end{{table}}
""")


def main():
    analysis = load("analysis.json")
    ragtruth_table(analysis, load("ragtruth_results.json"))
    budget_table(analysis, load("ragtruth_results.json"))
    squad_table(analysis, load("squad_results.json"))
    scifact_table(analysis, analysis.get("scifact_finetuned_verifier"))
    ablation_table(analysis)
    retrieval_table()


if __name__ == "__main__":
    main()
