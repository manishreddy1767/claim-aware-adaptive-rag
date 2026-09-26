"""Real-world evaluation on the SciFact dev set (Wadden et al., EMNLP 2020).

SciFact contains expert-written scientific claims, a corpus of 5,183 paper
abstracts, and gold labels (SUPPORT / CONTRADICT / no evidence) with rationale
sentences. It is downloaded once (~3 MB) into data/scifact/ (git-ignored).

1. Abstract retrieval: claims with gold evidence are used as queries against
   all abstracts. Compares BM25, semantic (MiniLM), hybrid top-k, and the
   adaptive hybrid retriever.
2. Claim verification: each claim is verified against the sentences of its
   gold (or, for no-evidence claims, first cited) abstract - the "oracle
   abstract" setting. Compares the similarity-threshold and NLI-top-1
   baselines with the claim-aware verifier, collapsed to 3 labels.

    python -m evaluation.run_scifact [--limit N]
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

from carag.config import RAGConfig
from carag.pipeline import ClaimAwareRAG
from carag.schema import EvidenceUnit, SourceDocument

from .metrics import (classification_report, mean, ndcg_at_k, precision_at_k, recall_at_k,
                      reciprocal_rank, set_precision_recall)
from .run_synthetic import NOT_ESTABLISHED, baseline_nli_top1, baseline_similarity, collapse

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "scifact"
RESULTS = Path(__file__).resolve().parent / "results"
URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
GOLD_MAP = {"SUPPORT": "SUPPORTED", "CONTRADICT": "CONTRADICTED"}


def ensure_data() -> Path:
    target = DATA_DIR / "data"
    if (target / "corpus.jsonl").exists():
        return target
    import requests
    print(f"Downloading SciFact from {URL} ...")
    response = requests.get(URL, timeout=120)
    response.raise_for_status()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        archive.extractall(DATA_DIR, filter="data")
    return target


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def abstract_document(corpus: list[dict]) -> SourceDocument:
    units = [EvidenceUnit(evidence_id=str(d["doc_id"]), text=f"{d['title']}. {' '.join(d['abstract'])}",
                          source="scifact-corpus", source_type="txt", position=i)
             for i, d in enumerate(corpus)]
    return SourceDocument(source="scifact-corpus", source_type="txt", units=units)


def sentence_document(doc: dict) -> SourceDocument:
    units = [EvidenceUnit(evidence_id=f"{doc['doc_id']}.s{i}", text=s.strip(), source=str(doc["doc_id"]),
                          source_type="txt", position=i)
             for i, s in enumerate(doc["abstract"]) if s.strip()]
    return SourceDocument(source=str(doc["doc_id"]), source_type="txt", units=units)


def evaluate_retrieval(claims: list[dict], corpus: list[dict], k: int = 10) -> dict:
    config = RAGConfig()
    config.retrieval.expand_neighbors = False   # abstracts are independent units
    rag = ClaimAwareRAG(config)
    start = time.perf_counter()
    rag.add_document(abstract_document(corpus))
    index_time = time.perf_counter() - start
    queries = [c for c in claims if c["evidence"]]

    systems = {
        "bm25": lambda q: rag.retriever.rank(q, k, "lexical"),
        "semantic": lambda q: rag.retriever.rank(q, k, "semantic"),
        "hybrid": lambda q: rag.retriever.rank(q, k, "hybrid"),
        "adaptive_hybrid": lambda q: rag.retrieve(q).evidence,
    }
    report = {"n_queries": len(queries), "n_abstracts": len(corpus), "k": k,
              "index_time_s": round(index_time, 2), "systems": {}}
    for name, run in systems.items():
        rows = []
        for c in queries:
            relevant = {str(d) for d in c["evidence"]}
            t0 = time.perf_counter()
            ranked = [e.unit.evidence_id for e in run(c["claim"])]
            latency = time.perf_counter() - t0
            sp, sr = set_precision_recall(ranked, relevant)
            rows.append({"p@1": precision_at_k(ranked, relevant, 1), "r@3": recall_at_k(ranked, relevant, 3),
                         "r@10": recall_at_k(ranked, relevant, 10), "mrr": reciprocal_rank(ranked, relevant),
                         "ndcg@10": ndcg_at_k(ranked, relevant, 10), "set_precision": sp, "set_recall": sr,
                         "n_retrieved": len(ranked), "latency_s": latency})
        report["systems"][name] = {m: mean(r[m] for r in rows) for m in rows[0]}
    return report


def evaluate_verification(claims: list[dict], corpus_by_id: dict[int, dict], nli_model: str | None = None) -> dict:
    gold, preds = [], {"similarity_threshold": [], "nli_top1": [], "claim_aware_verifier": []}
    five_way = []
    latency = {k: [] for k in preds}
    for c in claims:
        if c["evidence"]:
            doc_id, entries = next(iter(c["evidence"].items()))
            label = GOLD_MAP[entries[0]["label"]]
        elif c.get("cited_doc_ids"):
            doc_id, label = c["cited_doc_ids"][0], NOT_ESTABLISHED
        else:
            continue
        doc = corpus_by_id.get(int(doc_id))
        if doc is None:
            continue
        config = RAGConfig()
        if nli_model:
            config.models.nli_model = nli_model
        rag = ClaimAwareRAG(config)
        rag.add_document(sentence_document(doc))
        gold.append(label)
        for name, fn in (("similarity_threshold", baseline_similarity), ("nli_top1", baseline_nli_top1)):
            t0 = time.perf_counter()
            preds[name].append(fn(rag, c["claim"]))
            latency[name].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        status = rag.verify_claim(c["claim"]).status.value
        latency["claim_aware_verifier"].append(time.perf_counter() - t0)
        five_way.append(status)
        preds["claim_aware_verifier"].append(collapse(status))
    labels = ["SUPPORTED", "CONTRADICTED", NOT_ESTABLISHED]
    return {
        "n_claims": len(gold),
        "gold_distribution": {l: gold.count(l) for l in labels},
        "three_way": {name: classification_report(gold, p, labels) for name, p in preds.items()},
        "claim_aware_raw_label_distribution": {s: five_way.count(s) for s in sorted(set(five_way))},
        "latency_s": {k: mean(v) for k, v in latency.items()},
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N dev claims")
    parser.add_argument("--output", default=None)
    parser.add_argument("--nli-model", default=None, help="Override the NLI model (verification only)")
    parser.add_argument("--skip-retrieval", action="store_true")
    args = parser.parse_args(argv)

    data = ensure_data()
    corpus = read_jsonl(data / "corpus.jsonl")
    claims = read_jsonl(data / "claims_dev.jsonl")[: args.limit]
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    nli_name = args.nli_model or RAGConfig().models.nli_model
    if args.output is None:
        suffix = "" if args.nli_model is None else "_" + args.nli_model.split("/")[-1]
        args.output = str(RESULTS / f"scifact_results{suffix}.json")
    results = {"dataset": "SciFact dev (real, expert-annotated)", "source": URL, "nli_model": nli_name,
               "verification": evaluate_verification(claims, corpus_by_id, args.nli_model)}
    if not args.skip_retrieval:
        results["retrieval"] = evaluate_retrieval(claims, corpus)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n== SciFact abstract retrieval ==")
    for name, m in results.get("retrieval", {}).get("systems", {}).items():
        print(f"  {name:16s} P@1={m['p@1']:.3f} R@3={m['r@3']:.3f} R@10={m['r@10']:.3f} MRR={m['mrr']:.3f} "
              f"nDCG@10={m['ndcg@10']:.3f} setP={m['set_precision']:.3f} setR={m['set_recall']:.3f} "
              f"k={m['n_retrieved']:.1f}")
    print(f"\n== SciFact claim verification (oracle abstract, 3-way, NLI={nli_name}) ==")
    print(f"  gold distribution: {results['verification']['gold_distribution']}")
    for name, m in results["verification"]["three_way"].items():
        pc = m["per_class"]
        print(f"  {name:22s} acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f} "
              f"F1[S]={pc['SUPPORTED']['f1']:.3f} F1[C]={pc['CONTRADICTED']['f1']:.3f} "
              f"F1[NE]={pc[NOT_ESTABLISHED]['f1']:.3f}")
    print(f"\nFull results: {args.output}")
    return results


if __name__ == "__main__":
    main()
