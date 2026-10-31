"""Command-line interface.

Examples:
    python -m carag ask report.pdf https://example.org/page -q "Why did Trial Two use more energy?"
    python -m carag ask report.pdf            # interactive question loop
    python -m carag verify report.pdf --text "Trial Two used 40 Hz sampling."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import RAGConfig
from .ingestion import IngestionError
from .models import ModelLoadError
from .schema import ClaimVerification


def _print_claim(v: ClaimVerification, indent: str = "  ") -> None:
    print(f"{indent}[{v.status.value}] {v.claim}")
    print(f"{indent}    {v.explanation}")
    for label, items in (("supports", v.supporting), ("contradicts", v.contradicting)):
        for j in items[:2]:
            print(f"{indent}    {label}: {' + '.join(j.evidence_ids)}: \"{j.premise}\"")


def _print_answer(result, show_evidence: bool) -> None:
    print("\nANSWER")
    print(f"  {result.answer}")
    if result.answer_span:
        print(f"  Short answer: {result.answer_span} (QA answerability {result.answerability:.1f})")
    for note in result.notes:
        print(f"  Note: {note}")
    if result.citations:
        print("\nCITATIONS")
        for c in result.citations:
            print(f"  [{c.marker}] {c.unit.citation()} ({c.unit.evidence_id}): \"{c.unit.text}\"")
    if result.premise_check:
        print("\nQUESTION PREMISE CHECK")
        _print_claim(result.premise_check)
    if result.claims:
        print("\nCLAIM VERIFICATION")
        for v in result.claims:
            _print_claim(v)
    if result.removed_claims:
        print("\nREMOVED (not established by the sources)")
        for v in result.removed_claims:
            _print_claim(v)
    r = result.retrieval
    if r is not None:
        print(f"\nRETRIEVAL  sufficient={r.sufficient} expanded={r.expanded} refined={r.refined} "
              f"intents={sorted(r.analysis.intents) or ['factual']}")
        for step in r.trace:
            print(f"  round {step['round']}: {step['action']}")
        if show_evidence:
            for e in r.evidence:
                print(f"  {e.score:.3f} (sem {e.semantic:.2f}, lex {e.lexical:.2f}, bonus {e.bonus:.2f}) "
                      f"{e.unit.evidence_id}: {e.unit.text[:140]}")
    print(f"\nTIMINGS {result.timings}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carag", description="Claim-Aware Adaptive RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Answer questions about one or more sources")
    ask.add_argument("sources", nargs="+", help="PDF/TXT/DOCX paths or http(s) URLs")
    ask.add_argument("-q", "--question", action="append", help="Question (repeatable). Omit for interactive mode.")
    ask.add_argument("--no-verify", action="store_true", help="Skip claim verification (baseline)")
    ask.add_argument("--fixed-k", action="store_true", help="Use fixed top-k semantic retrieval (baseline)")
    ask.add_argument("--show-evidence", action="store_true", help="Print retrieved evidence with scores")
    ask.add_argument("--generate", action="store_true",
                     help="Experimental: draft the answer with a local LLM (Qwen2.5-0.5B-Instruct), "
                          "then verify and revise it. The default extractive answerer measured better.")

    verify = sub.add_parser("verify", help="Verify the claims in a piece of text against sources")
    verify.add_argument("sources", nargs="+")
    verify.add_argument("--text", required=True, help="Text whose claims should be checked")
    verify.add_argument("--budget", type=int, default=None,
                        help="Total NLI evidence checks shared by all claims (default: 4 per claim)")
    verify.add_argument("--no-budget", action="store_true", help="Verify every claim exhaustively")

    for p in (ask, verify):
        p.add_argument("--json", action="store_true", help="Print machine-readable JSON")
        p.add_argument("--device", default="auto", help="auto | cpu | cuda")
        p.add_argument("--nli-model", default=None,
                       help="NLI model (default cross-encoder/nli-deberta-v3-base; "
                            "cross-encoder/nli-deberta-v3-small uses less memory)")
        p.add_argument("--no-qa-check", action="store_true",
                       help="Disable the extractive-QA answerability check")
        p.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    config = RAGConfig()
    config.models.device = args.device
    if args.nli_model:
        config.models.nli_model = args.nli_model
    config.answer.relevance_check = not args.no_qa_check
    try:
        from .pipeline import ClaimAwareRAG
        rag = ClaimAwareRAG(config)
        for source in args.sources:
            doc = rag.add_source(source)
            print(f"Ingested {source}: {len(doc.units)} evidence sentences"
                  + (f", {doc.pages} pages" if doc.pages else ""), file=sys.stderr)
            for warning in doc.warnings:
                print(f"  warning: {warning}", file=sys.stderr)
    except (IngestionError, ModelLoadError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.command == "verify":
        rag.config.budget.enabled = not args.no_budget
        check = rag.check_answer(args.text, budget=args.budget)
        if args.json:
            print(json.dumps(check.to_dict(), indent=2))
            return 0
        if not check.claims:
            print("No factual claims were found in the text.")
            return 0
        print("CLAIMS")
        for v in check.claims:
            _print_claim(v)
        print("\nREVISED ANSWER (only source-supported statements are presented as facts)")
        print(f"  {check.revised.text}")
        for i, unit in enumerate(check.revised.citations, start=1):
            print(f"  [{i}] {unit.citation()} ({unit.evidence_id}): \"{unit.text}\"")
        if check.budget:
            b = check.budget
            print(f"\nEVIDENCE BUDGET  used {b.used_total}/{b.budget} NLI checks "
                  f"({b.extra_checks} on causal components), exhausted={b.exhausted}")
            for c in b.claims:
                print(f"  priority {c['base_priority']:.2f}  steps {c['attempts']}  checks {c['checks']}/"
                      f"{c['candidates']}  gain {c['total_gain']:.2f}  {c['status']}: {c['claim'][:70]}")
        return 0

    questions = args.question or []
    interactive = not questions
    while True:
        if interactive:
            try:
                question = input("\nQuestion (empty to quit): ").strip()
            except EOFError:
                break
            if not question:
                break
        else:
            if not questions:
                break
            question = questions.pop(0)
            print(f"\nQUESTION: {question}")
        if args.generate:
            gen = rag.generate_answer(question)
            if args.json:
                print(json.dumps(gen.to_dict(), indent=2))
                continue
            print(f"\nLLM DRAFT\n  {gen.draft}")
            print(f"\nVERIFIED ANSWER\n  {gen.final_text}")
            if gen.check:
                for i, unit in enumerate(gen.check.revised.citations, start=1):
                    print(f"  [{i}] {unit.citation()} ({unit.evidence_id}): \"{unit.text}\"")
                print("\nCLAIM VERIFICATION")
                for v in gen.check.claims:
                    _print_claim(v)
            continue
        result = rag.ask(question, verify=not args.no_verify, adaptive=not args.fixed_k)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            _print_answer(result, args.show_evidence)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
