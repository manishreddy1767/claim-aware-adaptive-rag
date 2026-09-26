"""Streamlit interface for Claim-Aware Adaptive RAG.

    streamlit run app.py
"""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from carag.config import RAGConfig
from carag.ingestion import SUPPORTED_EXTENSIONS, IngestionError
from carag.models import ModelLoadError, resolve_device
from carag.schema import ClaimStatus, ClaimVerification

st.set_page_config(page_title="Claim-Aware Adaptive RAG", layout="wide")

STATUS_STYLE = {
    ClaimStatus.SUPPORTED: ("✅", "green", "The sources support this claim."),
    ClaimStatus.PARTIALLY_SUPPORTED: ("🟡", "orange", "Only part of this claim is supported."),
    ClaimStatus.CONTRADICTED: ("❌", "red", "The sources contradict this claim."),
    ClaimStatus.INSUFFICIENT_EVIDENCE: ("⚪", "gray", "The sources do not establish this claim."),
    ClaimStatus.UNCERTAIN: ("❔", "violet", "Verification was inconclusive."),
}


def get_rag():
    """One evidence index per browser session. Model weights are cached per process
    in carag.models, so additional sessions do not load duplicate models."""
    if "rag" not in st.session_state:
        from carag.pipeline import ClaimAwareRAG
        with st.spinner("Loading the embedding model (the first run downloads it)..."):
            st.session_state.rag = ClaimAwareRAG(RAGConfig())
        st.session_state.ingest_log = []
    return st.session_state.rag


def sidebar_config() -> RAGConfig:
    cfg = copy.deepcopy(RAGConfig())
    with st.sidebar:
        st.header("Settings")
        st.caption(f"Compute device: **{resolve_device()}**")
        with st.expander("Retrieval", expanded=False):
            cfg.retrieval.max_k = st.slider("Maximum evidence sentences", 3, 20, cfg.retrieval.max_k)
            cfg.retrieval.initial_k = st.slider("Initial evidence sentences", 1, cfg.retrieval.max_k,
                                                min(cfg.retrieval.initial_k, cfg.retrieval.max_k))
            cfg.retrieval.sufficiency_score = st.slider("Evidence sufficiency threshold", 0.2, 0.8,
                                                        cfg.retrieval.sufficiency_score, 0.05,
                                                        help="Below this best-evidence score the system expands "
                                                             "retrieval and may abstain.")
            cfg.retrieval.sufficiency_coverage = st.slider("Required key-term coverage", 0.0, 1.0,
                                                           cfg.retrieval.sufficiency_coverage, 0.05)
            cfg.retrieval.semantic_weight = st.slider("Semantic vs. keyword weight", 0.0, 1.0,
                                                      cfg.retrieval.semantic_weight, 0.05)
            cfg.retrieval.lexical_weight = round(1.0 - cfg.retrieval.semantic_weight, 2)
        with st.expander("Models", expanded=False):
            cfg.models.nli_model = st.selectbox(
                "Claim-checking model",
                ["cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-deberta-v3-small"],
                help="base is more accurate (measured); small uses about half the GPU memory.")
            cfg.answer.relevance_check = st.checkbox(
                "Check that the evidence actually answers the question", cfg.answer.relevance_check,
                help="Uses a small extractive QA model; greatly reduces answers to unanswerable questions.")
        with st.expander("Verification", expanded=False):
            cfg.verification.support_threshold = st.slider("Support threshold (P entailment)", 0.5, 0.95,
                                                           cfg.verification.support_threshold, 0.05)
            cfg.verification.contradiction_threshold = st.slider("Contradiction threshold (P contradiction)",
                                                                 0.5, 0.95,
                                                                 cfg.verification.contradiction_threshold, 0.05)
            cfg.verification.relevance_threshold = st.slider("Evidence relevance threshold", 0.1, 0.8,
                                                             cfg.verification.relevance_threshold, 0.05)
        with st.expander("Evidence budget", expanded=False):
            cfg.budget.enabled = st.checkbox("Share an evidence budget across claims", cfg.budget.enabled,
                                             help="Verifies claims with a limited number of evidence checks, "
                                                  "spending more on claims that are hard to settle.")
            cfg.budget.checks_per_claim = st.slider("Evidence checks per claim (average)", 1.0, 12.0,
                                                    cfg.budget.checks_per_claim, 0.5)
            cfg.budget.strategy = st.selectbox("Scheduling", ["priority", "round_robin"],
                                               help="priority = evidence-gain-aware (default); "
                                                    "round_robin = baseline")
        with st.expander("Answering", expanded=False):
            cfg.answer.max_answer_sentences = st.slider("Max answer sentences", 1, 6,
                                                        cfg.answer.max_answer_sentences)
            cfg.answer.check_question_premise = st.checkbox("Check the question's assumptions",
                                                            cfg.answer.check_question_premise)
    return cfg


def render_claim(v: ClaimVerification, key: str) -> None:
    icon, color, meaning = STATUS_STYLE[v.status]
    st.markdown(f"{icon} :{color}[**{v.status.value.replace('_', ' ')}**] — {v.claim}")
    with st.expander("Why?", expanded=False):
        st.write(meaning + " " + v.explanation)
        for title, items in (("Supporting evidence", v.supporting), ("Contradicting evidence", v.contradicting)):
            if items:
                st.markdown(f"**{title}**")
                for j in items:
                    st.markdown(f"> {j.premise}\n\n`{' + '.join(j.evidence_ids)}` · "
                                f"entail {j.entailment:.2f} · contradict {j.contradiction:.2f} · "
                                f"relevance {j.relevance:.2f}")
        for part in v.parts:
            st.markdown(f"- Part: *{part.claim}* → **{part.status.value}**")
        if not v.supporting and not v.contradicting:
            st.caption("No evidence passage met the support or contradiction threshold.")


def render_budget(report) -> None:
    st.markdown(f"**Evidence budget:** used {report.used_total} of {report.budget} evidence checks"
                + (" (budget exhausted)" if report.exhausted else ""))
    st.dataframe(pd.DataFrame([{
        "Claim": c["claim"], "Priority": c["base_priority"], "Retrieval steps": c["attempts"],
        "Checks": f"{c['checks']}/{c['candidates']}", "Evidence gain": c["total_gain"], "Result": c["status"],
    } for c in report.claims]), hide_index=True, use_container_width=True)


def ingestion_panel(rag) -> None:
    st.subheader("1. Add sources")
    col1, col2 = st.columns(2)
    with col1:
        files = st.file_uploader("Upload documents", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                                 accept_multiple_files=True)
        if st.button("Add uploaded files", disabled=not files):
            for f in files:
                try:
                    with st.spinner(f"Reading {f.name}..."):
                        doc = rag.add_bytes(f.getvalue(), f.name)
                    st.session_state.ingest_log.append(("ok", f"{f.name}: {len(doc.units)} sentences"
                                                        + (f" from {doc.pages} pages" if doc.pages else "")))
                    for w in doc.warnings:
                        st.session_state.ingest_log.append(("warn", f"{f.name}: {w}"))
                except IngestionError as exc:
                    st.session_state.ingest_log.append(("error", str(exc)))
    with col2:
        url = st.text_input("Or enter a webpage URL", placeholder="https://en.wikipedia.org/wiki/Mars")
        if st.button("Add webpage", disabled=not url):
            try:
                with st.spinner("Fetching page..."):
                    doc = rag.add_source(url)
                st.session_state.ingest_log.append(("ok", f"{url}: {len(doc.units)} sentences"))
            except IngestionError as exc:
                st.session_state.ingest_log.append(("error", str(exc)))

    for level, message in st.session_state.ingest_log[-6:]:
        {"ok": st.success, "warn": st.warning, "error": st.error}[level](message)

    if rag.index.documents:
        rows = [{"Source": d.source, "Type": d.source_type.upper(), "Pages": d.pages or "",
                 "Evidence sentences": len(d.units), "Skipped fragments": d.dropped_units}
                for d in rag.index.documents]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        remove = st.selectbox("Remove a source", ["—"] + rag.index.sources)
        if remove != "—" and st.button("Remove"):
            rag.index.remove_source(remove)
            st.rerun()


def ask_panel(rag) -> None:
    st.subheader("2. Ask a question")
    question = st.text_input("Question", placeholder="Why did Trial Two consume more energy?")
    c1, c2, c3 = st.columns(3)
    adaptive = c1.toggle("Adaptive retrieval", True, help="Off = fixed top-k semantic baseline")
    verify = c2.toggle("Claim verification", True, help="Off = answer without checking claims")
    generate = c3.toggle("Local LLM draft (experimental)", False,
                         help="Draft the answer with Qwen2.5-0.5B-Instruct, then verify and revise it. "
                              "The default extractive answers measured more accurate.")
    if not st.button("Answer", type="primary", disabled=not question):
        return
    if not rag.index.documents:
        st.warning("Add at least one document or webpage first.")
        return
    if generate:
        with st.spinner("Generating a draft with the local LLM and verifying it (first use downloads ~1 GB)..."):
            gen = rag.generate_answer(question)
        st.markdown("#### LLM draft (unverified)")
        st.caption(gen.draft)
        st.markdown("#### Verified answer")
        (st.warning if gen.abstained else st.success)(gen.final_text)
        if gen.check:
            for i, unit in enumerate(gen.check.revised.citations, start=1):
                st.markdown(f"**[{i}]** {unit.citation()} · `{unit.evidence_id}`\n\n> {unit.text}")
            for n, v in enumerate(gen.check.claims):
                render_claim(v, f"g{n}")
            if gen.check.budget:
                render_budget(gen.check.budget)
        return
    try:
        with st.spinner("Retrieving evidence and verifying claims..."):
            result = rag.ask(question, verify=verify, adaptive=adaptive)
    except ModelLoadError as exc:
        st.error(f"Model loading failed: {exc}")
        return

    if result.abstained:
        st.warning(result.answer)
    else:
        st.markdown("#### Answer")
        if result.answer_span:
            st.markdown(f"**Short answer:** {result.answer_span}")
        st.info(result.answer)
    for note in result.notes:
        st.caption(f"ℹ️ {note}")

    if result.citations:
        st.markdown("#### Sources cited")
        for c in result.citations:
            where = c.unit.citation()
            link = f"[{where}]({c.unit.url})" if c.unit.url else where
            st.markdown(f"**[{c.marker}]** {link} · `{c.unit.evidence_id}`\n\n> {c.unit.text}")

    tab_claims, tab_evidence, tab_trace = st.tabs(["Claim checks", "Retrieved evidence", "How retrieval worked"])
    with tab_claims:
        if result.premise_check:
            st.markdown("**The question's assumption**")
            render_claim(result.premise_check, "premise")
        if result.claims:
            st.markdown("**Claims in the answer**")
            for n, v in enumerate(result.claims):
                render_claim(v, f"c{n}")
        if result.removed_claims:
            st.markdown("**Removed because the sources do not establish them**")
            for n, v in enumerate(result.removed_claims):
                render_claim(v, f"r{n}")
        if not (result.premise_check or result.claims or result.removed_claims):
            st.caption("Verification was not run for this answer.")
    r = result.retrieval
    with tab_evidence:
        if r and r.evidence:
            st.dataframe(pd.DataFrame([{
                "Score": e.score, "Semantic": e.semantic, "Keyword": e.lexical, "Intent bonus": e.bonus,
                "Found by": e.found_by, "Citation": e.unit.citation(), "Text": e.unit.text,
            } for e in r.evidence]), hide_index=True, use_container_width=True)
        else:
            st.caption("No evidence retrieved.")
    with tab_trace:
        if r:
            st.write(f"Question type: **{', '.join(sorted(r.analysis.intents)) or 'factual'}** · "
                     f"key terms: `{', '.join(r.analysis.key_terms)}`")
            st.write(f"Evidence judged sufficient: **{r.sufficient}** · expanded: **{r.expanded}** · "
                     f"weak evidence trimmed: **{r.refined}**")
            for reason in r.reasons:
                st.caption(reason)
            st.json(r.trace, expanded=False)
        if result.budget:
            render_budget(result.budget)
        st.caption(f"Timings (s): {result.timings}")


def verify_panel(rag) -> None:
    st.subheader("Check any text against your sources")
    st.caption("Paste an answer (for example from a chatbot). Each factual claim is checked against the "
               "ingested sources. This measures agreement with the sources, not real-world truth.")
    text = st.text_area("Text to verify", height=140)
    if st.button("Verify claims", disabled=not text):
        if not rag.index.documents:
            st.warning("Add at least one document or webpage first.")
            return
        with st.spinner("Verifying..."):
            check = rag.check_answer(text)
        if not check.claims:
            st.info("No factual claims were found in the text.")
            return
        st.markdown("#### Revised answer")
        st.caption("Only statements supported by your sources are kept as facts; contradicted statements "
                   "are corrected and unsupported ones removed.")
        (st.warning if check.revised.abstained else st.success)(check.revised.text)
        for i, unit in enumerate(check.revised.citations, start=1):
            st.markdown(f"**[{i}]** {unit.citation()} · `{unit.evidence_id}`\n\n> {unit.text}")
        st.markdown("#### Claim checks")
        counts = pd.Series([v.status.value for v in check.claims]).value_counts()
        st.write(dict(counts))
        for n, v in enumerate(check.claims):
            render_claim(v, f"v{n}")
        if check.budget:
            render_budget(check.budget)


def main() -> None:
    st.title("Claim-Aware Adaptive RAG")
    st.caption("Answers questions using only your documents, cites the exact sentences used, checks every "
               "claim against the sources, and says so when the evidence is not sufficient.")
    config = sidebar_config()
    try:
        rag = get_rag()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()
    rag.apply_config(config)
    ingestion_panel(rag)
    st.divider()
    tab_ask, tab_verify = st.tabs(["Ask", "Verify text"])
    with tab_ask:
        ask_panel(rag)
    with tab_verify:
        verify_panel(rag)


main()
