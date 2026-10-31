#!/usr/bin/env bash
# Runs every experiment reported in the paper, in order, after the verifier has
# been fine-tuned (python scripts/train_verifier.py build && ... train).
# Usage (Git Bash, from the repository root):  bash scripts/run_paper_experiments.sh
set -euo pipefail
PY=".venv/Scripts/python"
FT="models/nli-deberta-v3-base-ragtruth"
R="evaluation/results"

echo "== 1. RAGTruth test: fine-tuned verifier (merged into ragtruth_results.json as *_ft)"
$PY -u -m evaluation.run_ragtruth --split test --per-task 300 --nli-model "$FT" --multi-k 5 \
    --systems claim_aware claim_aware_strict claim_aware_budgeted --suffix _ft --merge

echo "== 2. RAGTruth ablations (fine-tuned verifier, 100 test responses per task)"
ablate () {   # name, extra args...
    local name=$1; shift
    $PY -u -m evaluation.run_ragtruth --split test --per-task 100 --nli-model "$FT" --multi-k 5 \
        --systems claim_aware claim_aware_budgeted --output "$R/ragtruth_ablation_${name}.json" "$@"
}
ablate full
ablate no_windows        --set verification.use_windows=false
ablate no_multi_sentence --set verification.use_multi_sentence=false
ablate no_decomposition  --set verification.decompose_causal=false
ablate no_relevance_gate --set verification.relevance_threshold=0 --set verification.contradiction_relevance_threshold=0
ablate top1_evidence_only --set verification.evidence_per_claim=1 --set verification.use_windows=false \
                          --set verification.use_multi_sentence=false
ablate budget_2 --set budget.checks_per_claim=2
ablate budget_6 --set budget.checks_per_claim=6

echo "== 3. SciFact: baselines incl. MiniCheck/HHEM (binary), then fine-tuned verifier (transfer)"
$PY -u -m evaluation.run_scifact --external
$PY -u -m evaluation.run_scifact --skip-retrieval --nli-model "$FT" --output "$R/scifact_results_ft.json"

echo "== 4. SQuAD 2.0 (larger sample for tighter confidence intervals)"
$PY -u -m evaluation.run_squad --per-article 20

echo "== 5. Budget sweep (per-claim items for significance tests)"
$PY -u -m evaluation.run_budget

echo "== 6. Statistics"
$PY -u -m evaluation.analyze > "$R/analysis_log.txt"
echo "done"
