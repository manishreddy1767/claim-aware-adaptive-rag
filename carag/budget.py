"""Budget-aware, evidence-gain-driven claim verification.

Verifying every claim against every candidate passage is wasteful: most claims
are settled by their first one or two passages, while a few need more. This
module spends a *shared* budget of NLI evidence checks across all claims of an
answer, following the scheduling design of the original prototype
(evidence_allocator.py):

* Each claim gets a base priority
      priority = w * evidence_gap + (1 - w) * linguistic_uncertainty
  where evidence_gap = 1 - (similarity of the best candidate passage) and
  linguistic_uncertainty counts hedges ("may", "possibly", ...).
* Every claim receives one retrieval step first (if the budget allows), so no
  claim is skipped silently.
* After that, the scheduler repeatedly picks the claim maximizing
      dynamic_priority / (1 + a * attempts) / (1 + b * low_gain_streak)
  and checks its next ``step_size`` passages.
* Evidence gain = increase in decisiveness, the best NLI entailment or
  contradiction probability among relevant passages. Steps with gain below
  ``low_gain_threshold`` extend a low-gain streak, which lowers priority, so the
  budget moves away from claims whose extra retrieval is not paying off.
* A claim is resolved (and stops consuming budget) as soon as it is SUPPORTED
  or CONTRADICTED under the verifier's rules.
* dynamic_priority = base_priority * (1 - decisiveness).

Claims the budget never reached are reported as UNCERTAIN ("not verified"),
never as supported.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

from .claims import decompose_causal
from .config import BudgetConfig
from .schema import ClaimStatus, ClaimVerification, EvidenceJudgement, ScoredEvidence
from .verification import ClaimVerifier

_HEDGES = ("possibly", "possible", "may", "might", "could", "likely", "perhaps", "suggests",
           "suggest", "appears", "appear", "reportedly", "it is believed", "probably", "approximately")


def linguistic_uncertainty(claim: str) -> float:
    """Heuristic uncertainty from hedging cues (not a calibrated probability)."""
    low = f" {claim.lower()} "
    hedges = sum(1 for h in _HEDGES if re.search(rf"(?<![a-z]){re.escape(h)}(?![a-z])", low))
    return min(1.0, 0.35 + 0.15 * hedges + (0.15 if len(claim) > 220 else 0.0))


@dataclass
class _ClaimState:
    index: int
    claim: str
    premises: list
    base_priority: float
    dynamic_priority: float
    judgements: list[EvidenceJudgement] = field(default_factory=list)
    cursor: int = 0
    attempts: int = 0
    decisiveness: float = 0.0
    total_gain: float = 0.0
    low_gain_streak: int = 0
    resolved: bool = False


@dataclass
class BudgetReport:
    strategy: str
    budget: int
    used: int
    extra_checks: int          # budget spent on causal decomposition after the scheduling loop
    exhausted: bool
    used_total: int = 0        # used + extra_checks (never exceeds budget)
    steps: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class BudgetedVerifier:
    def __init__(self, verifier: ClaimVerifier, config: BudgetConfig | None = None):
        self.verifier = verifier
        self.config = config or BudgetConfig()

    def _decisiveness(self, judgements: list[EvidenceJudgement]) -> float:
        threshold = self.verifier.config.relevance_threshold
        return max((max(j.entailment, j.contradiction) for j in judgements if j.relevance >= threshold),
                   default=0.0)

    def _pick(self, active: list[_ClaimState]) -> _ClaimState:
        cfg = self.config
        if cfg.strategy == "round_robin":
            return min(active, key=lambda s: (s.attempts, s.index))
        unseen = [s for s in active if s.attempts == 0]
        if unseen:   # first pass: every claim gets one step, highest priority first
            return max(unseen, key=lambda s: (s.base_priority, -s.index))
        return max(active, key=lambda s: (
            s.dynamic_priority / (1 + cfg.attempt_penalty * s.attempts)
            / (1 + cfg.low_gain_penalty * s.low_gain_streak), -s.index))

    def verify_claims(self, claims: list[str], pool: list[ScoredEvidence] | None = None,
                      budget: int | None = None) -> tuple[list[ClaimVerification], BudgetReport]:
        cfg = self.config
        if budget is None:
            budget = max(cfg.min_budget, math.ceil(cfg.checks_per_claim * len(claims))) if claims else 0
        states = []
        for i, claim in enumerate(claims):
            premises = self.verifier.candidate_premises(claim, pool)[: cfg.max_checks_per_claim]
            gap = 1.0 - max((p[2] for p in premises), default=0.0)
            priority = cfg.gap_weight * gap + (1 - cfg.gap_weight) * linguistic_uncertainty(claim)
            states.append(_ClaimState(i, claim, premises, round(priority, 4), priority))

        # Reserve part of the budget for checking the components of causal claims
        # ("A because B") after the main loop; at most a third of the budget.
        decomposable = [decompose_causal(c) for c in claims]
        reserve = min(budget // 3, sum(cfg.step_size * (1 + (d.cause is not None)) for d in decomposable if d))
        loop_budget = budget - reserve

        report = BudgetReport(strategy=cfg.strategy, budget=budget, used=0, extra_checks=0, exhausted=False)
        while report.used < loop_budget:
            active = [s for s in states if not s.resolved and s.cursor < len(s.premises)]
            if not active:
                break
            state = self._pick(active)
            take = min(cfg.step_size, loop_budget - report.used, len(state.premises) - state.cursor)
            batch = state.premises[state.cursor: state.cursor + take]
            state.judgements += self.verifier.judge(state.claim, batch)
            state.cursor += take
            state.attempts += 1
            report.used += take

            new_decisiveness = self._decisiveness(state.judgements)
            gain = max(0.0, new_decisiveness - state.decisiveness)
            state.decisiveness = new_decisiveness
            state.total_gain += gain
            state.low_gain_streak = state.low_gain_streak + 1 if gain < cfg.low_gain_threshold else 0
            state.dynamic_priority = state.base_priority * (1.0 - new_decisiveness)
            interim = self.verifier.decide(state.claim, state.judgements, pool, decompose=False).status
            state.resolved = interim in (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED)
            report.steps.append({"claim": state.index, "attempt": state.attempts, "checks": take,
                                 "gain": round(gain, 4), "decisiveness": round(new_decisiveness, 4),
                                 "interim_status": interim.value, "resolved": state.resolved})

        report.exhausted = any(not s.resolved and s.cursor < len(s.premises) for s in states)
        results = []
        for state in states:
            if state.attempts == 0 and state.premises:
                result = ClaimVerification(
                    state.claim, ClaimStatus.UNCERTAIN,
                    "Not verified: the evidence budget ran out before this claim was checked.")
            else:
                # Causal decomposition checks each component; it draws on the remaining
                # budget (a few passages per component) and is skipped when none is left.
                parts = decomposable[state.index] if not state.resolved else None
                n_parts = (1 + (parts.cause is not None)) if parts else 1
                remaining = budget - report.used - report.extra_checks
                per_part = min(2 * cfg.step_size, max(0, remaining) // n_parts)
                result = self.verifier.decide(state.claim, state.judgements, pool, decompose=True,
                                              part_premises=per_part)
                report.extra_checks += result.checks_used
                if (not state.resolved and state.cursor < len(state.premises)
                        and result.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.UNCERTAIN)):
                    result.explanation += (f" (Checked {state.cursor} of {len(state.premises)} candidate "
                                           "passages before the budget ran out.)")
            result.checks_used = state.cursor + sum(p.checks_used for p in result.parts)
            result.priority = state.base_priority
            results.append(result)
            report.used_total = report.used + report.extra_checks
            report.claims.append({"claim": state.claim, "base_priority": state.base_priority,
                                  "attempts": state.attempts, "checks": state.cursor,
                                  "candidates": len(state.premises), "total_gain": round(state.total_gain, 4),
                                  "status": result.status.value})
        return results, report
