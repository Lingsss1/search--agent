from __future__ import annotations

from dataclasses import asdict, dataclass

from .metrics import EvidenceRef, answer_f1, duplicate_rate, evidence_precision_recall_f1


@dataclass(frozen=True)
class TrajectoryOutcome:
    prediction: str
    references: tuple[str, ...]
    predicted_evidence: tuple[EvidenceRef, ...]
    gold_evidence: tuple[EvidenceRef, ...]
    citations: tuple[EvidenceRef, ...]
    queries: tuple[str, ...] = ()
    returned_doc_ids: tuple[tuple[str, ...], ...] = ()
    search_count: int = 0
    open_count: int = 0
    generated_tokens: int = 0
    invalid_action_count: int = 0
    valid_tools: bool = True
    valid_citations: bool = True


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    answer_f1: float
    support_recall: float
    citation_precision: float
    duplicate_rate: float
    search_cost: float
    open_cost: float
    token_cost: float
    protocol_penalty: float
    valid: bool
    phase: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return asdict(self)


def compute_reward(outcome: TrajectoryOutcome, phase: str = "A") -> RewardBreakdown:
    if phase not in {"A", "B"}:
        raise ValueError("phase must be A or B")
    protocol_penalty = min(0.5, 0.05 * max(0, outcome.invalid_action_count))
    if not outcome.valid_tools or not outcome.valid_citations:
        # Keep every terminally invalid trajectory on the same outcome reward.
        # Ranking them by the number of recoverable mistakes makes an immediate
        # EOS (-1.05) look better than a longer attempt (-1.10, -1.15, ...)
        # whenever an entire GRPO group fails.  The protocol count remains in
        # the breakdown for auditing, but only a trajectory that eventually
        # forms a legal answer pays the bounded penalty in the learning reward.
        return RewardBreakdown(
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            protocol_penalty,
            False,
            phase,
        )

    f1 = answer_f1(outcome.prediction, outcome.references)
    _, support_recall, _ = evidence_precision_recall_f1(
        outcome.predicted_evidence, outcome.gold_evidence
    )
    citation_precision, _, _ = evidence_precision_recall_f1(
        outcome.citations, outcome.gold_evidence
    )
    repeated = duplicate_rate(outcome.queries, outcome.returned_doc_ids)
    quality = 2 * f1 + 0.8 * support_recall + 0.4 * citation_precision - 0.15 * repeated

    search_cost = open_cost = token_cost = 0.0
    total = quality - protocol_penalty
    if phase == "B":
        if f1 < 0.8:
            total -= 0.5
        else:
            search_cost = 0.03 * outcome.search_count
            open_cost = 0.01 * outcome.open_count
            token_cost = 0.0001 * outcome.generated_tokens
            total -= search_cost + open_cost + token_cost
    return RewardBreakdown(
        total,
        f1,
        support_recall,
        citation_precision,
        repeated,
        search_cost,
        open_cost,
        token_cost,
        protocol_penalty,
        True,
        phase,
    )
