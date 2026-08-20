from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openstatesearch.agent.schemas import (
    Action,
    KeepAction,
    OpenAction,
    SearchAction,
    VerifyAction,
)

from .metrics import EvidenceRef


@dataclass(frozen=True)
class ABCCreditConfig:
    """Deterministic evidence-stage credit used by Phase-A GRPO.

    The maximum positive process return is deliberately one quarter of the
    magnitude of an invalid terminal reward.  This keeps ANSWER quality as the
    dominant objective while making useful prefixes distinguishable inside an
    otherwise all-invalid GRPO group.
    """

    search_stage: float = 0.1
    open_stage: float = 0.3
    keep_stage: float = 1.0
    alpha: float = 0.25
    invalid_action_penalty: float = -0.05
    process_positive_cap: float = 0.25
    process_negative_cap: float = 0.10
    beta: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.search_stage <= self.open_stage <= self.keep_stage:
            raise ValueError("ABC stage scores must be non-decreasing and non-negative")
        if self.keep_stage <= 0.0:
            raise ValueError("ABC keep_stage must be positive")
        if (
            self.alpha < 0.0
            or self.process_positive_cap <= 0.0
            or self.process_negative_cap <= 0.0
            or self.beta < 0.0
        ):
            raise ValueError("ABC alpha/beta must be non-negative and caps must be positive")
        if self.invalid_action_penalty > 0.0:
            raise ValueError("ABC invalid_action_penalty must not be positive")


@dataclass(frozen=True)
class CreditTransition:
    action: str
    valid: bool
    phi_before: float
    phi_after: float
    raw_reward: float
    process_reward: float
    newly_recalled: tuple[EvidenceRef, ...] = ()
    newly_opened: tuple[EvidenceRef, ...] = ()
    newly_kept: tuple[EvidenceRef, ...] = ()


def normalize_episode_rewards(rewards: Iterable[float], eps: float = 1e-8) -> list[float]:
    """Population-normalize terminal rewards for one prompt's rollout group."""

    values = [float(value) for value in rewards]
    if not values:
        raise ValueError("at least one terminal reward is required")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    denominator = variance**0.5 + eps
    return [(value - mean) / denominator for value in values]


def combine_group_advantages(
    episode_rewards: Iterable[float],
    process_rewards: Iterable[Iterable[float]],
    *,
    beta: float = 1.0,
) -> list[list[float]]:
    """Combine group-normalized terminal credit with local process credit."""

    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    normalized = normalize_episode_rewards(episode_rewards)
    local = [[float(value) for value in row] for row in process_rewards]
    if len(local) != len(normalized):
        raise ValueError("one process-reward sequence is required per terminal reward")
    return [
        [episode_advantage + beta * reward for reward in rewards]
        for episode_advantage, rewards in zip(normalized, local)
    ]


class EvidenceCreditTracker:
    """Track first progress through SEARCH -> OPEN -> KEEP for gold evidence.

    SEARCH results expose document IDs but not canonical sentence IDs, so a
    returned gold document advances every gold sentence in that document to
    the SEARCH stage. OPEN and KEEP use exact stable ``(doc_id, sent_id)``
    references. Repeated stages cannot increase the potential.
    """

    def __init__(
        self,
        gold_evidence: Iterable[EvidenceRef],
        config: ABCCreditConfig | None = None,
    ) -> None:
        self.config = config or ABCCreditConfig()
        self.gold = tuple(
            sorted({(str(doc_id), int(sent_id)) for doc_id, sent_id in gold_evidence})
        )
        self._stage = {reference: 0.0 for reference in self.gold}
        self._positive_process_return = 0.0
        self._negative_process_return = 0.0

    @property
    def phi(self) -> float:
        if not self._stage:
            return 0.0
        return sum(self._stage.values()) / (len(self._stage) * self.config.keep_stage)

    @property
    def stage_by_evidence(self) -> dict[EvidenceRef, float]:
        return dict(self._stage)

    def _advance(self, references: Iterable[EvidenceRef], stage: float) -> tuple[EvidenceRef, ...]:
        advanced: list[EvidenceRef] = []
        for reference in references:
            if reference in self._stage and self._stage[reference] < stage:
                self._stage[reference] = stage
                advanced.append(reference)
        return tuple(sorted(advanced))

    def _cap(self, reward: float) -> float:
        if reward >= 0.0:
            remaining = max(
                0.0,
                self.config.process_positive_cap - self._positive_process_return,
            )
            bounded = min(remaining, reward)
            self._positive_process_return += bounded
            return bounded

        remaining = max(
            0.0,
            self.config.process_negative_cap - self._negative_process_return,
        )
        bounded = max(-remaining, reward)
        self._negative_process_return += abs(bounded)
        return bounded

    def score(self, action: Action | None, result: dict[str, object]) -> CreditTransition:
        """Score one observed environment transition without semantic judging."""

        before = self.phi
        valid = bool(result.get("ok")) and action is not None
        action_name = str(result.get("action") or "INVALID")
        recalled: tuple[EvidenceRef, ...] = ()
        opened: tuple[EvidenceRef, ...] = ()
        kept: tuple[EvidenceRef, ...] = ()

        if valid and isinstance(action, (SearchAction, VerifyAction)):
            payload = result.get("payload")
            hits = payload.get("hits", []) if isinstance(payload, dict) else []
            returned_docs = {
                str(hit["doc_id"]) for hit in hits if isinstance(hit, dict) and "doc_id" in hit
            }
            recalled = self._advance(
                (reference for reference in self.gold if reference[0] in returned_docs),
                self.config.search_stage,
            )
        elif valid and isinstance(action, OpenAction):
            payload = result.get("payload")
            sentences = payload.get("sentences", []) if isinstance(payload, dict) else []
            shown = {
                (action.doc_id, int(sentence["sent_id"]))
                for sentence in sentences
                if isinstance(sentence, dict) and "sent_id" in sentence
            }
            opened = self._advance(shown, self.config.open_stage)
        elif valid and isinstance(action, KeepAction):
            kept = self._advance(
                ((action.doc_id, sent_id) for sent_id in action.sent_ids),
                self.config.keep_stage,
            )

        after = self.phi
        raw_reward = (
            self.config.alpha * (after - before) if valid else self.config.invalid_action_penalty
        )
        return CreditTransition(
            action=action_name,
            valid=valid,
            phi_before=before,
            phi_after=after,
            raw_reward=raw_reward,
            process_reward=self._cap(raw_reward),
            newly_recalled=recalled,
            newly_opened=opened,
            newly_kept=kept,
        )
