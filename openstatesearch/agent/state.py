from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Budget:
    search_left: int = 4
    open_left: int = 4
    token_left: int = 8192

    def __post_init__(self) -> None:
        if min(self.search_left, self.open_left, self.token_left) < 0:
            raise ValueError("budget values must be non-negative")


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    title: str
    score: float
    snippet: str


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    sent_ids: tuple[int, ...]
    claim: str
    constraint_id: str


@dataclass
class SearchState:
    question: str
    constraints: list[str] = field(default_factory=list)
    candidate_pool: list[Candidate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)

    def observation(self) -> dict[str, Any]:
        """Return the complete policy-visible state. Gold data has no field here."""
        return asdict(self)
