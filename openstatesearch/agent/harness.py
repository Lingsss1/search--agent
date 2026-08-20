from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from openstatesearch.retriever.types import Document, SearchHit

from .schemas import AnswerAction, KeepAction, OpenAction, SearchAction, VerifyAction
from .state import Candidate, Evidence, SearchState


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[SearchHit]: ...
    def get_document(self, doc_id: str) -> Document | None: ...


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().casefold())


@dataclass
class ToolResult:
    ok: bool
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchHarness:
    """Deterministic environment that owns state, budgets and citation validity."""

    def __init__(self, state: SearchState, retriever: Retriever, top_k: int = 5):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.state = state
        self.retriever = retriever
        self.top_k = top_k
        self.opened: dict[str, Document] = {}
        self.finished = False
        self.answer: AnswerAction | None = None
        self.events: list[dict[str, Any]] = []

    def consume_tokens(self, count: int) -> None:
        if count < 0:
            raise ValueError("token count must be non-negative")
        self.state.budget.token_left = max(0, self.state.budget.token_left - count)

    def apply(
        self, action: SearchAction | OpenAction | KeepAction | VerifyAction | AnswerAction
    ) -> ToolResult:
        if self.finished:
            result = ToolResult(False, action.type.value, error="trajectory already finished")
        elif isinstance(action, SearchAction):
            result = self._search(action.query, action.target_constraint, "SEARCH")
        elif isinstance(action, VerifyAction):
            result = self._search(action.query, action.claim, "VERIFY")
        elif isinstance(action, OpenAction):
            result = self._open(action)
        elif isinstance(action, KeepAction):
            result = self._keep(action)
        else:
            result = self._answer(action)
        self.events.append({"action": asdict(action), "result": result.to_dict()})
        return result

    def _search(self, query: str, target: str, action_name: str) -> ToolResult:
        normalized = normalize_query(query)
        if normalized in {normalize_query(item) for item in self.state.query_history}:
            return ToolResult(
                True,
                action_name,
                payload={"query": query, "hits": []},
                duplicate=True,
            )
        if self.state.budget.search_left <= 0:
            return ToolResult(False, action_name, error="search budget exhausted")

        hits = self.retriever.search(query, self.top_k)
        self.state.budget.search_left -= 1
        self.state.query_history.append(query)
        known = {candidate.doc_id for candidate in self.state.candidate_pool}
        for hit in hits:
            if hit.doc_id not in known:
                self.state.candidate_pool.append(
                    Candidate(hit.doc_id, hit.title, hit.score, hit.snippet)
                )
                known.add(hit.doc_id)
        return ToolResult(
            True,
            action_name,
            payload={"query": query, "target": target, "hits": [asdict(hit) for hit in hits]},
        )

    def _open(self, action: OpenAction) -> ToolResult:
        if self.state.budget.open_left <= 0:
            return ToolResult(False, "OPEN", error="open budget exhausted")
        candidate_ids = {item.doc_id for item in self.state.candidate_pool}
        if action.doc_id not in candidate_ids:
            return ToolResult(False, "OPEN", error="doc_id is not in candidate_pool")
        document = self.retriever.get_document(action.doc_id)
        if document is None:
            return ToolResult(False, "OPEN", error="document not found")
        already_open = action.doc_id in self.opened
        if not already_open:
            self.state.budget.open_left -= 1
            self.opened[action.doc_id] = document
        return ToolResult(
            True,
            "OPEN",
            payload={
                "doc_id": document.doc_id,
                "title": document.title,
                "sentences": [
                    {"sent_id": i, "text": text} for i, text in enumerate(document.sentences)
                ],
            },
            duplicate=already_open,
        )

    def _validate_sentence_refs(self, doc_id: str, sent_ids: tuple[int, ...]) -> str | None:
        document = self.opened.get(doc_id)
        if document is None:
            return "doc_id has not been opened"
        if any(index >= len(document.sentences) for index in sent_ids):
            return "sent_id is out of range"
        return None

    def _keep(self, action: KeepAction) -> ToolResult:
        error = self._validate_sentence_refs(action.doc_id, action.sent_ids)
        if error:
            return ToolResult(False, "KEEP", error=error)
        evidence = Evidence(action.doc_id, action.sent_ids, action.claim, action.constraint_id)
        if evidence not in self.state.evidence:
            self.state.evidence.append(evidence)
        return ToolResult(True, "KEEP", payload=asdict(evidence))

    def _answer(self, action: AnswerAction) -> ToolResult:
        kept_refs = {
            (evidence.doc_id, sent_id)
            for evidence in self.state.evidence
            for sent_id in evidence.sent_ids
        }
        for citation in action.citations:
            error = self._validate_sentence_refs(citation.doc_id, citation.sent_ids)
            if error:
                return ToolResult(False, "ANSWER", error=f"invalid citation: {error}")
            if any((citation.doc_id, sent_id) not in kept_refs for sent_id in citation.sent_ids):
                return ToolResult(False, "ANSWER", error="invalid citation: sentence was not kept")
        self.finished = True
        self.answer = action
        return ToolResult(
            True,
            "ANSWER",
            payload={"answer": action.answer, "citations": [asdict(c) for c in action.citations]},
        )

    def trajectory(self) -> dict[str, Any]:
        return {
            "question": self.state.question,
            "events": self.events,
            "final_state": self.state.observation(),
            "finished": self.finished,
        }
