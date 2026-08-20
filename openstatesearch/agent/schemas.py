from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ActionType(str, Enum):
    SEARCH = "SEARCH"
    OPEN = "OPEN"
    KEEP = "KEEP"
    VERIFY = "VERIFY"
    ANSWER = "ANSWER"


class ActionValidationError(ValueError):
    pass


def _strict_keys(
    data: Mapping[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = required - set(data)
    extra = set(data) - required - optional
    if missing:
        raise ActionValidationError(f"missing fields: {sorted(missing)}")
    if extra:
        raise ActionValidationError(f"unexpected fields: {sorted(extra)}")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class SearchAction:
    query: str
    target_constraint: str
    type: ActionType = ActionType.SEARCH


@dataclass(frozen=True)
class OpenAction:
    doc_id: str
    type: ActionType = ActionType.OPEN


@dataclass(frozen=True)
class KeepAction:
    doc_id: str
    sent_ids: tuple[int, ...]
    claim: str
    constraint_id: str
    type: ActionType = ActionType.KEEP


@dataclass(frozen=True)
class VerifyAction:
    claim: str
    query: str
    type: ActionType = ActionType.VERIFY


@dataclass(frozen=True)
class Citation:
    claim: str
    doc_id: str
    sent_ids: tuple[int, ...]


@dataclass(frozen=True)
class AnswerAction:
    answer: str
    citations: tuple[Citation, ...]
    type: ActionType = ActionType.ANSWER


Action = SearchAction | OpenAction | KeepAction | VerifyAction | AnswerAction


def _sent_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ActionValidationError("sent_ids must be a non-empty list")
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value):
        raise ActionValidationError("sent_ids must contain non-negative integers")
    if len(set(value)) != len(value):
        raise ActionValidationError("sent_ids must not contain duplicates")
    return tuple(value)


def parse_action(data: Mapping[str, Any]) -> Action:
    """Parse a strict action object; unknown fields are rejected."""
    if not isinstance(data, Mapping):
        raise ActionValidationError("action must be an object")
    raw_type = data.get("type")
    try:
        action_type = ActionType(raw_type)
    except (ValueError, TypeError) as exc:
        raise ActionValidationError(f"unsupported action type: {raw_type!r}") from exc

    if action_type is ActionType.SEARCH:
        _strict_keys(data, {"type", "query", "target_constraint"})
        return SearchAction(
            _text(data["query"], "query"), _text(data["target_constraint"], "target_constraint")
        )
    if action_type is ActionType.OPEN:
        _strict_keys(data, {"type", "doc_id"})
        return OpenAction(_text(data["doc_id"], "doc_id"))
    if action_type is ActionType.KEEP:
        # ``constraint_id`` is bookkeeping metadata, not part of the evidence
        # safety boundary. Evidence legality is determined mechanically from
        # the opened ``doc_id`` and ``sent_ids`` in ``SearchHarness``. Treat a
        # missing/empty identifier as unlabelled evidence so a harmless label
        # omission cannot break an otherwise legal OPEN -> KEEP -> ANSWER chain.
        _strict_keys(
            data,
            {"type", "doc_id", "sent_ids", "claim"},
            {"constraint_id"},
        )
        raw_constraint_id = data.get("constraint_id")
        constraint_id = (
            raw_constraint_id.strip()
            if isinstance(raw_constraint_id, str) and raw_constraint_id.strip()
            else "unlabelled"
        )
        return KeepAction(
            _text(data["doc_id"], "doc_id"),
            _sent_ids(data["sent_ids"]),
            _text(data["claim"], "claim"),
            constraint_id,
        )
    if action_type is ActionType.VERIFY:
        _strict_keys(data, {"type", "claim", "query"})
        return VerifyAction(_text(data["claim"], "claim"), _text(data["query"], "query"))

    _strict_keys(data, {"type", "answer", "citations"})
    if not isinstance(data["citations"], list) or not data["citations"]:
        raise ActionValidationError("citations must be a non-empty list")
    citations: list[Citation] = []
    for index, item in enumerate(data["citations"]):
        if not isinstance(item, Mapping):
            raise ActionValidationError(f"citations[{index}] must be an object")
        _strict_keys(item, {"claim", "doc_id", "sent_ids"})
        citations.append(
            Citation(
                _text(item["claim"], "claim"),
                _text(item["doc_id"], "doc_id"),
                _sent_ids(item["sent_ids"]),
            )
        )
    return AnswerAction(_text(data["answer"], "answer"), tuple(citations))
