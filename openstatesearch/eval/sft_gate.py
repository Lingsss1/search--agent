from __future__ import annotations

from typing import Any

from openstatesearch.agent.schemas import (
    ActionValidationError,
    AnswerAction,
    KeepAction,
    OpenAction,
    SearchAction,
    parse_action,
)


def sft_gate_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("gate records cannot be empty")
    valid_json = nonempty_search = search_total = action_total = 0
    invalid_doc_refs = doc_ref_total = completed = 0
    for record in records:
        steps = record.get("actions")
        if not isinstance(steps, list):
            steps = [record]
        completed += bool(record.get("trajectory_completed", False))
        for step in steps:
            if not isinstance(step, dict):
                action_total += 1
                continue
            action_total += 1
            raw = step.get("action")
            try:
                action = parse_action(raw)
                valid_json += 1
            except (ActionValidationError, TypeError):
                continue
            if isinstance(action, SearchAction):
                search_total += 1
                nonempty_search += bool(action.query.strip())
            if isinstance(action, OpenAction):
                doc_ref_total += 1
                invalid_doc_refs += action.doc_id not in set(step.get("visible_doc_ids", []))
            if isinstance(action, KeepAction):
                doc_ref_total += 1
                invalid_doc_refs += action.doc_id not in set(step.get("opened_doc_ids", []))
            if isinstance(action, AnswerAction):
                opened = set(step.get("opened_doc_ids", []))
                doc_ref_total += len(action.citations)
                invalid_doc_refs += sum(
                    citation.doc_id not in opened for citation in action.citations
                )
    if not action_total:
        raise ValueError("gate records contain no actions")
    return {
        "valid_tool_json": valid_json / action_total,
        "invalid_doc_reference": invalid_doc_refs / doc_ref_total if doc_ref_total else 0.0,
        "nonempty_search_query": nonempty_search / search_total if search_total else 1.0,
        "completion_rate": completed / len(records),
        "trajectories": len(records),
        "actions": action_total,
    }


def passes_sft_gate(metrics: dict[str, float], gate: dict[str, float]) -> bool:
    return (
        metrics["valid_tool_json"] >= gate["valid_tool_json"]
        and metrics["invalid_doc_reference"] <= gate["invalid_doc_reference_max"]
        and metrics["nonempty_search_query"] >= gate["nonempty_search_query"]
        and metrics["completion_rate"] >= gate["completion_rate"]
    )
