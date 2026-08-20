from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from openstatesearch.agent.harness import SearchHarness
from openstatesearch.agent.schemas import parse_action
from openstatesearch.agent.state import SearchState
from openstatesearch.retriever import SearchHit


@dataclass(frozen=True)
class FrozenReplayDocument:
    doc_id: str
    title: str
    sentences: tuple[str, ...]


class FrozenTraceRetriever:
    """Replay the exact SEARCH/OPEN environment captured in a trajectory."""

    def __init__(self, trajectory: dict[str, Any]) -> None:
        searches: dict[str, list[list[SearchHit]]] = defaultdict(list)
        documents: dict[str, FrozenReplayDocument] = {}
        for event in trajectory.get("events", []):
            action = event.get("action")
            result = event.get("result")
            if not isinstance(action, dict) or not isinstance(result, dict):
                continue
            action_type = str(action.get("type", "")).upper()
            if (
                action_type in {"SEARCH", "VERIFY"}
                and result.get("ok")
                and not result.get("duplicate")
            ):
                hits = result.get("payload", {}).get("hits", [])
                searches[str(action.get("query", ""))].append(
                    [
                        SearchHit(
                            str(hit["doc_id"]),
                            str(hit["title"]),
                            float(hit["score"]),
                            str(hit["snippet"]),
                            str(hit.get("source", "unknown")),
                        )
                        for hit in hits
                    ]
                )
            if action_type == "OPEN" and result.get("ok"):
                payload = result.get("payload", {})
                sentence_rows = payload.get("sentences", [])
                sent_ids = [int(row["sent_id"]) for row in sentence_rows]
                if sent_ids != list(range(len(sentence_rows))):
                    raise ValueError("frozen OPEN sentences require contiguous sent_id values")
                document = FrozenReplayDocument(
                    doc_id=str(payload["doc_id"]),
                    title=str(payload["title"]),
                    sentences=tuple(str(row["text"]) for row in sentence_rows),
                )
                existing = documents.get(document.doc_id)
                if existing is not None and existing != document:
                    raise ValueError(f"conflicting frozen OPEN payload for {document.doc_id}")
                documents[document.doc_id] = document
        self._searches = dict(searches)
        self._search_offsets: dict[str, int] = defaultdict(int)
        self._documents = documents

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        offset = self._search_offsets[query]
        values = self._searches.get(query, [])
        if offset >= len(values):
            raise ValueError(f"no frozen search result remains for query {query!r}")
        self._search_offsets[query] += 1
        return values[offset][:k]

    def get_document(self, doc_id: str) -> FrozenReplayDocument | None:
        return self._documents.get(doc_id)


def trajectory_from_prediction(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a formal evaluation row into the canonical replay format."""
    events = []
    for turn, item in enumerate(record.get("actions", [])):
        action = item.get("action")
        result = item.get("tool_result")
        if not isinstance(action, dict) or not isinstance(result, dict):
            raise ValueError(f"formal prediction turn {turn} is not replayable")
        events.append({"action": action, "result": result})
    if not events or not record.get("trajectory_completed"):
        raise ValueError("formal replay requires a completed non-empty trajectory")
    action_types = {str(event["action"].get("type", "")).upper() for event in events}
    if not {"SEARCH", "OPEN", "KEEP", "ANSWER"}.issubset(action_types):
        raise ValueError("formal replay requires SEARCH, OPEN, KEEP, and ANSWER")
    generation_budget = int(record.get("generation_token_budget", 8192))
    generated_tokens = int(record.get("generated_tokens", 0))
    return {
        "question": str(record["question"]),
        "events": events,
        "final_state": {
            "budget": {
                "search_left": max(0, 4 - int(record.get("search_count", 0))),
                "open_left": max(0, 4 - int(record.get("open_count", 0))),
                "token_left": max(0, generation_budget - generated_tokens),
            }
        },
        "finished": True,
        "source_identity": {
            "dataset": str(record.get("dataset", "")),
            "id": str(record.get("id", "")),
        },
        "model_provenance_sha256": str(record.get("model_provenance_sha256", "")),
        "retriever_provenance_sha256": str(record.get("retriever_provenance_sha256", "")),
        "run_config_sha256": str(record.get("run_config_sha256", "")),
    }


def replay_frozen_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    return replay_trajectory(trajectory, FrozenTraceRetriever(trajectory))


def replay_trajectory(trajectory: dict[str, Any], retriever: Any) -> dict[str, Any]:
    """Re-execute a saved trace and report deterministic behavioral divergence."""
    final_state = trajectory.get("final_state", {})
    initial_searches = final_state.get("budget", {}).get("search_left", 0) + sum(
        1
        for event in trajectory.get("events", [])
        if isinstance(event.get("action"), dict)
        and event["action"].get("type") in {"SEARCH", "VERIFY"}
        and event.get("result", {}).get("ok")
        and not event.get("result", {}).get("duplicate")
    )
    initial_opens = final_state.get("budget", {}).get("open_left", 0) + sum(
        1
        for event in trajectory.get("events", [])
        if isinstance(event.get("action"), dict)
        and event["action"].get("type") == "OPEN"
        and event.get("result", {}).get("ok")
        and not event.get("result", {}).get("duplicate")
    )
    state = SearchState(str(trajectory["question"]))
    state.budget.search_left = initial_searches
    state.budget.open_left = initial_opens
    harness = SearchHarness(state, retriever)
    divergences: list[dict[str, Any]] = []
    for turn, saved in enumerate(trajectory.get("events", [])):
        if not isinstance(saved.get("action"), dict):
            divergences.append(
                {"turn": turn, "field": "action", "reason": "unparseable saved action"}
            )
            continue
        # In-memory dataclass traces contain tuples; persisted JSON traces contain lists.
        json_action = json.loads(json.dumps(saved["action"], ensure_ascii=False))
        actual = harness.apply(parse_action(json_action)).to_dict()
        expected = saved.get("result", {})
        for field in ("ok", "action", "error", "duplicate", "payload"):
            expected_value = json.loads(json.dumps(expected.get(field), ensure_ascii=False))
            actual_value = json.loads(json.dumps(actual.get(field), ensure_ascii=False))
            if actual_value != expected_value:
                divergences.append(
                    {
                        "turn": turn,
                        "field": field,
                        "expected": expected_value,
                        "actual": actual_value,
                    }
                )
        expected_hits = [hit["doc_id"] for hit in expected.get("payload", {}).get("hits", [])]
        actual_hits = [hit["doc_id"] for hit in actual.get("payload", {}).get("hits", [])]
        if expected_hits != actual_hits:
            divergences.append(
                {
                    "turn": turn,
                    "field": "hit_doc_ids",
                    "expected": expected_hits,
                    "actual": actual_hits,
                }
            )
    return {
        "replayed_events": len(harness.events),
        "finished": harness.finished,
        "expected_finished": bool(trajectory.get("finished")),
        "divergences": divergences,
        "exact": not divergences and harness.finished == bool(trajectory.get("finished")),
    }
