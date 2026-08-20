from __future__ import annotations

from typing import Any, Iterable


def behavior_metrics(events: Iterable[dict[str, Any]]) -> dict[str, float]:
    events = list(events)
    searches = [
        event for event in events if event.get("action", {}).get("type") in {"SEARCH", "VERIFY"}
    ]
    opens = [event for event in events if event.get("action", {}).get("type") == "OPEN"]
    invalid = [event for event in events if not event.get("result", {}).get("ok", False)]
    duplicate = [event for event in searches if event.get("result", {}).get("duplicate", False)]
    return {
        "search_count": float(len(searches)),
        "open_count": float(len(opens)),
        "invalid_action_count": float(len(invalid)),
        "duplicate_query_rate": len(duplicate) / len(searches) if searches else 0.0,
    }
