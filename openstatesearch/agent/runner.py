from __future__ import annotations

from typing import Any, Protocol

from .harness import SearchHarness
from .schemas import ActionValidationError, parse_action


class Policy(Protocol):
    def next_action(
        self, observation: dict[str, Any], last_result: dict[str, Any] | None
    ) -> dict[str, Any]: ...


def run_agent(policy: Policy, harness: SearchHarness, max_turns: int = 16) -> dict[str, Any]:
    """Execute one trajectory while keeping the Policy isolated from gold labels."""
    last_result: dict[str, Any] | None = None
    for _ in range(max_turns):
        raw_action = policy.next_action(harness.state.observation(), last_result)
        try:
            action = parse_action(raw_action)
        except ActionValidationError as exc:
            last_result = {"ok": False, "error": str(exc), "action": "INVALID"}
            harness.events.append({"action": raw_action, "result": last_result})
            continue
        result = harness.apply(action)
        last_result = result.to_dict()
        if harness.finished:
            break
    return harness.trajectory()
