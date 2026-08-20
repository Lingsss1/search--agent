from __future__ import annotations

import json

import pytest

from openstatesearch.eval.reward_audit import (
    normalize_reward_audit,
    select_reward_audit_records,
)


def _records(step: int, count: int) -> list[dict[str, object]]:
    return [
        {"step": step, "trajectory_id": f"trajectory-{index}", "total": index / 10}
        for index in range(count)
    ]


def test_reward_audit_selection_is_exact_and_order_independent() -> None:
    records = _records(100, 64) + _records(200, 49)
    selected, counts = select_reward_audit_records(records, sample_size=50, seed=36)
    reversed_selected, _ = select_reward_audit_records(reversed(records), sample_size=50, seed=36)
    assert counts == {100: 64, 200: 49}
    assert sum(record["step"] == 100 for record in selected) == 50
    assert sum(record["step"] == 200 for record in selected) == 49
    assert selected == reversed_selected


def test_reward_audit_normalization_preserves_source_and_emits_summary(tmp_path) -> None:
    source = tmp_path / "raw.jsonl"
    destination = tmp_path / "normalized.jsonl"
    summary_path = tmp_path / "summary.json"
    raw = _records(100, 64)
    source.write_text("".join(json.dumps(record) + "\n" for record in raw))

    summary = normalize_reward_audit(source, destination, summary_path)

    assert len(source.read_text().splitlines()) == 64
    assert len(destination.read_text().splitlines()) == 50
    assert summary["steps"]["100"] == {"raw_count": 64, "selected_count": 50}
    assert json.loads(summary_path.read_text()) == summary


def test_reward_audit_normalization_rejects_in_place_output(tmp_path) -> None:
    source = tmp_path / "audit.jsonl"
    source.write_text(json.dumps({"step": 100}) + "\n")
    with pytest.raises(ValueError, match="must differ"):
        normalize_reward_audit(source, source, tmp_path / "summary.json")
