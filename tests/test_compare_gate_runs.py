import json

import pytest

from scripts.compare_gate_runs import compare


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _record(identity, prediction, *, completed=True):
    return {
        "dataset": "hotpotqa",
        "id": identity,
        "question": "q",
        "actions": [
            {
                "action": {"type": "SEARCH", "query": "q", "target_constraint": "c"},
                "tool_result": {"ok": True, "action": "SEARCH"},
            }
        ],
        "trajectory_completed": completed,
        "prediction": prediction,
        "answers": ["Paris"],
        "evidence": [{"doc_id": "d", "sent_ids": [0]}],
        "citations": [{"doc_id": "d", "sent_ids": [0]}],
        "gold_evidence": [{"doc_id": "d", "sent_ids": [0]}],
    }


def test_compare_gate_runs_reports_paired_length_and_completion_effects(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    target = tmp_path / "target.jsonl"
    _write(baseline, [_record("one", "Paris"), _record("two", "", completed=False)])
    _write(
        target,
        [
            _record("one", "The answer is Paris"),
            _record("two", "Paris"),
        ],
    )

    report = compare(baseline, target, bootstrap_samples=100)

    assert report["examples"] == 2
    assert report["prediction_tokens"] == {"baseline_mean": 0.5, "target_mean": 2.0}
    assert report["transitions"]["completion_0_to_1"] == 1
    assert report["attribution"]["verbosity_regressions_with_full_reference_recall"] == 1
    assert report["exact_match"] == {"actions": 2, "core_trajectory": 0}
    assert len(report["answer_f1"]["paired_bootstrap_95_ci"]) == 2


def test_compare_gate_runs_reports_exact_backend_parity(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    target = tmp_path / "target.jsonl"
    record = _record("one", "Paris")
    _write(baseline, [record])
    _write(target, [{**record, "experiment": "different-metadata"}])

    report = compare(baseline, target, bootstrap_samples=100)

    assert report["exact_match"] == {"actions": 1, "core_trajectory": 1}


def test_compare_gate_runs_rejects_unpaired_identities(tmp_path):
    baseline = tmp_path / "baseline.jsonl"
    target = tmp_path / "target.jsonl"
    _write(baseline, [_record("one", "Paris")])
    _write(target, [_record("two", "Paris")])

    with pytest.raises(ValueError, match="identities differ"):
        compare(baseline, target, bootstrap_samples=100)
