import json

from scripts.summarize_periodic_grpo_evals import build_index


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_periodic_index_includes_quality_and_joint_protocol_metrics(tmp_path):
    step = tmp_path / "step_000010"
    step.mkdir()
    predictions = step / "predictions.jsonl"
    rows = [
        {
            "id": "valid",
            "trajectory_completed": True,
            "actions": [{"tool_result": {"ok": True, "error": None}}],
        },
        {
            "id": "bad-citation",
            "trajectory_completed": True,
            "actions": [
                {
                    "tool_result": {
                        "ok": False,
                        "error": "invalid citation: sentence was not kept",
                    }
                }
            ],
        },
        {
            "id": "unfinished",
            "trajectory_completed": False,
            "actions": [
                {"tool_result": {"ok": False, "error": "invalid action JSON"}},
                {"tool_result": {"ok": False, "error": "invalid action JSON"}},
            ],
        },
    ]
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _write_json(step / "predictions.jsonl.metrics.json", {"trajectories": 3})
    _write_json(
        step / "predictions.jsonl.eval_metrics.json",
        {"overall": {"answer_f1": 0.5, "support_recall": 0.25}},
    )
    _write_json(step / "predictions.jsonl.manifest.json", {"schema_version": 1})
    _write_json(
        step / "paired_vs_step_000006.json",
        {"answer_f1": {"mean_delta": -0.01, "paired_bootstrap_95_ci": [-0.1, 0.1]}},
    )

    index = build_index(tmp_path)

    assert index["schema_version"] == 2
    assert index["completed_steps"] == [10]
    row = index["rows"][0]
    assert row["evaluation"]["answer_f1"] == 0.5
    assert row["paired_comparison"]["answer_f1"]["mean_delta"] == -0.01
    assert "paired_vs_step_000006.json" in row["artifacts"]
    assert row["protocol"] == {
        "trajectories": 3,
        "unique_ids": 3,
        "completed": 2,
        "completion_rate": 2 / 3,
        "joint_valid": 1,
        "joint_valid_rate": 1 / 3,
        "row_failure_categories_nonexclusive": {
            "invalid_action_json": 1,
            "invalid_citation": 1,
            "no_final_answer": 1,
        },
        "action_error_counts": {"invalid_action_json": 2, "invalid_citation": 1},
    }
