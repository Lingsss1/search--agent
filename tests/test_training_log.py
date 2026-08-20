from openstatesearch.eval.training_log import parse_training_log, summarize_training_rows


def test_training_log_parser_binds_metrics_to_completed_steps(tmp_path):
    path = tmp_path / "main.log"
    path.write_text(
        "\x1b[92m(AReaL) 20260814-01:00:00.000 Stats INFO: "
        "Step 1/3 Train step 1/3 done.\x1b[0m\n"
        "│ ppo_actor/update/grad_norm │ 1.0000e-01 │ rollout/valid_reward │ 0.0000e+00 │\n"
        "│ ppo_actor/update/behave_imp_weight_applied/avg │ 8.0000e-01 │ "
        "ppo_actor/update/behave_imp_weight_sq/avg │ 1.0000e+00 │\n"
        "(AReaL) 20260814-01:01:00.000 Stats INFO: Step 2/3 Train step 2/3 done.\n"
        "│ ppo_actor/update/grad_norm │ 2.0000e-01 │ rollout/valid_reward │ 3.1250e-02 │\n"
    )
    rows, metadata = parse_training_log(path)
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[1]["metrics"]["rollout/valid_reward"] == 0.03125
    assert (
        abs(rows[0]["metrics"]["derived/behavior_normalized_effective_sample_size"] - 0.64) < 1e-12
    )
    assert metadata["declared_total_steps"] == 3
    assert metadata["missing_steps_in_observed_range"] == []
    windows = summarize_training_rows(rows, window_size=2)
    reward = windows[0]["metrics"]["rollout/valid_reward"]
    assert reward["mean"] == 0.015625
    assert reward["nonzero"] == 1


def test_training_log_parser_reports_observed_gaps(tmp_path):
    path = tmp_path / "main.log"
    path.write_text(
        "(AReaL) 20260814-01:00:00.000 Stats INFO: Step 1/3 Train step done.\n"
        "(AReaL) 20260814-01:02:00.000 Stats INFO: Step 3/3 Train step done.\n"
    )
    _, metadata = parse_training_log(path)
    assert metadata["missing_steps_in_observed_range"] == [2]
