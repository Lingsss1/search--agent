from __future__ import annotations

import json
import os

import pytest

from openstatesearch.eval.grpo_rollout_trend import summarize_phase_a_rollouts


def _prompt(question: str, start: bool) -> str:
    value = {"state": {"question": question}, "last_tool_result": None if start else {}}
    return "<|im_start|>user\n" + json.dumps(value) + "<|im_end|>\n<|im_start|>assistant\n"


def _write_version(root, version: int, rewards: list[float]) -> None:
    folder = root / str(version)
    folder.mkdir(parents=True)
    values = []
    for rollout_index, reward in enumerate(rewards):
        values.append(
            {
                "task_id": version,
                "sample_idx": rollout_index,
                "head_version": version,
                "tail_version": version,
                "prompt": _prompt(f"q{version}", True),
                "completion": "{}",
                "original_reward": reward,
            }
        )
    (folder / f"{version}.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))


def test_summarize_phase_a_rollouts_is_strict_and_windowed(tmp_path) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 1, [-1.05, -1.10, 0.5, -1.0])
    _write_version(root, 2, [0.0, -1.5, 0.2, -1.05])
    output = tmp_path / "trend.json"
    result = summarize_phase_a_rollouts(
        rollout_root=root,
        through_version=2,
        output_path=output,
        expected_episodes_per_version=4,
        window_size=1,
    )
    assert [value["valid_count"] for value in result["versions"]] == [1, 2]
    assert result["windows"][0]["valid_rate"] == 0.25
    assert result["windows"][1]["valid_rate"] == 0.5
    manifest = json.loads(output.with_suffix(".json.manifest.json").read_text())
    assert manifest["episodes"] == 8


def test_summarize_phase_a_rollouts_rejects_incomplete_version(tmp_path) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 1, [-1.0] * 4)
    with pytest.raises(ValueError, match="expected 5"):
        summarize_phase_a_rollouts(
            rollout_root=root,
            through_version=1,
            output_path=tmp_path / "trend.json",
            expected_episodes_per_version=5,
        )


def test_summarize_phase_a_rollouts_can_include_fresh_version_zero(tmp_path) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 0, [0.1, 0.2, 0.3, 0.4])
    _write_version(root, 1, [0.5, 0.6, 0.7, 0.8])
    output = tmp_path / "trend.json"
    result = summarize_phase_a_rollouts(
        rollout_root=root,
        start_version=0,
        through_version=1,
        output_path=output,
        expected_episodes_per_version=4,
        window_size=1,
    )
    assert result["start_version"] == 0
    assert [value["version"] for value in result["versions"]] == [0, 1]
    manifest = json.loads(output.with_suffix(".json.manifest.json").read_text())
    assert manifest["start_version"] == 0
    assert manifest["episodes"] == 8


def test_summarize_phase_a_rollouts_selects_latest_complete_retry(tmp_path) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 1, [-1.0] * 4)
    original = root / "1" / "1.jsonl"
    retry = root / "1" / "retry.jsonl"
    values = []
    for rollout_index in range(4):
        values.append(
            {
                "task_id": 101,
                "sample_idx": rollout_index,
                "head_version": 1,
                "tail_version": 1,
                "prompt": _prompt("q1", True),
                "completion": "{}",
                "original_reward": 1.0,
            }
        )
    retry.write_text("".join(json.dumps(value) + "\n" for value in values))
    os.utime(original, ns=(1_000_000_000, 1_000_000_000))
    os.utime(retry, ns=(2_000_000_000, 2_000_000_000))

    result = summarize_phase_a_rollouts(
        rollout_root=root,
        through_version=1,
        output_path=tmp_path / "trend.json",
        expected_episodes_per_version=4,
    )
    version = result["versions"][0]
    assert version["episodes"] == 4
    assert version["observed_episodes"] == 8
    assert version["discarded_retry_episodes"] == 4
    assert version["mean_reward"] == 1.0


def test_summarize_phase_a_rollouts_selects_latest_appended_batch(tmp_path) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 1, [-1.0] * 4)
    path = root / "1" / "1.jsonl"
    appended = []
    for rollout_index in range(4):
        appended.append(
            {
                "task_id": 1,
                "sample_idx": rollout_index,
                "head_version": 1,
                "tail_version": 1,
                "prompt": _prompt("q1", True),
                "completion": "{}",
                "original_reward": 1.0,
            }
        )
    with path.open("a") as handle:
        handle.write("".join(json.dumps(value) + "\n" for value in appended))

    result = summarize_phase_a_rollouts(
        rollout_root=root,
        through_version=1,
        output_path=tmp_path / "trend.json",
        expected_episodes_per_version=4,
    )

    version = result["versions"][0]
    assert version["episodes"] == 4
    assert version["observed_episodes"] == 8
    assert version["discarded_retry_episodes"] == 4
    assert version["source_files"] == 1
    assert version["observed_source_files"] == 2
    assert version["mean_reward"] == 1.0


def test_summarize_rejects_exact_episode_count_from_duplicate_task_batches(
    tmp_path,
) -> None:
    root = tmp_path / "rollout"
    _write_version(root, 1, [-1.0] * 4)
    path = root / "1" / "1.jsonl"
    duplicate_task_batch = []
    for rollout_index in range(4):
        duplicate_task_batch.append(
            {
                "task_id": 1,
                "sample_idx": rollout_index,
                "head_version": 1,
                "tail_version": 1,
                "prompt": _prompt("q1", True),
                "completion": "{}",
                "original_reward": 1.0,
            }
        )
    with path.open("a") as handle:
        handle.write("".join(json.dumps(value) + "\n" for value in duplicate_task_batch))

    with pytest.raises(ValueError, match="only 1 latest complete task sources; need 2"):
        summarize_phase_a_rollouts(
            rollout_root=root,
            through_version=1,
            output_path=tmp_path / "trend.json",
            expected_episodes_per_version=8,
        )
