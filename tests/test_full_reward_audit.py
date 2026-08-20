from __future__ import annotations

import hashlib
import json

import pytest

from openstatesearch.eval.full_reward_audit import (
    attach_reward_breakdowns,
    build_full_reward_audit,
    load_rollout_trajectories,
)


def _prompt(question: str, last_tool_result: object) -> str:
    policy_input = {
        "state": {"question": question},
        "last_tool_result": last_tool_result,
    }
    return (
        "<|im_start|>system\ntest<|im_end|>\n<|im_start|>user\n"
        + json.dumps(policy_input)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _interaction(task_id: int, question: str, reward: float, start: bool) -> dict:
    return {
        "task_id": task_id,
        "sample_idx": 0,
        "head_version": 100,
        "tail_version": 100,
        "prompt": _prompt(question, None if start else {"ok": True}),
        "completion": "{}",
        "original_reward": reward,
    }


def test_full_reward_audit_links_and_selects_exact_sample(tmp_path) -> None:
    rollout_root = tmp_path / "rollout"
    version_root = rollout_root / "100"
    version_root.mkdir(parents=True)
    prompts = []
    raw = []
    for task_id in range(13):
        question = f"question {task_id}"
        prompt_id = f"id-{task_id}"
        prompts.append({"question": question, "dataset": "test", "id": prompt_id})
        interactions = []
        for rollout_index in range(4):
            reward = task_id + rollout_index / 10
            interactions.extend(
                [
                    _interaction(task_id, question, reward, True),
                    _interaction(task_id, question, reward, False),
                ]
            )
            raw.append(
                {
                    "step": 100,
                    "trajectory_id": prompt_id,
                    "total": reward,
                    "answer_f1": 1.0,
                    "support_recall": 1.0,
                    "citation_precision": 1.0,
                    "duplicate_rate": 0.0,
                    "search_cost": 0.0,
                    "open_cost": 0.0,
                    "token_cost": 0.0,
                    "protocol_penalty": 0.0,
                    "valid": 1.0,
                }
            )
        (version_root / f"{task_id}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in interactions)
        )
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text("".join(json.dumps(item) + "\n" for item in prompts))
    reward_path = tmp_path / "reward.jsonl"
    reward_path.write_text("".join(json.dumps(item) + "\n" for item in raw))
    output = tmp_path / "full.jsonl"
    summary_path = tmp_path / "summary.json"
    model_manifest = tmp_path / "merge_manifest.json"
    model_manifest.write_text('{"kind":"merged_model"}\n')
    checkpoint_manifest = tmp_path / "checkpoint_manifest.json"
    checkpoint_manifest.write_text('{"step":100}\n')

    summary = build_full_reward_audit(
        rollout_root=rollout_root,
        reward_audit_path=reward_path,
        prompts_path=prompt_path,
        output_path=output,
        summary_path=summary_path,
        version=100,
        model_provenance_path=model_manifest,
        checkpoint_manifest_path=checkpoint_manifest,
    )

    values = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(values) == 50
    assert all(len(value["interactions"]) == 2 for value in values)
    assert summary["matching"]["matched_records"] == 52
    assert summary["matching"]["ambiguous_reward_matches"] == 0
    assert summary["selected_metrics"]["trajectories"] == 50
    assert summary["selected_metrics"]["valid_count"] == 50
    assert summary["selected_metrics"]["valid_rate"] == 1.0
    assert summary["selected_metrics"]["interaction_count"] == {
        "mean": 2.0,
        "min": 2,
        "max": 2,
    }
    assert summary["selected_metrics"]["datasets"]["test"] == {
        "trajectories": 50,
        "valid_count": 50,
        "valid_rate": 1.0,
    }
    assert summary["selected_metrics"]["reward_components"]["total"]["min"] >= 0
    assert (
        summary["model_provenance_sha256"]
        == hashlib.sha256(model_manifest.read_bytes()).hexdigest()
    )
    assert summary["checkpoint_manifest"]["path"] == str(checkpoint_manifest)
    assert json.loads(summary_path.read_text()) == summary


def test_rollout_loader_rejects_missing_rollout_start(tmp_path) -> None:
    root = tmp_path / "rollout" / "100"
    root.mkdir(parents=True)
    value = _interaction(7, "question", 0.0, False)
    (root / "7.jsonl").write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="0 rollout starts"):
        load_rollout_trajectories(root.parent, 100)


def test_rollout_loader_splits_appended_complete_batches(tmp_path) -> None:
    root = tmp_path / "rollout" / "100"
    root.mkdir(parents=True)
    values = []
    for batch_index in range(2):
        for rollout_index in range(4):
            values.append(
                _interaction(
                    7,
                    "question",
                    float(batch_index * 10 + rollout_index),
                    True,
                )
            )
    path = root / "7.jsonl"
    path.write_text("".join(json.dumps(value) + "\n" for value in values))

    trajectories, sources = load_rollout_trajectories(root.parent, 100)

    assert len(trajectories) == 8
    assert len(sources) == 2
    assert [source["batch_index"] for source in sources] == [0, 1]
    assert len({item["trajectory_key"] for item in trajectories}) == 8
    assert [item["rollout_batch_index"] for item in trajectories] == [0] * 4 + [1] * 4
    assert all(item["rollout_source_key"] == sources[0]["source_key"] for item in trajectories[:4])
    assert all(item["rollout_source_key"] == sources[1]["source_key"] for item in trajectories[4:])


def test_rollout_loader_rejects_partial_appended_batch(tmp_path) -> None:
    root = tmp_path / "rollout" / "100"
    root.mkdir(parents=True)
    values = [_interaction(7, "question", float(index), True) for index in range(5)]
    (root / "7.jsonl").write_text("".join(json.dumps(value) + "\n" for value in values))
    with pytest.raises(ValueError, match="positive multiple of 4"):
        load_rollout_trajectories(root.parent, 100)


def test_reward_matching_uses_action_sequence_to_disambiguate_equal_rewards() -> None:
    action_names = ("SEARCH", "OPEN", "KEEP", "ANSWER")
    trajectories = []
    reward_records = []
    for rollout_index, action_name in enumerate(action_names):
        interaction = _interaction(7, "question", 1.0, True)
        interaction["completion"] = json.dumps({"type": action_name})
        trajectories.append(
            {
                "trajectory_key": f"100:7:0:{rollout_index}",
                "question": "question",
                "original_reward": 1.0,
                "interactions": [interaction],
            }
        )
        reward_records.append(
            {
                "trajectory_id": "prompt-id",
                "total": 1.0,
                "credit_transitions": [{"action": action_name}],
            }
        )

    matched, matching = attach_reward_breakdowns(
        trajectories,
        reward_records,
        [{"question": "question", "dataset": "test", "id": "prompt-id"}],
    )

    assert matching["matched_records"] == 4
    assert matching["ambiguous_reward_matches_before_action_signature"] == 3
    assert matching["action_signature_disambiguated"] == 3
    assert matching["ambiguous_reward_matches"] == 0
    assert matching["action_signature_unmatched"] == 0
    assert {
        value["trajectory_key"]: value["reward_breakdown"]["credit_transitions"][0]["action"]
        for value in matched
    } == {
        "100:7:0:0": "SEARCH",
        "100:7:0:1": "OPEN",
        "100:7:0:2": "KEEP",
        "100:7:0:3": "ANSWER",
    }
