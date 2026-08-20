from __future__ import annotations

import json

import pytest

from openstatesearch.eval.abc_replay import (
    build_abc_replay_report,
    replay_episode,
    segment_rollout_rows,
)


def _prompt(question: str, last_result, *, query_history=None) -> str:
    observation = {
        "state": {
            "question": question,
            "constraints": [],
            "candidate_pool": [],
            "evidence": [],
            "conflicts": [],
            "query_history": query_history or [],
            "budget": {"search_left": 4, "open_left": 4, "token_left": 8192},
        },
        "opened_doc_ids": [],
        "remaining_turns": 16,
        "legal_action_space": {},
        "last_tool_result": last_result,
    }
    return "<|im_start|>user\n" + json.dumps(observation) + "<|im_end|>\n<|im_start|>assistant\n"


def test_replay_segments_group_and_scores_observed_transition() -> None:
    search_result = {
        "ok": True,
        "action": "SEARCH",
        "payload": {"query": "q", "target": "a", "hits": [{"doc_id": "gold"}]},
    }
    rows = [
        {
            "sample_idx": 0,
            "prompt": _prompt("question", None),
            "completion": json.dumps({"type": "SEARCH", "query": "q", "target_constraint": "a"}),
            "original_reward": -1.0,
        },
        {
            "sample_idx": 1,
            "prompt": _prompt("question", search_result, query_history=["q"]),
            "completion": "bad final action",
            "original_reward": -1.0,
        },
        {
            "sample_idx": 2,
            "prompt": _prompt("question", None),
            "completion": "another rollout",
            "original_reward": -1.0,
        },
    ]

    episodes = segment_rollout_rows(rows)
    replayed = replay_episode(episodes[0], (("gold", 2),))

    assert len(episodes) == 2
    assert replayed.question == "question"
    assert replayed.process_rewards == (0.025,)
    assert replayed.actions == ("SEARCH",)
    assert replayed.final_transition_unobserved


def test_replay_report_splits_final_training_advantage_by_action(tmp_path) -> None:
    rollout_root = tmp_path / "rollout" / "1"
    rollout_root.mkdir(parents=True)
    search_result = {
        "ok": True,
        "action": "SEARCH",
        "payload": {"query": "q", "target": "a", "hits": [{"doc_id": "gold"}]},
    }
    rows = []
    terminal_rewards = [-1.0, 0.0, 1.0, 2.0]
    for rollout_index, terminal_reward in enumerate(terminal_rewards):
        rows.extend(
            [
                {
                    "sample_idx": rollout_index * 2,
                    "prompt": _prompt("question", None),
                    "completion": json.dumps(
                        {
                            "type": "SEARCH",
                            "query": "q",
                            "target_constraint": "a",
                        }
                    ),
                    "original_reward": terminal_reward,
                },
                {
                    "sample_idx": rollout_index * 2 + 1,
                    "prompt": _prompt("question", search_result, query_history=["q"]),
                    "completion": "final action",
                    "original_reward": terminal_reward,
                },
            ]
        )
    (rollout_root / "0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "question": "question",
                "gold_evidence": [{"doc_id": "gold", "sent_ids": [2]}],
            }
        )
        + "\n"
    )

    report = build_abc_replay_report(rollout_root.parent, dataset)

    search = report["by_action"]["SEARCH"]
    assert search["count"] == 4
    assert search["mean_process_reward"] == pytest.approx(0.025)
    assert search["training_advantage_count"] == 4
    assert search["mean_training_advantage"] == pytest.approx(0.025)
    assert search["training_advantage_nonzero"] == 4
