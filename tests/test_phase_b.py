from __future__ import annotations

import pytest

from openstatesearch.data.phase_b import phase_b_features, select_phase_b_hard


def _prompt(dataset: str, identity: str, answer: str = "right") -> dict:
    return {
        "dataset": dataset,
        "id": identity,
        "question": identity,
        "answer": answer,
        "answers": [answer],
        "gold_evidence": [{"doc_id": "doc", "sent_ids": [1]}],
    }


def _trajectory(dataset: str, identity: str, answer: str, keep: bool, searches: int) -> dict:
    actions = [
        {"action": {"type": "SEARCH", "query": f"q-{index}", "target_constraint": ""}}
        for index in range(searches)
    ]
    if keep:
        actions.append(
            {
                "action": {
                    "type": "KEEP",
                    "doc_id": "doc",
                    "sent_ids": [1],
                    "claim": "claim",
                    "constraint_id": "constraint",
                }
            }
        )
    return {
        "dataset": dataset,
        "id": identity,
        "actions": actions,
        "final_answer": {"answer": answer, "citations": []},
    }


def test_phase_b_features_match_all_frozen_rules() -> None:
    features = phase_b_features(
        _prompt("hotpotqa", "one"),
        _trajectory("hotpotqa", "one", "wrong", keep=False, searches=4),
    )
    assert features["criteria"] == {
        "sft_wrong": True,
        "incomplete_evidence": True,
        "search_count_gt_3": True,
    }


def test_phase_b_selection_is_balanced_and_deterministic() -> None:
    prompts = [
        _prompt(dataset, f"{dataset}-{index}")
        for dataset in ("2wiki", "hotpotqa", "musique")
        for index in range(4)
    ]
    trajectories = [
        _trajectory(
            prompt["dataset"],
            prompt["id"],
            "wrong" if int(prompt["id"].rsplit("-", 1)[1]) < 3 else "right",
            keep=False,
            searches=4,
        )
        for prompt in prompts
    ]
    selected, manifest = select_phase_b_hard(prompts, trajectories, per_dataset=2)
    reversed_selected, _ = select_phase_b_hard(
        reversed(prompts), reversed(trajectories), per_dataset=2
    )
    assert [record["id"] for record in selected] == [record["id"] for record in reversed_selected]
    assert manifest["selected_rows"] == 6
    assert {record["dataset"] for record in selected} == {"2wiki", "hotpotqa", "musique"}


def test_phase_b_selection_fails_when_hard_pool_is_too_small() -> None:
    prompts = [_prompt("hotpotqa", "easy")]
    trajectories = [_trajectory("hotpotqa", "easy", "right", keep=True, searches=1)]
    with pytest.raises(ValueError, match="no hard prompts"):
        select_phase_b_hard(prompts, trajectories, per_dataset=1)
