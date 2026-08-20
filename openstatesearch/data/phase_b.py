from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Iterable

from openstatesearch.rewards.metrics import (
    answer_exact_match,
    evidence_precision_recall_f1,
)


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("dataset", "")), str(record.get("id", ""))


def _evidence(items: Iterable[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(item["doc_id"]), int(sent_id))
        for item in items
        for sent_id in item.get("sent_ids", [])
    }


def phase_b_features(prompt: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    final_answer = trajectory.get("final_answer")
    prediction = str(final_answer.get("answer", "")) if isinstance(final_answer, dict) else ""
    references = prompt.get("answers", [prompt.get("answer", "")])
    if isinstance(references, str):
        references = [references]
    answer_em = answer_exact_match(prediction, [str(value) for value in references])

    kept: list[dict[str, Any]] = []
    search_count = 0
    for step in trajectory.get("actions", []):
        action = step.get("action") if isinstance(step, dict) else None
        if not isinstance(action, dict):
            continue
        action_type = str(action.get("type", "")).upper()
        if action_type == "SEARCH":
            search_count += 1
        elif action_type == "KEEP" and action.get("doc_id") is not None:
            kept.append(action)
    _, support_recall, _ = evidence_precision_recall_f1(
        _evidence(kept), _evidence(prompt.get("gold_evidence", []))
    )
    criteria = {
        "sft_wrong": answer_em == 0.0,
        "incomplete_evidence": support_recall < 1.0,
        "search_count_gt_3": search_count > 3,
    }
    return {
        "prediction": prediction,
        "answer_em": answer_em,
        "support_recall": support_recall,
        "search_count": search_count,
        "criteria": criteria,
    }


def select_phase_b_hard(
    prompts: Iterable[dict[str, Any]],
    trajectories: Iterable[dict[str, Any]],
    per_dataset: int = 1000,
    seed: int = 36,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    trajectory_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for trajectory in trajectories:
        key = _identity(trajectory)
        if not all(key):
            raise ValueError(f"trajectory is missing dataset/id: {key!r}")
        if key in trajectory_by_id:
            raise ValueError(f"duplicate trajectory identity: {key}")
        trajectory_by_id[key] = trajectory

    ranked_by_dataset: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = defaultdict(list)
    prompt_ids: set[tuple[str, str]] = set()
    for prompt in prompts:
        key = _identity(prompt)
        if not all(key):
            raise ValueError(f"prompt is missing dataset/id: {key!r}")
        if key in prompt_ids:
            raise ValueError(f"duplicate prompt identity: {key}")
        prompt_ids.add(key)
        if key not in trajectory_by_id:
            raise ValueError(f"missing SFT trajectory for prompt: {key}")
        features = phase_b_features(prompt, trajectory_by_id[key])
        if not any(features["criteria"].values()):
            continue
        digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).hexdigest()
        hardness = (
            -sum(features["criteria"].values()),
            features["answer_em"],
            features["support_recall"],
            -features["search_count"],
            digest,
        )
        enriched = dict(prompt)
        enriched["phase_b_hardness"] = features
        ranked_by_dataset[key[0]].append((hardness, enriched))

    extra = sorted(set(trajectory_by_id) - prompt_ids)
    if extra:
        raise ValueError(f"trajectories not present in pool: {extra[:3]}")

    selected: list[dict[str, Any]] = []
    dataset_summary: dict[str, Any] = {}
    for dataset in sorted(ranked_by_dataset):
        ranked = sorted(ranked_by_dataset[dataset], key=lambda item: item[0])
        if len(ranked) < per_dataset:
            raise ValueError(
                f"dataset {dataset} has only {len(ranked)} hard prompts; need {per_dataset}"
            )
        values = [record for _, record in ranked[:per_dataset]]
        selected.extend(values)
        criteria_counts = Counter(
            criterion
            for record in values
            for criterion, matched in record["phase_b_hardness"]["criteria"].items()
            if matched
        )
        dataset_summary[dataset] = {
            "hard_candidates": len(ranked),
            "selected": len(values),
            "criteria_matches_in_selected": dict(sorted(criteria_counts.items())),
        }

    datasets_in_pool = sorted({key[0] for key in prompt_ids})
    missing_datasets = sorted(set(datasets_in_pool) - set(ranked_by_dataset))
    if missing_datasets:
        raise ValueError(f"datasets contain no hard prompts: {missing_datasets}")
    return selected, {
        "schema_version": 1,
        "seed": seed,
        "selection": ["sft_wrong", "incomplete_evidence", "search_count_gt_3"],
        "ranking": "criteria_count desc, answer_em asc, support_recall asc, search_count desc, seeded hash",
        "pool_rows": len(prompt_ids),
        "trajectory_rows": len(trajectory_by_id),
        "selected_rows": len(selected),
        "per_dataset": per_dataset,
        "datasets": dataset_summary,
    }
