#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.agent.schemas import ActionValidationError, parse_action
from openstatesearch.eval.runner import read_predictions
from openstatesearch.rewards.metrics import (
    answer_f1,
    evidence_precision_recall_f1,
    normalize_answer,
)


CORE_TRAJECTORY_KEYS = (
    "actions",
    "prediction",
    "trajectory_completed",
    "evidence",
    "citations",
    "generated_tokens",
    "input_tokens",
    "search_count",
    "open_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(record: dict[str, Any]) -> tuple[str, str]:
    identity = (str(record.get("dataset", "")), str(record.get("id", "")))
    if not all(identity):
        raise ValueError("every gate record requires non-empty dataset and id")
    return identity


def _indexed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records = read_predictions(path)
    result = {_identity(record): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"duplicate identities in {path}")
    return result


def _refs(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(item["doc_id"]), int(sent_id))
        for item in items
        for sent_id in item.get("sent_ids", [])
    }


def _reference_recall(prediction: str, references: list[str]) -> float:
    predicted = Counter(normalize_answer(prediction).split())
    values = []
    for reference in references:
        expected = Counter(normalize_answer(reference).split())
        values.append(
            sum((predicted & expected).values()) / sum(expected.values()) if expected else 0.0
        )
    return max(values, default=0.0)


def _record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    references = record["answers"]
    if isinstance(references, str):
        references = [references]
    references = [str(item) for item in references]
    predicted_evidence = _refs(record.get("evidence", []))
    citations = _refs(record.get("citations", []))
    gold = _refs(record.get("gold_evidence", []))
    evidence_precision, support_recall, support_f1 = evidence_precision_recall_f1(
        predicted_evidence, gold
    )
    citation_precision, _, _ = evidence_precision_recall_f1(citations, gold)
    malformed_steps = 0
    failed_tool_steps = 0
    for step in record.get("actions", []):
        try:
            parse_action(step.get("action"))
        except (ActionValidationError, TypeError):
            malformed_steps += 1
        result = step.get("tool_result")
        failed_tool_steps += int(isinstance(result, dict) and not result.get("ok", False))
    completed = bool(record.get("trajectory_completed", False))
    return {
        "answer_f1": answer_f1(str(record.get("prediction", "")), references),
        "reference_token_recall": _reference_recall(str(record.get("prediction", "")), references),
        "prediction_tokens": len(normalize_answer(str(record.get("prediction", ""))).split()),
        "evidence_precision": evidence_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "citation_precision": citation_precision,
        "completed": completed,
        "malformed_steps": malformed_steps,
        "failed_tool_steps": failed_tool_steps,
        "protocol_clean": completed and malformed_steps == 0 and failed_tool_steps == 0,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap_ci(values: list[float], *, seed: int, samples: int) -> list[float]:
    if samples < 100:
        raise ValueError("bootstrap-samples must be at least 100")
    rng = random.Random(seed)
    draws = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    return [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]]


def compare(
    baseline_path: Path,
    target_path: Path,
    *,
    seed: int = 36,
    bootstrap_samples: int = 20_000,
) -> dict[str, Any]:
    baseline = _indexed(baseline_path)
    target = _indexed(target_path)
    if set(baseline) != set(target):
        raise ValueError(
            f"paired gate identities differ: baseline={len(baseline)} target={len(target)} "
            f"common={len(set(baseline) & set(target))}"
        )
    records = []
    by_dataset: dict[str, list[float]] = defaultdict(list)
    transitions: Counter[str] = Counter()
    for identity in sorted(baseline):
        actions_equal = baseline[identity].get("actions") == target[identity].get("actions")
        core_trajectory_equal = all(
            baseline[identity].get(key) == target[identity].get(key) for key in CORE_TRAJECTORY_KEYS
        )
        base_metrics = _record_metrics(baseline[identity])
        target_metrics = _record_metrics(target[identity])
        delta = target_metrics["answer_f1"] - base_metrics["answer_f1"]
        by_dataset[identity[0]].append(delta)
        transitions[
            f"completion_{int(base_metrics['completed'])}_to_{int(target_metrics['completed'])}"
        ] += 1
        transitions[
            f"protocol_clean_{int(base_metrics['protocol_clean'])}_to_{int(target_metrics['protocol_clean'])}"
        ] += 1
        records.append(
            {
                "dataset": identity[0],
                "id": identity[1],
                "question": baseline[identity].get("question"),
                "references": baseline[identity].get("answers"),
                "baseline_prediction": baseline[identity].get("prediction"),
                "target_prediction": target[identity].get("prediction"),
                "baseline": base_metrics,
                "target": target_metrics,
                "answer_f1_delta": delta,
                "actions_exact_match": actions_equal,
                "core_trajectory_exact_match": core_trajectory_equal,
            }
        )
    deltas = [float(item["answer_f1_delta"]) for item in records]
    target_completed = [
        float(item["answer_f1_delta"]) for item in records if item["target"]["completed"]
    ]
    both_clean = [
        float(item["answer_f1_delta"])
        for item in records
        if item["baseline"]["protocol_clean"] and item["target"]["protocol_clean"]
    ]
    verbosity_regressions = [
        item
        for item in records
        if item["answer_f1_delta"] < 0
        and item["target"]["completed"]
        and item["target"]["reference_token_recall"] == 1.0
    ]
    incomplete_regressions = [
        item for item in records if item["answer_f1_delta"] < 0 and not item["target"]["completed"]
    ]
    return {
        "schema_version": 1,
        "sources": {
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "target": {"path": str(target_path), "sha256": _sha256(target_path)},
        },
        "examples": len(records),
        "answer_f1": {
            "baseline_mean": _mean([item["baseline"]["answer_f1"] for item in records]),
            "target_mean": _mean([item["target"]["answer_f1"] for item in records]),
            "mean_delta": _mean(deltas),
            "paired_bootstrap_95_ci": _bootstrap_ci(deltas, seed=seed, samples=bootstrap_samples),
            "improved": sum(value > 0 for value in deltas),
            "regressed": sum(value < 0 for value in deltas),
            "tied": sum(value == 0 for value in deltas),
            "target_completed_subset_mean_delta": _mean(target_completed),
            "both_protocol_clean_subset_mean_delta": _mean(both_clean),
        },
        "prediction_tokens": {
            "baseline_mean": _mean([item["baseline"]["prediction_tokens"] for item in records]),
            "target_mean": _mean([item["target"]["prediction_tokens"] for item in records]),
        },
        "exact_match": {
            "actions": sum(item["actions_exact_match"] for item in records),
            "core_trajectory": sum(item["core_trajectory_exact_match"] for item in records),
        },
        "attribution": {
            "verbosity_regressions_with_full_reference_recall": len(verbosity_regressions),
            "verbosity_regression_delta_sum": sum(
                item["answer_f1_delta"] for item in verbosity_regressions
            ),
            "target_incomplete_regressions": len(incomplete_regressions),
            "target_incomplete_regression_delta_sum": sum(
                item["answer_f1_delta"] for item in incomplete_regressions
            ),
        },
        "transitions": dict(sorted(transitions.items())),
        "dataset_mean_delta": {name: _mean(values) for name, values in sorted(by_dataset.items())},
        "records": sorted(records, key=lambda item: item["answer_f1_delta"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit two gate runs on identical prompts")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    report = compare(
        args.baseline,
        args.target,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
