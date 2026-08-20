#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_summary(predictions: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line
    ]
    row_failures: Counter[str] = Counter()
    action_errors: Counter[str] = Counter()
    joint_valid = 0
    completed = 0

    for row in rows:
        trajectory_completed = bool(row.get("trajectory_completed"))
        completed += int(trajectory_completed)
        categories: set[str] = set()
        for action in row.get("actions", []):
            result = action.get("tool_result") or {}
            if result.get("ok"):
                continue
            error = str(result.get("error") or "")
            if "invalid action JSON" in error:
                category = "invalid_action_json"
            elif "invalid citation" in error:
                category = "invalid_citation"
            elif "budget exhausted" in error:
                category = "budget_exhausted"
            elif "doc_id" in error or "reference" in error:
                category = "invalid_reference"
            else:
                category = "other_protocol"
            action_errors[category] += 1
            categories.add(category)
        if not trajectory_completed:
            categories.add("no_final_answer")
        row_failures.update(categories)
        joint_valid += int(trajectory_completed and not categories)

    count = len(rows)
    return {
        "trajectories": count,
        "unique_ids": len({row.get("id") for row in rows}),
        "completed": completed,
        "completion_rate": completed / count if count else 0.0,
        "joint_valid": joint_valid,
        "joint_valid_rate": joint_valid / count if count else 0.0,
        "row_failure_categories_nonexclusive": dict(sorted(row_failures.items())),
        "action_error_counts": dict(sorted(action_errors.items())),
    }


def build_index(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("step_*/predictions.jsonl.metrics.json")):
        match = re.fullmatch(r"step_(\d+)", metrics_path.parent.name)
        if match is None:
            continue
        predictions = metrics_path.parent / "predictions.jsonl"
        manifest = predictions.with_suffix(predictions.suffix + ".manifest.json")
        eval_metrics = predictions.with_suffix(predictions.suffix + ".eval_metrics.json")
        required = (predictions, manifest, eval_metrics)
        if not all(path.is_file() for path in required):
            continue
        evaluation = json.loads(eval_metrics.read_text(encoding="utf-8"))
        comparison_paths = sorted(metrics_path.parent.glob("paired_vs_step*.json"))
        comparison_path = comparison_paths[-1] if comparison_paths else None
        artifact_paths = [predictions, metrics_path, eval_metrics, manifest]
        if comparison_path is not None:
            artifact_paths.append(comparison_path)
        rows.append(
            {
                "step": int(match.group(1)),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
                "evaluation": evaluation.get("overall", evaluation),
                "protocol": _protocol_summary(predictions),
                "paired_comparison": (
                    json.loads(comparison_path.read_text(encoding="utf-8"))
                    if comparison_path is not None
                    else None
                ),
                "artifacts": {
                    path.name: {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifact_paths
                },
            }
        )

    return {"schema_version": 2, "completed_steps": [row["step"] for row in rows], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index completed periodic GRPO evaluations")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    value = build_index(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "completed_steps": value["completed_steps"]}))


if __name__ == "__main__":
    main()
