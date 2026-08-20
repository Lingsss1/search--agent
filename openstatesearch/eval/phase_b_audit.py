from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from openstatesearch.eval.runner import evaluate_by_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            records.append(value)
    return records


def _identities(records: list[dict[str, Any]], label: str) -> set[tuple[str, str]]:
    values = [(str(record.get("dataset", "")), str(record.get("id", ""))) for record in records]
    if any(not all(identity) for identity in values):
        raise ValueError(f"{label} contains a missing dataset/id")
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"{label} contains duplicate identities")
    return unique


def verify_phase_b_pool(
    *,
    pool_path: str | Path,
    merged_path: str | Path,
    hard_path: str | Path,
    phase_a_path: str | Path,
) -> dict[str, Any]:
    pool = Path(pool_path)
    merged = Path(merged_path)
    hard = Path(hard_path)
    phase_a = Path(phase_a_path)
    merged_manifest_path = Path(f"{merged}.manifest.json")
    hard_manifest_path = Path(f"{hard}.manifest.json")
    gate_metrics_path = Path(f"{merged}.metrics.json")
    eval_metrics_path = Path(f"{merged}.eval_metrics.json")
    required = (
        pool,
        merged,
        hard,
        phase_a,
        merged_manifest_path,
        hard_manifest_path,
        gate_metrics_path,
        eval_metrics_path,
    )
    for path in required:
        if not path.is_file():
            raise ValueError(f"Phase-B evidence file is missing: {path}")

    pool_records = _read_jsonl(pool)
    merged_records = _read_jsonl(merged)
    hard_records = _read_jsonl(hard)
    phase_a_records = _read_jsonl(phase_a)
    pool_ids = _identities(pool_records, "Phase-B pool")
    merged_ids = _identities(merged_records, "merged SFT trajectories")
    hard_ids = _identities(hard_records, "Phase-B hard set")
    phase_a_ids = _identities(phase_a_records, "Phase-A train set")
    if len(pool_records) != 4500 or len(merged_records) != 4500:
        raise ValueError("Phase-B pool and merged trajectories must each contain 4500 rows")
    if pool_ids != merged_ids:
        raise ValueError("merged SFT trajectory identities do not exactly match the pool")
    if len(hard_records) != 3000 or not hard_ids <= pool_ids:
        raise ValueError("Phase-B hard set must be a 3000-row subset of the pool")
    overlap = hard_ids & phase_a_ids
    if overlap:
        raise ValueError(f"Phase-B hard set overlaps Phase A: {sorted(overlap)[:3]}")
    dataset_counts = Counter(str(record["dataset"]) for record in hard_records)
    if dataset_counts != Counter({"2wiki": 1000, "hotpotqa": 1000, "musique": 1000}):
        raise ValueError(f"Phase-B hard set is not exactly balanced: {dataset_counts}")
    expected_selection = ["sft_wrong", "incomplete_evidence", "search_count_gt_3"]
    for record in hard_records:
        hardness = record.get("phase_b_hardness")
        criteria = hardness.get("criteria") if isinstance(hardness, dict) else None
        if not isinstance(criteria, dict) or not any(
            criteria.get(key) is True for key in expected_selection
        ):
            raise ValueError("Phase-B hard record does not satisfy a frozen hard criterion")

    merged_manifest = _read_json(merged_manifest_path)
    hard_manifest = _read_json(hard_manifest_path)
    merged_output = merged_manifest.get("output", {})
    if int(merged_output.get("rows", -1)) != 4500 or merged_output.get("sha256") != _sha256(merged):
        raise ValueError("merged trajectory manifest does not match its output")
    prompts = merged_manifest.get("prompts", {})
    if prompts.get("sha256") != _sha256(pool):
        raise ValueError("merged trajectory manifest does not match the frozen pool")
    metrics = merged_manifest.get("metrics")
    if isinstance(metrics, dict):
        if metrics.get("sha256") != _sha256(eval_metrics_path):
            raise ValueError("merged trajectory manifest does not match evaluation metrics")
        claimed_metrics_path = metrics.get("path")
        if claimed_metrics_path is not None and (
            not isinstance(claimed_metrics_path, str)
            or Path(claimed_metrics_path).resolve() != eval_metrics_path.resolve()
        ):
            raise ValueError("merged trajectory manifest names unexpected metrics")
    elif isinstance(metrics, str):
        # Early formal Phase-B pool artifacts used a path-only metrics field.
        # They cannot bind the sidecar by hash, so independently recompute the
        # complete aggregate before accepting that legacy evidence chain.
        if Path(metrics).resolve() != eval_metrics_path.resolve():
            raise ValueError("legacy merged manifest names unexpected metrics")
        if _read_json(eval_metrics_path) != evaluate_by_dataset(merged_records):
            raise ValueError("legacy evaluation metrics do not match merged trajectories")
    else:
        raise ValueError("merged trajectory manifest has invalid metrics evidence")
    if (
        int(hard_manifest.get("pool_rows", -1)) != 4500
        or int(hard_manifest.get("trajectory_rows", -1)) != 4500
        or int(hard_manifest.get("selected_rows", -1)) != 3000
        or int(hard_manifest.get("per_dataset", -1)) != 1000
        or hard_manifest.get("selection") != expected_selection
        or int(hard_manifest.get("phase_a_identity_overlap", -1)) != 0
        or hard_manifest.get("sha256") != _sha256(hard)
    ):
        raise ValueError("Phase-B hard-set manifest is inconsistent")
    manifest_datasets = hard_manifest.get("datasets", {})
    if set(manifest_datasets) != set(dataset_counts) or any(
        int(manifest_datasets[name].get("selected", -1)) != 1000 for name in dataset_counts
    ):
        raise ValueError("Phase-B manifest dataset balance is inconsistent")

    evidence = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in {
            "pool": pool,
            "merged_trajectories": merged,
            "merged_manifest": merged_manifest_path,
            "gate_metrics": gate_metrics_path,
            "eval_metrics": eval_metrics_path,
            "hard_set": hard,
            "hard_manifest": hard_manifest_path,
            "phase_a": phase_a,
        }.items()
    }
    return {
        "schema_version": 1,
        "passed": True,
        "pool_rows": len(pool_records),
        "trajectory_rows": len(merged_records),
        "selected_rows": len(hard_records),
        "phase_a_identity_overlap": 0,
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "selection": expected_selection,
        "evidence": evidence,
    }
