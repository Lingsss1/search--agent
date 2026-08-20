from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openstatesearch.eval.phase_b_audit import verify_phase_b_pool
from openstatesearch.eval.runner import evaluate_by_dataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verify_phase_b_pool_checks_exact_balanced_evidence(tmp_path: Path) -> None:
    datasets = ("2wiki", "hotpotqa", "musique")
    pool_records = [
        {
            "dataset": dataset,
            "id": f"{dataset}-{index}",
            "answers": ["answer"],
            "prediction": "answer",
            "evidence": [],
            "citations": [],
            "gold_evidence": [],
            "legal_evidence": [],
            "search_count": 0,
            "open_count": 0,
            "input_tokens": 1,
            "generated_tokens": 1,
        }
        for dataset in datasets
        for index in range(1500)
    ]
    hard_records = [
        {
            **record,
            "phase_b_hardness": {
                "criteria": {
                    "sft_wrong": True,
                    "incomplete_evidence": False,
                    "search_count_gt_3": False,
                }
            },
        }
        for record in pool_records
        if int(str(record["id"]).rsplit("-", 1)[1]) < 1000
    ]
    pool = tmp_path / "pool.jsonl"
    merged = tmp_path / "merged.jsonl"
    hard = tmp_path / "hard.jsonl"
    phase_a = tmp_path / "phase_a.jsonl"
    _write_jsonl(pool, pool_records)
    _write_jsonl(merged, pool_records)
    _write_jsonl(hard, hard_records)
    _write_jsonl(phase_a, [{"dataset": "hotpotqa", "id": "phase-a-only"}])
    gate_metrics = Path(f"{merged}.metrics.json")
    eval_metrics = Path(f"{merged}.eval_metrics.json")
    gate_metrics.write_text("{}\n", encoding="utf-8")
    eval_metrics.write_text("{}\n", encoding="utf-8")
    Path(f"{merged}.manifest.json").write_text(
        json.dumps(
            {
                "prompts": {"sha256": _sha256(pool)},
                "output": {"rows": 4500, "sha256": _sha256(merged)},
                "metrics": {"sha256": _sha256(eval_metrics)},
            }
        ),
        encoding="utf-8",
    )
    Path(f"{hard}.manifest.json").write_text(
        json.dumps(
            {
                "pool_rows": 4500,
                "trajectory_rows": 4500,
                "selected_rows": 3000,
                "per_dataset": 1000,
                "selection": [
                    "sft_wrong",
                    "incomplete_evidence",
                    "search_count_gt_3",
                ],
                "phase_a_identity_overlap": 0,
                "sha256": _sha256(hard),
                "datasets": {dataset: {"selected": 1000} for dataset in datasets},
            }
        ),
        encoding="utf-8",
    )

    report = verify_phase_b_pool(
        pool_path=pool,
        merged_path=merged,
        hard_path=hard,
        phase_a_path=phase_a,
    )
    assert report["passed"] is True
    assert report["dataset_counts"] == {
        "2wiki": 1000,
        "hotpotqa": 1000,
        "musique": 1000,
    }
    assert report["phase_a_identity_overlap"] == 0

    # Legacy merged manifests recorded only the metrics path. The verifier
    # accepts that schema only after recomputing all metrics from trajectories.
    eval_metrics.write_text(json.dumps(evaluate_by_dataset(pool_records)), encoding="utf-8")
    merged_manifest_path = Path(f"{merged}.manifest.json")
    merged_manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    merged_manifest["metrics"] = str(eval_metrics)
    merged_manifest_path.write_text(json.dumps(merged_manifest), encoding="utf-8")
    legacy_report = verify_phase_b_pool(
        pool_path=pool,
        merged_path=merged,
        hard_path=hard,
        phase_a_path=phase_a,
    )
    assert legacy_report["passed"] is True
