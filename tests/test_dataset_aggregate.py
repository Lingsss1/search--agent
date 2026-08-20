from __future__ import annotations

import hashlib
import json

import pytest

from openstatesearch.eval.dataset_aggregate import aggregate_dataset_runs


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path, name: str, rows: int):
    predictions = tmp_path / name / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    values = []
    for index in range(rows):
        values.append(
            {
                "dataset": name,
                "id": str(index),
                "prediction": "answer",
                "answers": ["answer"],
                "evidence": [],
                "citations": [],
                "gold_evidence": [],
                "legal_evidence": [],
                "search_count": 1,
                "open_count": 1,
                "input_tokens": 100,
                "generated_tokens": 10,
                "model": "grpo",
                "model_provenance_sha256": "b" * 64,
                "state_mode": "external_state",
                "retriever": "live_web_duckduckgo",
                "memory_token_budget": 8192,
                "generation_token_budget": 8192,
                "retriever_provenance_sha256": "a" * 64,
            }
        )
    predictions.write_text("".join(json.dumps(value) + "\n" for value in values))
    manifest = tmp_path / name / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "output": {
                    "path": str(predictions),
                    "rows": rows,
                    "sha256": _sha(predictions),
                },
                "expected": {
                    "retriever_provenance_sha256": "a" * 64,
                    "model_provenance_sha256": "b" * 64,
                },
            }
        )
    )
    return manifest


def test_aggregate_dataset_runs_recomputes_weighted_metrics(tmp_path) -> None:
    first = _source(tmp_path, "xbench", 2)
    second = _source(tmp_path, "browsecomp_zh", 3)
    output = tmp_path / "chinese.jsonl"
    manifest = aggregate_dataset_runs(
        manifests={"xbench": first, "browsecomp_zh": second},
        output_path=output,
        expected_rows=5,
    )
    metrics = json.loads(output.with_suffix(".jsonl.eval_metrics.json").read_text())
    assert manifest["output"]["rows"] == 5
    assert metrics["overall"]["examples"] == 5
    assert set(metrics) == {"overall", "browsecomp_zh", "xbench"}


def test_aggregate_dataset_runs_rejects_wrong_total(tmp_path) -> None:
    first = _source(tmp_path, "one", 1)
    second = _source(tmp_path, "two", 1)
    with pytest.raises(ValueError, match="expected 3"):
        aggregate_dataset_runs(
            manifests={"one": first, "two": second},
            output_path=tmp_path / "bad.jsonl",
            expected_rows=3,
        )
