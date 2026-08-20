from __future__ import annotations

import hashlib
import json

from openstatesearch.eval.matrix_summary import (
    BUDGETS,
    EXPECTED,
    EXPERIMENTS,
    summarize_eval_matrix,
)
from openstatesearch.eval.runner import evaluate_by_dataset


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_matrix_summary_is_strict_and_emits_cost_and_failure_artifacts(tmp_path) -> None:
    root = tmp_path / "matrix"
    tag = "formal"
    for experiment in EXPERIMENTS:
        state_mode, retriever = EXPECTED[experiment]
        for budget in BUDGETS:
            run = root / experiment / tag / f"budget_{budget}"
            run.mkdir(parents=True)
            prediction = run / "predictions.jsonl"
            row = {
                "id": "shared-id",
                "dataset": "hotpotqa",
                "experiment": experiment,
                "state_mode": state_mode,
                "retriever": retriever,
                "memory_token_budget": budget,
                "prediction": "wrong" if experiment == "F" else "answer",
                "answers": ["answer"],
                "evidence": [],
                "gold_evidence": [{"doc_id": "d", "sent_ids": [0]}],
                "search_count": 4 if experiment == "F" else 2,
                "trajectory_completed": experiment != "F",
                "model": "grpo-model" if experiment in "EF" else "sft-model",
                "model_provenance_sha256": ("e" * 64 if experiment in "EF" else "d" * 64),
                "retriever_provenance_sha256": ("b" * 64 if experiment in "AB" else "c" * 64),
                "citations": [],
                "legal_evidence": [],
                "open_count": 1,
                "input_tokens": float(budget - 10),
                "generated_tokens": 10.0,
            }
            prediction.write_text(json.dumps(row) + "\n")
            metrics = evaluate_by_dataset([row])
            (run / "predictions.jsonl.eval_metrics.json").write_text(json.dumps(metrics))
            identity_sha = hashlib.sha256(b"hotpotqa\tshared-id").hexdigest()
            manifest = {
                "output": {
                    "sha256": _sha(prediction),
                    "rows": 1,
                    "identity_sha256": identity_sha,
                },
                "expected": {
                    "retriever_provenance_sha256": ("b" * 64 if experiment in "AB" else "c" * 64),
                    "model_provenance_sha256": ("e" * 64 if experiment in "EF" else "d" * 64),
                },
            }
            (run / "predictions.jsonl.manifest.json").write_text(json.dumps(manifest))

    summary = summarize_eval_matrix(
        matrix_root=root,
        tag=tag,
        output_dir=tmp_path / "summary",
        failure_limit=50,
    )

    assert len(summary["experiments"]) == 12
    assert summary["prompt_identities"]["rows"] == 1
    assert summary["models"] == {
        "sft": "sft-model",
        "sft_provenance_sha256": "d" * 64,
        "grpo": "grpo-model",
        "grpo_provenance_sha256": "e" * 64,
    }
    assert summary["retrievers"] == {
        "base_hybrid_sha256": "b" * 64,
        "lrat_hybrid_sha256": "c" * 64,
    }
    assert summary["accuracy_token_cost_curve"]["points"] == 12
    assert summary["failure_cases"]["rows"] == 1
    assert len(summary["comparisons"]["external_state"]) == 6
    assert len(summary["comparisons"]["grpo"]) == 4
