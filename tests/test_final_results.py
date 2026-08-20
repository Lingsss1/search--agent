from __future__ import annotations

import hashlib
import json

import pytest

from openstatesearch.demo import run_demo
from openstatesearch.eval.final_results import (
    REQUIRED_DATASET_ROWS,
    assemble_final_results,
)
from scripts.replay_formal_demo import build_formal_replay_artifact


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_results_require_and_link_all_acceptance_evidence(tmp_path) -> None:
    curve = tmp_path / "curve.json"
    failures = tmp_path / "failures.jsonl"
    curve.write_text("[]")
    failures.write_text("{}\n")
    overall = {
        "examples": 10,
        "answer_f1": 0.9,
        "avg_total_tokens": 100.0,
        "avg_input_tokens": 90.0,
        "avg_generated_tokens": 10.0,
        "avg_search": 1.0,
        "citation_validity_micro": 1.0,
        "citation_precision_micro": 0.9,
    }
    selected_metrics = {
        "trajectories": 50,
        "valid_rate": 1.0,
        "reward_components": {
            name: {"mean": 1.0}
            for name in (
                "total",
                "answer_f1",
                "support_recall",
                "citation_precision",
                "protocol_penalty",
            )
        },
    }
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    adapter_files = []
    for name, contents in (
        ("adapter_config.json", "{}\n"),
        ("adapter_model.safetensors", "weights\n"),
    ):
        path = adapter / name
        path.write_text(contents)
        adapter_files.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    checkpoint = tmp_path / "checkpoint_manifest.json"
    _write(
        checkpoint,
        {
            "kind": "grpo_lora_checkpoint",
            "step": 187,
            "output": {"path": str(adapter), "files": adapter_files},
            "audit_metrics": selected_metrics,
        },
    )
    merge_manifest = tmp_path / "merge_manifest.json"
    _write(
        merge_manifest,
        {"adapter": {"path": str(adapter), "files": adapter_files}},
    )
    grpo_model_sha = _sha(merge_manifest)
    lrat_retriever_sha = "c" * 64

    demo_trace = run_demo()
    formal_record = {
        "dataset": "demo",
        "id": "formal-row-1",
        "question": demo_trace["question"],
        "actions": [
            {"action": event["action"], "tool_result": event["result"]}
            for event in demo_trace["events"]
        ],
        "trajectory_completed": True,
        "search_count": sum(
            event["action"]["type"] in {"SEARCH", "VERIFY"} for event in demo_trace["events"]
        ),
        "open_count": sum(event["action"]["type"] == "OPEN" for event in demo_trace["events"]),
        "generated_tokens": 100,
        "generation_token_budget": 8192,
        "model_provenance_sha256": grpo_model_sha,
        "retriever_provenance_sha256": lrat_retriever_sha,
        "run_config_sha256": "f" * 64,
    }
    experiments = {}
    for letter in "ABCDEF":
        for budget in (4096, 8192):
            source = {}
            for name in ("predictions", "metrics", "manifest"):
                path = tmp_path / "matrix_sources" / f"{letter}_{budget}_{name}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if letter == "F" and budget == 8192 and name == "predictions":
                    path.write_text(json.dumps(formal_record) + "\n")
                else:
                    path.write_text(f"{letter}-{budget}-{name}\n")
                source[name] = str(path)
                source[f"{name}_sha256"] = _sha(path)
            experiments[f"{letter}_{budget}"] = {
                "metrics": {"overall": overall},
                "source": source,
            }
    external = {
        "target": "D",
        "baseline": "C",
        "memory_token_budget": 8192,
        "answer_f1_gain_percentage_points": 2.0,
        "total_token_reduction_fraction": 0.25,
        "search_reduction_fraction": 0.1,
        "quality_non_decreasing": True,
    }
    grpo = {
        "target": "F",
        "baseline": "D",
        "memory_token_budget": 8192,
        "answer_f1_gain_percentage_points": 4.0,
        "total_token_reduction_fraction": 0.1,
        "search_reduction_fraction": 0.2,
        "quality_non_decreasing": True,
    }
    matrix_path = tmp_path / "matrix.json"
    _write(
        matrix_path,
        {
            "experiments": experiments,
            "prompt_identities": {"rows": 10, "sha256": "a" * 64},
            "models": {
                "sft": "sft-model",
                "sft_provenance_sha256": "d" * 64,
                "grpo": "grpo-model",
                "grpo_provenance_sha256": grpo_model_sha,
            },
            "retrievers": {
                "base_hybrid_sha256": "b" * 64,
                "lrat_hybrid_sha256": lrat_retriever_sha,
            },
            "comparisons": {"external_state": [external], "grpo": [grpo]},
            "accuracy_token_cost_curve": {"path": str(curve), "sha256": _sha(curve), "points": 12},
            "failure_cases": {"path": str(failures), "sha256": _sha(failures), "rows": 1},
        },
    )
    retriever = tmp_path / "retriever.json"
    _write(retriever, {"pairwise_absolute_gains": {"R4-R0": {"recall_at_20": 0.04}}})
    reward = tmp_path / "reward.json"
    _write(
        reward,
        {
            "step": 187,
            "sample_size": 50,
            "matching": {"unmatched_raw_records": 0},
            "selected_metrics": selected_metrics,
            "model_provenance_sha256": grpo_model_sha,
            "model_provenance": {
                "path": str(merge_manifest),
                "sha256": grpo_model_sha,
            },
            "checkpoint_manifest": {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
            },
        },
    )
    demo_dir = tmp_path / "formal_demo"
    build_formal_replay_artifact(
        tmp_path / "matrix_sources" / "F_8192_predictions.json",
        demo_dir,
    )
    demo = demo_dir / "manifest.json"
    costs = {}
    for name in ("main_test", "in_domain_dev", "chinese_test"):
        path = tmp_path / f"{name}.json"
        _write(path, {"overall": overall})
        costs[name] = path
    dataset_manifests = {}
    for name, rows in REQUIRED_DATASET_ROWS.items():
        prediction = tmp_path / "datasets" / name / "predictions.jsonl"
        metrics = tmp_path / "datasets" / name / "metrics.json"
        prompts = tmp_path / "datasets" / name / "prompts.jsonl"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text("{}\n")
        _write(metrics, {"overall": overall})
        prompts.write_text("{}\n")
        manifest = tmp_path / "datasets" / name / "manifest.json"
        _write(
            manifest,
            {
                "output": {
                    "path": str(prediction),
                    "rows": rows,
                    "sha256": _sha(prediction),
                },
                "metrics": {"path": str(metrics), "sha256": _sha(metrics)},
                "prompts": {"path": str(prompts), "sha256": _sha(prompts)},
                "expected": {
                    "retriever_provenance_sha256": "d" * 64,
                    "model_provenance_sha256": grpo_model_sha,
                },
            },
        )
        dataset_manifests[name] = manifest
    thresholds = {
        "retriever_recall_at_20_absolute_gain": 0.03,
        "external_state_token_reduction_at_same_f1": 0.2,
        "external_state_f1_gain_at_same_tokens": 2.0,
        "grpo_f1_gain": 3.0,
        "grpo_search_reduction_if_quality_stable": 0.15,
        "citation_validity": 0.98,
        "citation_precision": 0.85,
        "require_cost_records": ["main_test", "in_domain_dev", "chinese_test"],
        "completion_artifacts": [
            "A-F results",
            "R0-R4 retriever comparison",
            "reward audit",
            "accuracy-token cost curve",
            "failure cases",
            "replayable demo",
        ],
    }
    results, report = assemble_final_results(
        matrix_summary_path=matrix_path,
        retriever_results_path=retriever,
        reward_audit_summary_path=reward,
        demo_manifest_path=demo,
        cost_metric_paths=costs,
        dataset_manifest_paths=dataset_manifests,
        thresholds=thresholds,
        output_path=tmp_path / "results.json",
    )
    assert report["passed"]
    assert results["retriever_recall_at_20_gain"] == 0.04
    assert set(results["evidence"]["cost_records"]) == set(costs)
    assert set(results["evidence"]["dataset_records"]) == set(REQUIRED_DATASET_ROWS)

    valid_demo = json.loads(demo.read_text())
    _write(demo, {"exact": True})
    with pytest.raises(ValueError, match="exact formal F/8192"):
        assemble_final_results(
            matrix_summary_path=matrix_path,
            retriever_results_path=retriever,
            reward_audit_summary_path=reward,
            demo_manifest_path=demo,
            cost_metric_paths=costs,
            dataset_manifest_paths=dataset_manifests,
            thresholds=thresholds,
            output_path=tmp_path / "rejected-toy-demo.json",
        )
    _write(demo, valid_demo)

    valid_reward = json.loads(reward.read_text())
    mismatched_reward = {**valid_reward, "model_provenance_sha256": "0" * 64}
    _write(reward, mismatched_reward)
    with pytest.raises(ValueError, match="final GRPO model provenance"):
        assemble_final_results(
            matrix_summary_path=matrix_path,
            retriever_results_path=retriever,
            reward_audit_summary_path=reward,
            demo_manifest_path=demo,
            cost_metric_paths=costs,
            dataset_manifest_paths=dataset_manifests,
            thresholds=thresholds,
            output_path=tmp_path / "rejected-reward.json",
        )
