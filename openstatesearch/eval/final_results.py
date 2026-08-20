from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openstatesearch.eval.acceptance import audit_acceptance
from openstatesearch.eval.replay import replay_frozen_trajectory, trajectory_from_prediction


REQUIRED_DATASET_ROWS = {
    "in_domain_dev": 1500,
    "browsecomp_plus": 830,
    "xbench_deepsearch": 100,
    "browsecomp_zh": 289,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing evidence file: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _verify_declared_evidence(value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} evidence is missing")
    path = Path(str(value.get("path", "")))
    expected_sha = value.get("sha256")
    if not path.is_file() or not isinstance(expected_sha, str) or _sha256(path) != expected_sha:
        raise ValueError(f"{label} evidence does not match its declared SHA")
    return path


def _inventory(entries: Any, label: str) -> dict[str, tuple[int, str]]:
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} file inventory is missing")
    inventory: dict[str, tuple[int, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{label} file inventory is malformed")
        name = Path(str(entry.get("path", ""))).name
        size = entry.get("bytes")
        sha256 = entry.get("sha256")
        if (
            not name
            or name in inventory
            or not isinstance(size, int)
            or size < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ValueError(f"{label} file inventory is malformed")
        inventory[name] = (size, sha256)
    return inventory


def _verify_inventory(root_value: Any, entries: Any, label: str) -> dict[str, tuple[int, str]]:
    root = Path(str(root_value or ""))
    inventory = _inventory(entries, label)
    if not root.is_dir():
        raise ValueError(f"{label} directory is missing: {root}")
    for name, (expected_size, expected_sha) in inventory.items():
        path = root / name
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256(path) != expected_sha
        ):
            raise ValueError(f"{label} file evidence mismatch: {path}")
    return inventory


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            values.append(value)
    return values


def _select_comparison(
    comparisons: list[dict[str, Any]], target: str, baseline: str, budget: int
) -> dict[str, Any]:
    values = [
        item
        for item in comparisons
        if item.get("target") == target
        and item.get("baseline") == baseline
        and item.get("memory_token_budget") == budget
    ]
    if len(values) != 1:
        raise ValueError(
            f"expected one comparison {target}-{baseline} at budget {budget}; found {len(values)}"
        )
    return values[0]


def assemble_final_results(
    *,
    matrix_summary_path: str | Path,
    retriever_results_path: str | Path,
    reward_audit_summary_path: str | Path,
    demo_manifest_path: str | Path,
    cost_metric_paths: dict[str, str | Path],
    dataset_manifest_paths: dict[str, str | Path],
    thresholds: dict[str, Any],
    output_path: str | Path,
    retriever_target: str = "R4",
    retriever_baseline: str = "R0",
    external_target: str = "D",
    external_baseline: str = "C",
    grpo_target: str = "F",
    grpo_baseline: str = "D",
    comparison_budget: int = 8192,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_path = Path(matrix_summary_path)
    retriever_path = Path(retriever_results_path)
    audit_path = Path(reward_audit_summary_path)
    demo_path = Path(demo_manifest_path)
    matrix = _load(matrix_path)
    retrievers = _load(retriever_path)
    reward_audit = _load(audit_path)
    demo = _load(demo_path)

    expected_runs = {f"{letter}_{budget}" for letter in "ABCDEF" for budget in (4096, 8192)}
    if set(matrix.get("experiments", {})) != expected_runs:
        raise ValueError("matrix summary must contain exactly A-F at 4096 and 8192 tokens")
    prompt_identities = matrix.get("prompt_identities", {})
    if int(prompt_identities.get("rows", 0)) <= 0:
        raise ValueError("matrix summary has no common prompt identities")
    identity_sha = prompt_identities.get("sha256")
    if not isinstance(identity_sha, str) or len(identity_sha) != 64:
        raise ValueError("matrix summary has no valid prompt identity SHA")
    models = matrix.get("models", {})
    if not models.get("sft") or not models.get("grpo") or models["sft"] == models["grpo"]:
        raise ValueError("matrix summary must identify distinct SFT and GRPO models")
    sft_model_sha = models.get("sft_provenance_sha256")
    grpo_model_sha = models.get("grpo_provenance_sha256")
    if (
        not isinstance(sft_model_sha, str)
        or len(sft_model_sha) != 64
        or not isinstance(grpo_model_sha, str)
        or len(grpo_model_sha) != 64
        or sft_model_sha == grpo_model_sha
    ):
        raise ValueError("matrix summary must identify distinct model provenance")
    retriever_provenance = matrix.get("retrievers", {})
    base_retriever_sha = retriever_provenance.get("base_hybrid_sha256")
    lrat_retriever_sha = retriever_provenance.get("lrat_hybrid_sha256")
    if (
        not isinstance(base_retriever_sha, str)
        or len(base_retriever_sha) != 64
        or not isinstance(lrat_retriever_sha, str)
        or len(lrat_retriever_sha) != 64
        or base_retriever_sha == lrat_retriever_sha
    ):
        raise ValueError("matrix summary must identify distinct retriever provenance")
    for run_name, run in matrix["experiments"].items():
        source = run.get("source", {})
        for source_name in ("predictions", "metrics", "manifest"):
            source_path = Path(source.get(source_name, ""))
            expected_sha = source.get(f"{source_name}_sha256")
            if not source_path.is_file() or _sha256(source_path) != expected_sha:
                raise ValueError(f"matrix source evidence mismatch for {run_name} {source_name}")
    curve = Path(matrix.get("accuracy_token_cost_curve", {}).get("path", ""))
    failures = Path(matrix.get("failure_cases", {}).get("path", ""))
    if int(matrix.get("accuracy_token_cost_curve", {}).get("points", 0)) != 12:
        raise ValueError("accuracy-token cost curve must contain 12 points")
    if int(matrix.get("failure_cases", {}).get("rows", 0)) <= 0:
        raise ValueError("formal failure-case artifact is empty")
    if _sha256(curve) != matrix["accuracy_token_cost_curve"]["sha256"]:
        raise ValueError("cost curve SHA mismatch")
    if _sha256(failures) != matrix["failure_cases"]["sha256"]:
        raise ValueError("failure cases SHA mismatch")

    gain_key = f"{retriever_target}-{retriever_baseline}"
    gain = retrievers.get("pairwise_absolute_gains", {}).get(gain_key, {}).get("recall_at_20")
    if not isinstance(gain, (int, float)):
        raise ValueError(f"retriever result has no {gain_key} Recall@20 gain")
    external = _select_comparison(
        matrix["comparisons"]["external_state"],
        external_target,
        external_baseline,
        comparison_budget,
    )
    grpo = _select_comparison(
        matrix["comparisons"]["grpo"],
        grpo_target,
        grpo_baseline,
        comparison_budget,
    )
    final_key = f"{grpo_target}_{comparison_budget}"
    final_metrics = matrix["experiments"][final_key]["metrics"]["overall"]

    matching = reward_audit.get("matching", {})
    if int(reward_audit.get("sample_size", 0)) != 50:
        raise ValueError("reward audit must contain exactly 50 selected trajectories")
    if int(matching.get("unmatched_raw_records", -1)) != 0:
        raise ValueError("reward audit contains unlinked raw records")
    selected_metrics = reward_audit.get("selected_metrics", {})
    if int(selected_metrics.get("trajectories", 0)) != 50:
        raise ValueError("reward audit summary must report metrics for 50 trajectories")
    if not isinstance(selected_metrics.get("valid_rate"), (int, float)):
        raise ValueError("reward audit summary is missing valid_rate")
    reward_components = selected_metrics.get("reward_components", {})
    for field in (
        "total",
        "answer_f1",
        "support_recall",
        "citation_precision",
        "protocol_penalty",
    ):
        if not isinstance(reward_components.get(field, {}).get("mean"), (int, float)):
            raise ValueError(f"reward audit summary is missing mean {field}")
    if reward_audit.get("model_provenance_sha256") != grpo_model_sha:
        raise ValueError("reward audit is not bound to the final GRPO model provenance")
    reward_model_path = _verify_declared_evidence(
        reward_audit.get("model_provenance"), "reward-audit model provenance"
    )
    if _sha256(reward_model_path) != grpo_model_sha:
        raise ValueError("reward-audit model provenance differs from matrix GRPO provenance")
    checkpoint_path = _verify_declared_evidence(
        reward_audit.get("checkpoint_manifest"), "reward-audit checkpoint manifest"
    )
    checkpoint = _load(checkpoint_path)
    if checkpoint.get("kind") != "grpo_lora_checkpoint":
        raise ValueError("reward audit checkpoint is not a GRPO LoRA checkpoint")
    if int(checkpoint.get("step", -1)) != int(reward_audit.get("step", -2)):
        raise ValueError("reward audit and checkpoint describe different steps")
    if checkpoint.get("audit_metrics") != selected_metrics:
        raise ValueError("reward audit metrics differ from the archived checkpoint audit")
    checkpoint_output = checkpoint.get("output", {})
    checkpoint_inventory = _verify_inventory(
        checkpoint_output.get("path"),
        checkpoint_output.get("files"),
        "reward-audited checkpoint adapter",
    )
    model_provenance = _load(reward_model_path)
    merged_adapter = model_provenance.get("adapter", {})
    if (
        Path(str(merged_adapter.get("path", ""))).resolve()
        != Path(str(checkpoint_output.get("path", ""))).resolve()
    ):
        raise ValueError("final GRPO model is not merged from the reward-audited adapter")
    if _inventory(merged_adapter.get("files"), "merged-model adapter") != checkpoint_inventory:
        raise ValueError("merged-model adapter inventory differs from the audited checkpoint")

    if demo.get("kind") != "formal_f8192_frozen_environment_replay" or not demo.get("exact"):
        raise ValueError("replayable demo is not an exact formal F/8192 replay")
    if demo.get("model_provenance_sha256") != grpo_model_sha:
        raise ValueError("replayable demo does not use the final GRPO model provenance")
    if demo.get("retriever_provenance_sha256") != lrat_retriever_sha:
        raise ValueError("replayable demo does not use the formal LRAT retriever provenance")
    source_predictions = _verify_declared_evidence(
        demo.get("source_predictions"), "replay-demo source predictions"
    )
    final_source = matrix["experiments"][final_key]["source"]
    if _sha256(source_predictions) != final_source.get("predictions_sha256"):
        raise ValueError("replayable demo is not sourced from the formal F/8192 predictions")
    selected_identity = demo.get("selected_identity", {})
    if not all(str(selected_identity.get(field, "")) for field in ("dataset", "id")):
        raise ValueError("replayable demo has no selected formal identity")
    artifact_paths = {}
    for name in ("trajectory", "report"):
        artifact_paths[name] = _verify_declared_evidence(
            demo.get("artifacts", {}).get(name), f"replay-demo {name}"
        )
    trajectory = _load(artifact_paths["trajectory"])
    source_matches = [
        record
        for record in _load_jsonl(source_predictions)
        if str(record.get("dataset", "")) == str(selected_identity["dataset"])
        and str(record.get("id", "")) == str(selected_identity["id"])
    ]
    if len(source_matches) != 1 or trajectory_from_prediction(source_matches[0]) != trajectory:
        raise ValueError("replay trajectory does not match its selected formal prediction")
    replay_report = replay_frozen_trajectory(trajectory)
    if not replay_report.get("exact") or _load(artifact_paths["report"]) != replay_report:
        raise ValueError("replay-demo report is not an exact fresh replay")

    cost_evidence = {}
    for name, raw_path in sorted(cost_metric_paths.items()):
        path = Path(raw_path)
        metrics = _load(path)
        overall = metrics.get("overall", metrics)
        if int(overall.get("examples", 0)) <= 0:
            raise ValueError(f"cost metric {name} has no examples")
        for field in ("avg_input_tokens", "avg_generated_tokens", "avg_total_tokens"):
            if not isinstance(overall.get(field), (int, float)):
                raise ValueError(f"cost metric {name} is missing {field}")
        cost_evidence[name] = {**_evidence(path), "metrics": overall}

    if set(dataset_manifest_paths) != set(REQUIRED_DATASET_ROWS):
        raise ValueError(
            "dataset manifests must be exactly: " + ", ".join(sorted(REQUIRED_DATASET_ROWS))
        )
    dataset_evidence = {}
    for name, expected_rows in REQUIRED_DATASET_ROWS.items():
        manifest_path = Path(dataset_manifest_paths[name])
        manifest = _load(manifest_path)
        output = manifest.get("output", {})
        prediction_path = Path(output.get("path", ""))
        if int(output.get("rows", -1)) != expected_rows:
            raise ValueError(f"dataset {name} must contain {expected_rows} predictions")
        if not prediction_path.is_file() or _sha256(prediction_path) != output.get("sha256"):
            raise ValueError(f"dataset {name} prediction evidence does not match manifest")
        metrics = manifest.get("metrics", {})
        metrics_path = Path(metrics.get("path", ""))
        if not metrics_path.is_file() or _sha256(metrics_path) != metrics.get("sha256"):
            raise ValueError(f"dataset {name} metric evidence does not match manifest")
        prompts = manifest.get("prompts", {})
        prompts_path = Path(prompts.get("path", ""))
        if not prompts_path.is_file() or _sha256(prompts_path) != prompts.get("sha256"):
            raise ValueError(f"dataset {name} prompt evidence does not match manifest")
        provenance = manifest.get("expected", {}).get("retriever_provenance_sha256")
        if not isinstance(provenance, str) or len(provenance) != 64:
            raise ValueError(f"dataset {name} has no retriever provenance")
        model_provenance = manifest.get("expected", {}).get("model_provenance_sha256")
        if model_provenance != grpo_model_sha:
            raise ValueError(f"dataset {name} does not use the final GRPO model provenance")
        dataset_evidence[name] = {
            "manifest": _evidence(manifest_path),
            "predictions": _evidence(prediction_path),
            "metrics": _evidence(metrics_path),
            "prompts": _evidence(prompts_path),
            "rows": expected_rows,
            "retriever_provenance_sha256": provenance,
            "model_provenance_sha256": model_provenance,
        }

    artifact_evidence = {
        "A-F results": _evidence(matrix_path),
        "R0-R4 retriever comparison": _evidence(retriever_path),
        "reward audit": _evidence(audit_path),
        "accuracy-token cost curve": _evidence(curve),
        "failure cases": _evidence(failures),
        "replayable demo": _evidence(demo_path),
    }
    external_tokens = external.get("total_token_reduction_fraction")
    results = {
        "schema_version": 1,
        "retriever_recall_at_20_gain": float(gain),
        "external_state_token_reduction": (
            float(external_tokens)
            if external.get("quality_non_decreasing") and external_tokens is not None
            else -1.0
        ),
        "external_state_f1_gain": float(external["answer_f1_gain_percentage_points"]),
        "grpo_f1_gain": float(grpo["answer_f1_gain_percentage_points"]),
        "grpo_quality_non_decreasing": bool(grpo["quality_non_decreasing"]),
        "grpo_search_reduction": (
            float(grpo["search_reduction_fraction"])
            if grpo.get("search_reduction_fraction") is not None
            else -1.0
        ),
        "citation_validity": float(final_metrics["citation_validity_micro"]),
        "citation_precision": float(final_metrics["citation_precision_micro"]),
        "cost_records": sorted(cost_evidence),
        "dataset_records": sorted(dataset_evidence),
        "artifacts": sorted(artifact_evidence),
        "selection_policy": {
            "retriever": gain_key,
            "external_state": f"{external_target}-{external_baseline}",
            "grpo": f"{grpo_target}-{grpo_baseline}",
            "comparison_budget": comparison_budget,
        },
        "evidence": {
            "artifacts": artifact_evidence,
            "cost_records": cost_evidence,
            "dataset_records": dataset_evidence,
            "external_state_comparison": external,
            "grpo_comparison": grpo,
        },
    }
    report = audit_acceptance(results, thresholds)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = destination.with_suffix(destination.suffix + ".acceptance.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return results, report
