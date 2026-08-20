from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openstatesearch.eval.runner import evaluate_by_dataset, read_predictions
from openstatesearch.rewards.metrics import answer_f1


EXPERIMENTS = ("A", "B", "C", "D", "E", "F")
BUDGETS = (4096, 8192)
EXPECTED = {
    "A": ("transcript", "base_hybrid"),
    "B": ("external_state", "base_hybrid"),
    "C": ("transcript", "lrat_hybrid"),
    "D": ("external_state", "lrat_hybrid"),
    "E": ("transcript", "lrat_hybrid"),
    "F": ("external_state", "lrat_hybrid"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identity_digest(predictions: list[dict[str, Any]]) -> tuple[list[tuple[str, str]], str]:
    identities = [
        (str(record.get("dataset", "")), str(record.get("id", ""))) for record in predictions
    ]
    if any(not all(identity) for identity in identities):
        raise ValueError("matrix predictions require non-empty dataset/id identities")
    if len(set(identities)) != len(identities):
        raise ValueError("matrix predictions contain duplicate identities")
    digest = hashlib.sha256(
        "\n".join(f"{dataset}\t{identity}" for dataset, identity in identities).encode()
    ).hexdigest()
    return identities, digest


def _load_run(root: Path, experiment: str, tag: str, budget: int) -> dict[str, Any]:
    run_root = root / experiment / tag / f"budget_{budget}"
    predictions_path = run_root / "predictions.jsonl"
    metrics_path = run_root / "predictions.jsonl.eval_metrics.json"
    manifest_path = run_root / "predictions.jsonl.manifest.json"
    for path in (predictions_path, metrics_path, manifest_path):
        if not path.is_file():
            raise ValueError(f"missing formal matrix artifact: {path}")
    predictions = read_predictions(predictions_path)
    identities, identity_sha256 = _identity_digest(predictions)
    expected_state, expected_retriever = EXPECTED[experiment]
    for record in predictions:
        actual = (
            record.get("experiment"),
            record.get("state_mode"),
            record.get("retriever"),
            record.get("memory_token_budget"),
        )
        expected = (experiment, expected_state, expected_retriever, budget)
        if actual != expected:
            raise ValueError(f"matrix record identity mismatch: {actual!r} != {expected!r}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(metrics["overall"]["examples"]) != len(predictions):
        raise ValueError(f"prediction/metric row mismatch in {run_root}")
    recomputed_metrics = evaluate_by_dataset(predictions)
    if _canonical(metrics) != _canonical(recomputed_metrics):
        raise ValueError(f"stored metrics do not match predictions: {run_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output", {}).get("sha256") != _sha256(predictions_path):
        raise ValueError(f"prediction SHA does not match manifest: {predictions_path}")
    if int(manifest.get("output", {}).get("rows", -1)) != len(predictions):
        raise ValueError(f"prediction row count does not match manifest: {predictions_path}")
    if manifest.get("output", {}).get("identity_sha256") != identity_sha256:
        raise ValueError(f"prediction identities do not match manifest: {predictions_path}")
    models = {str(record.get("model", "")) for record in predictions}
    if len(models) != 1 or not next(iter(models)):
        raise ValueError(f"matrix run must contain exactly one non-empty model: {run_root}")
    retriever_provenance = {
        str(record.get("retriever_provenance_sha256", "")) for record in predictions
    }
    if len(retriever_provenance) != 1 or len(next(iter(retriever_provenance))) != 64:
        raise ValueError(f"matrix run must contain exactly one retriever provenance: {run_root}")
    provenance_sha = retriever_provenance.pop()
    if manifest.get("expected", {}).get("retriever_provenance_sha256") != provenance_sha:
        raise ValueError(f"retriever provenance does not match manifest: {run_root}")
    model_provenance = {str(record.get("model_provenance_sha256", "")) for record in predictions}
    if len(model_provenance) != 1 or len(next(iter(model_provenance))) != 64:
        raise ValueError(f"matrix run must contain exactly one model provenance: {run_root}")
    model_provenance_sha = model_provenance.pop()
    if manifest.get("expected", {}).get("model_provenance_sha256") != model_provenance_sha:
        raise ValueError(f"model provenance does not match manifest: {run_root}")
    return {
        "predictions": predictions,
        "identities": identities,
        "identity_sha256": identity_sha256,
        "model": models.pop(),
        "retriever_provenance_sha256": provenance_sha,
        "model_provenance_sha256": model_provenance_sha,
        "metrics": metrics,
        "source": {
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256(predictions_path),
            "metrics": str(metrics_path),
            "metrics_sha256": _sha256(metrics_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
    }


def _comparison(
    runs: dict[tuple[str, int], dict[str, Any]],
    target: str,
    baseline: str,
    budget: int,
) -> dict[str, Any]:
    target_metrics = runs[(target, budget)]["metrics"]["overall"]
    baseline_metrics = runs[(baseline, budget)]["metrics"]["overall"]
    target_f1 = float(target_metrics["answer_f1"])
    baseline_f1 = float(baseline_metrics["answer_f1"])
    target_tokens = float(target_metrics["avg_total_tokens"])
    baseline_tokens = float(baseline_metrics["avg_total_tokens"])
    target_search = float(target_metrics["avg_search"])
    baseline_search = float(baseline_metrics["avg_search"])
    return {
        "target": target,
        "baseline": baseline,
        "memory_token_budget": budget,
        "answer_f1_gain_percentage_points": 100.0 * (target_f1 - baseline_f1),
        "total_token_reduction_fraction": (
            1.0 - target_tokens / baseline_tokens if baseline_tokens else None
        ),
        "search_reduction_fraction": (
            1.0 - target_search / baseline_search if baseline_search else None
        ),
        "quality_non_decreasing": target_f1 >= baseline_f1,
        "target_metrics": target_metrics,
        "baseline_metrics": baseline_metrics,
    }


def _failure_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in records:
        references = record.get("answers", [])
        if isinstance(references, str):
            references = [references]
        f1 = answer_f1(str(record.get("prediction", "")), references)
        evidence = {
            (str(item["doc_id"]), int(sent_id))
            for item in record.get("evidence", [])
            for sent_id in item.get("sent_ids", [])
        }
        gold = {
            (str(item["doc_id"]), int(sent_id))
            for item in record.get("gold_evidence", [])
            for sent_id in item.get("sent_ids", [])
        }
        categories = []
        if f1 < 0.8:
            categories.append("answer_f1_below_0.8")
        if gold and len(evidence & gold) < len(gold):
            categories.append("incomplete_evidence")
        if float(record.get("search_count", 0)) > 3:
            categories.append("search_count_gt_3")
        if not record.get("trajectory_completed", False):
            categories.append("incomplete_trajectory")
        if categories:
            failures.append({"categories": categories, "answer_f1": f1, "record": record})
    return sorted(
        failures,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{_canonical(item)}".encode()).hexdigest(),
            _canonical(item),
        ),
    )[:limit]


def summarize_eval_matrix(
    *,
    matrix_root: str | Path,
    tag: str,
    output_dir: str | Path,
    failure_limit: int = 50,
    seed: int = 36,
) -> dict[str, Any]:
    if failure_limit <= 0:
        raise ValueError("failure_limit must be positive")
    root = Path(matrix_root)
    runs = {
        (experiment, budget): _load_run(root, experiment, tag, budget)
        for experiment in EXPERIMENTS
        for budget in BUDGETS
    }
    identity_digests = {run["identity_sha256"] for run in runs.values()}
    if len(identity_digests) != 1:
        raise ValueError("all 12 matrix runs must use identical ordered prompt identities")
    sft_models = {
        runs[(experiment, budget)]["model"] for experiment in "ABCD" for budget in BUDGETS
    }
    grpo_models = {runs[(experiment, budget)]["model"] for experiment in "EF" for budget in BUDGETS}
    if len(sft_models) != 1:
        raise ValueError("A-D must use one identical SFT model")
    if len(grpo_models) != 1:
        raise ValueError("E-F must use one identical GRPO model")
    if sft_models == grpo_models:
        raise ValueError("E-F must use a GRPO model distinct from the A-D SFT model")
    sft_model_provenance = {
        runs[(experiment, budget)]["model_provenance_sha256"]
        for experiment in "ABCD"
        for budget in BUDGETS
    }
    grpo_model_provenance = {
        runs[(experiment, budget)]["model_provenance_sha256"]
        for experiment in "EF"
        for budget in BUDGETS
    }
    if len(sft_model_provenance) != 1 or len(grpo_model_provenance) != 1:
        raise ValueError("matrix model provenance must be stable within SFT and GRPO groups")
    if sft_model_provenance == grpo_model_provenance:
        raise ValueError("SFT and GRPO model provenance must be distinct")
    base_retrievers = {
        runs[(experiment, budget)]["retriever_provenance_sha256"]
        for experiment in "AB"
        for budget in BUDGETS
    }
    lrat_retrievers = {
        runs[(experiment, budget)]["retriever_provenance_sha256"]
        for experiment in "CDEF"
        for budget in BUDGETS
    }
    if len(base_retrievers) != 1:
        raise ValueError("A-B must use one identical base retriever provenance")
    if len(lrat_retrievers) != 1:
        raise ValueError("C-F must use one identical LRAT retriever provenance")
    if base_retrievers == lrat_retrievers:
        raise ValueError("base and LRAT retriever provenance must be distinct")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cost_curve = [
        {
            "experiment": experiment,
            "memory_token_budget": budget,
            **runs[(experiment, budget)]["metrics"]["overall"],
        }
        for experiment in EXPERIMENTS
        for budget in BUDGETS
    ]
    state_pairs = (("B", "A"), ("D", "C"), ("F", "E"))
    grpo_pairs = (("E", "C"), ("F", "D"))
    comparisons = {
        "external_state": [
            _comparison(runs, target, baseline, budget)
            for target, baseline in state_pairs
            for budget in BUDGETS
        ],
        "grpo": [
            _comparison(runs, target, baseline, budget)
            for target, baseline in grpo_pairs
            for budget in BUDGETS
        ],
    }
    failures = _failure_records(runs[("F", 8192)]["predictions"], failure_limit, seed)
    failures_path = output_root / "failure_cases.jsonl"
    failures_payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in failures
    )
    failures_path.write_text(failures_payload, encoding="utf-8")
    curve_path = output_root / "accuracy_token_cost_curve.json"
    curve_path.write_text(
        json.dumps(cost_curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "schema_version": 1,
        "tag": tag,
        "prompt_identities": {
            "rows": len(next(iter(runs.values()))["identities"]),
            "sha256": next(iter(identity_digests)),
        },
        "models": {
            "sft": next(iter(sft_models)),
            "sft_provenance_sha256": next(iter(sft_model_provenance)),
            "grpo": next(iter(grpo_models)),
            "grpo_provenance_sha256": next(iter(grpo_model_provenance)),
        },
        "retrievers": {
            "base_hybrid_sha256": next(iter(base_retrievers)),
            "lrat_hybrid_sha256": next(iter(lrat_retrievers)),
        },
        "experiments": {
            f"{experiment}_{budget}": {
                "metrics": runs[(experiment, budget)]["metrics"],
                "source": runs[(experiment, budget)]["source"],
            }
            for experiment in EXPERIMENTS
            for budget in BUDGETS
        },
        "comparisons": comparisons,
        "accuracy_token_cost_curve": {
            "path": str(curve_path),
            "sha256": _sha256(curve_path),
            "points": len(cost_curve),
        },
        "failure_cases": {
            "path": str(failures_path),
            "sha256": _sha256(failures_path),
            "rows": len(failures),
            "source_experiment": "F",
            "source_memory_token_budget": 8192,
        },
    }
    summary_path = output_root / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
