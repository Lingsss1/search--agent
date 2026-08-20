from __future__ import annotations

from typing import Any


def audit_acceptance(results: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all numeric success criteria without silently accepting absent evidence."""
    checks = {
        "retriever_gain": (
            results.get("retriever_recall_at_20_gain") is not None
            and results["retriever_recall_at_20_gain"]
            >= thresholds["retriever_recall_at_20_absolute_gain"]
        ),
        "external_state": (
            results.get("external_state_token_reduction", -1)
            >= thresholds["external_state_token_reduction_at_same_f1"]
            or results.get("external_state_f1_gain", -1)
            >= thresholds["external_state_f1_gain_at_same_tokens"]
        ),
        "grpo": (
            results.get("grpo_f1_gain", -1) >= thresholds["grpo_f1_gain"]
            or (
                results.get("grpo_quality_non_decreasing") is True
                and results.get("grpo_search_reduction", -1)
                >= thresholds["grpo_search_reduction_if_quality_stable"]
            )
        ),
        "citation_validity": (
            results.get("citation_validity") is not None
            and results["citation_validity"] >= thresholds["citation_validity"]
        ),
        "citation_precision": (
            results.get("citation_precision") is not None
            and results["citation_precision"] >= thresholds["citation_precision"]
        ),
        "cost_records": all(
            name in set(results.get("cost_records", []))
            for name in thresholds["require_cost_records"]
        ),
        "artifacts": all(
            name in set(results.get("artifacts", [])) for name in thresholds["completion_artifacts"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_or_failed": [k for k, v in checks.items() if not v],
    }
