#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VARIANTS = ("R0", "R1", "R2", "R3", "R4")
METRICS = ("recall_at_5", "recall_at_20", "recall_at_100", "ndcg_at_10")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_retriever_results(inputs: list[str | Path], output: str | Path) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    sources: list[dict[str, Any]] = []
    results: dict[str, dict[str, float | int]] = {}
    for raw_path in inputs:
        path = Path(raw_path)
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"retriever result must be an object: {path}")
        sources.append({"path": str(path), "sha256": _sha256(path)})
        for variant, metrics in value.items():
            if variant not in VARIANTS:
                raise ValueError(f"unknown retriever variant: {variant}")
            if not isinstance(metrics, dict):
                raise ValueError(f"metrics for {variant} must be an object")
            required = {"examples", *METRICS}
            missing = required - set(metrics)
            if missing:
                raise ValueError(f"metrics for {variant} missing {sorted(missing)}")
            normalized = {name: metrics[name] for name in ("examples", *METRICS)}
            if variant in results and results[variant] != normalized:
                raise ValueError(f"conflicting results for {variant}")
            results[variant] = normalized
    missing_variants = sorted(set(VARIANTS) - set(results))
    if missing_variants:
        raise ValueError(f"missing retriever variants: {missing_variants}")
    example_counts = {int(results[name]["examples"]) for name in VARIANTS}
    if len(example_counts) != 1:
        raise ValueError(f"retriever variants use different example counts: {example_counts}")

    gains: dict[str, dict[str, float]] = {}
    for target in VARIANTS:
        for baseline in VARIANTS:
            if target == baseline:
                continue
            gains[f"{target}-{baseline}"] = {
                metric: float(results[target][metric]) - float(results[baseline][metric])
                for metric in METRICS
            }
    payload = {
        "schema_version": 1,
        "examples": example_counts.pop(),
        "results": {name: results[name] for name in VARIANTS},
        "pairwise_absolute_gains": gains,
        "gain_selection": None,
        "gain_selection_note": (
            "No acceptance baseline is inferred; select target/baseline only from the "
            "authoritative project specification."
        ),
        "sources": sources,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "output": str(destination),
        "sha256": _sha256(destination),
        "variants": list(VARIANTS),
        "examples": payload["examples"],
        "sources": sources,
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge R0-R4 results and report every pairwise absolute gain"
    )
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = merge_retriever_results(args.inputs, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
