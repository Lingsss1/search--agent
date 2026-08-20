from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openstatesearch.eval.runner import evaluate_by_dataset, read_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    return value


def aggregate_dataset_runs(
    *,
    manifests: dict[str, str | Path],
    output_path: str | Path,
    expected_rows: int,
) -> dict[str, Any]:
    if len(manifests) < 2:
        raise ValueError("dataset aggregation requires at least two named manifests")
    if expected_rows < 1:
        raise ValueError("expected_rows must be positive")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    identity_order: list[tuple[str, str]] = []
    source_evidence = []
    configurations: set[tuple[Any, ...]] = set()
    for name, raw_path in sorted(manifests.items()):
        path = Path(raw_path)
        manifest = _load(path)
        output = manifest.get("output", {})
        predictions_path = Path(output.get("path", ""))
        if not predictions_path.is_file() or _sha256(predictions_path) != output.get("sha256"):
            raise ValueError(f"prediction evidence mismatch for {name}")
        values = read_predictions(predictions_path)
        if len(values) != int(output.get("rows", -1)):
            raise ValueError(f"prediction row mismatch for {name}")
        expected = manifest.get("expected", {})
        provenance = expected.get("retriever_provenance_sha256")
        if not isinstance(provenance, str) or len(provenance) != 64:
            raise ValueError(f"source {name} has no retriever provenance")
        model_provenance = expected.get("model_provenance_sha256")
        if not isinstance(model_provenance, str) or len(model_provenance) != 64:
            raise ValueError(f"source {name} has no model provenance")
        for record in values:
            identity = (str(record.get("dataset", "")), str(record.get("id", "")))
            if not all(identity) or identity in identities:
                raise ValueError(f"duplicate or empty aggregate identity: {identity}")
            identities.add(identity)
            identity_order.append(identity)
            if record.get("retriever_provenance_sha256") != provenance:
                raise ValueError(f"source {name} record has wrong retriever provenance")
            if record.get("model_provenance_sha256") != model_provenance:
                raise ValueError(f"source {name} record has wrong model provenance")
            configurations.add(
                (
                    record.get("model"),
                    record.get("model_provenance_sha256"),
                    record.get("state_mode"),
                    record.get("retriever"),
                    record.get("memory_token_budget"),
                    record.get("generation_token_budget"),
                    record.get("retriever_provenance_sha256"),
                )
            )
        records.extend(values)
        source_evidence.append(
            {
                "name": name,
                "manifest": str(path),
                "manifest_sha256": _sha256(path),
                "predictions": str(predictions_path),
                "predictions_sha256": _sha256(predictions_path),
                "rows": len(values),
            }
        )
    if len(records) != expected_rows:
        raise ValueError(f"aggregate has {len(records)} rows; expected {expected_rows}")
    if len(configurations) != 1:
        raise ValueError("aggregate sources do not use one identical evaluation configuration")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    destination.write_text(payload, encoding="utf-8")
    metrics_path = destination.with_suffix(destination.suffix + ".eval_metrics.json")
    metrics_path.write_text(
        json.dumps(evaluate_by_dataset(records), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    identity_sha = hashlib.sha256(
        "\n".join(f"{dataset}\t{identity}" for dataset, identity in identity_order).encode()
    ).hexdigest()
    configuration = next(iter(configurations))
    manifest = {
        "schema_version": 1,
        "sources": source_evidence,
        "output": {
            "path": str(destination),
            "rows": len(records),
            "sha256": _sha256(destination),
            "identity_set_sha256": identity_sha,
        },
        "metrics": {"path": str(metrics_path), "sha256": _sha256(metrics_path)},
        "configuration": {
            "model": configuration[0],
            "model_provenance_sha256": configuration[1],
            "state_mode": configuration[2],
            "retriever": configuration[3],
            "memory_token_budget": configuration[4],
            "generation_token_budget": configuration[5],
            "retriever_provenance_sha256": configuration[6],
        },
    }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
