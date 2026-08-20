#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"evidence file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def archive_adapter(
    *,
    source: Path,
    output: Path,
    base: Path,
    experiment: str,
    trial: str,
    step: int,
    training_log: Path,
    training_curve: Path,
    grpo_config: Path,
    areal_config: Path,
    reward_audit_summary: Path,
    rollout_trend: Path,
) -> dict[str, Any]:
    if source.name != f"weight_update_v{step}":
        raise ValueError(
            f"source directory {source.name!r} does not identify requested step {step}"
        )
    adapter_config = source / "adapter_config.json"
    adapter_weights = source / "adapter_model.safetensors"
    for path in (adapter_config, adapter_weights):
        if not path.is_file():
            raise ValueError(f"GRPO adapter source is missing: {path}")
    if not base.is_dir():
        raise ValueError(f"base model directory is missing: {base}")
    base_provenance = base / "model_provenance.json"
    config = json.loads(adapter_config.read_text(encoding="utf-8"))
    declared_base = Path(str(config.get("base_model_name_or_path", "")))
    if not str(declared_base):
        raise ValueError("adapter_config.json has no base_model_name_or_path")
    declared_matches = str(declared_base) == str(base)
    if declared_base.exists():
        declared_matches = declared_matches or declared_base.resolve() == base.resolve()
    if not declared_matches:
        raise ValueError(f"adapter declares base {declared_base}, expected audited base {base}")

    output.mkdir(parents=True, exist_ok=True)
    log_snapshot = output / "training.log.snapshot"
    source_log_sha = _sha256(training_log)
    if log_snapshot.exists() and _sha256(log_snapshot) != source_log_sha:
        raise ValueError(
            "archived training log is immutable and differs from the current source log"
        )
    if not log_snapshot.exists():
        shutil.copy2(training_log, log_snapshot)

    linked_evidence = {
        "training_log_snapshot": _evidence(log_snapshot),
        "training_curve": _evidence(training_curve),
        "training_curve_manifest": _evidence(
            training_curve.with_suffix(training_curve.suffix + ".manifest.json")
        ),
        "grpo_config": _evidence(grpo_config),
        "areal_config": _evidence(areal_config),
        "base_model_provenance": _evidence(base_provenance),
        "reward_audit_summary": _evidence(reward_audit_summary),
        "rollout_trend": _evidence(rollout_trend),
    }
    reward_audit = json.loads(reward_audit_summary.read_text(encoding="utf-8"))
    trend = json.loads(rollout_trend.read_text(encoding="utf-8"))
    curve = json.loads(training_curve.read_text(encoding="utf-8"))
    if int(reward_audit.get("step", -1)) != step:
        raise ValueError("reward audit does not describe the requested checkpoint step")
    if int(reward_audit.get("sample_size", -1)) != 50:
        raise ValueError("reward audit must contain exactly 50 selected trajectories")
    if int(trend.get("through_version", -1)) != step:
        raise ValueError("rollout trend does not end at the requested checkpoint step")
    if int(curve.get("through_step", -1)) != step:
        raise ValueError("training curve does not end at the requested checkpoint step")

    source_files = []
    output_files = []
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        source_path = source / name
        destination = output / name
        source_sha = _sha256(source_path)
        if destination.exists() and _sha256(destination) != source_sha:
            raise ValueError(f"refusing to overwrite mismatched archived file: {destination}")
        if not destination.exists():
            shutil.copy2(source_path, destination)
        destination_sha = _sha256(destination)
        if destination_sha != source_sha:
            raise ValueError(f"archived adapter SHA mismatch: {destination}")
        source_files.append(
            {"path": str(source_path), "bytes": source_path.stat().st_size, "sha256": source_sha}
        )
        output_files.append(
            {"path": name, "bytes": destination.stat().st_size, "sha256": destination_sha}
        )
    output_files.append(
        {
            "path": log_snapshot.name,
            "bytes": log_snapshot.stat().st_size,
            "sha256": _sha256(log_snapshot),
        }
    )

    manifest = {
        "schema_version": 1,
        "kind": "grpo_lora_checkpoint",
        "experiment": experiment,
        "trial": trial,
        "step": step,
        "source": {"path": str(source), "files": source_files},
        "source_training_log": {
            "path": str(training_log),
            "sha256_at_archive": source_log_sha,
        },
        "base_model": {
            "path": str(base),
            "provenance_sha256": linked_evidence["base_model_provenance"]["sha256"],
        },
        "output": {"path": str(output), "files": output_files},
        "evidence": linked_evidence,
        "audit_metrics": reward_audit["selected_metrics"],
    }
    manifest_path = output / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive one exact AReaL GRPO LoRA version with an evidence chain"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--training-curve", type=Path, required=True)
    parser.add_argument("--grpo-config", type=Path, required=True)
    parser.add_argument("--areal-config", type=Path, required=True)
    parser.add_argument("--reward-audit-summary", type=Path, required=True)
    parser.add_argument("--rollout-trend", type=Path, required=True)
    args = parser.parse_args()
    if args.step <= 0:
        parser.error("--step must be positive")
    manifest = archive_adapter(
        source=args.source,
        output=args.output,
        base=args.base,
        experiment=args.experiment,
        trial=args.trial,
        step=args.step,
        training_log=args.training_log,
        training_curve=args.training_curve,
        grpo_config=args.grpo_config,
        areal_config=args.areal_config,
        reward_audit_summary=args.reward_audit_summary,
        rollout_trend=args.rollout_trend,
    )
    print(
        json.dumps(
            {
                "output": manifest["output"]["path"],
                "step": manifest["step"],
                "adapter_sha256": manifest["output"]["files"][1]["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
