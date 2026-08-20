#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.replay import replay_frozen_trajectory, trajectory_from_prediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _candidate_rank(record: dict[str, Any], seed: int) -> tuple[str, str, str]:
    identity = f"{record.get('dataset', '')}\t{record.get('id', '')}"
    return (
        hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest(),
        identity,
        str(record.get("question", "")),
    )


def build_formal_replay_artifact(
    predictions_path: Path, artifact_dir: Path, seed: int = 36
) -> dict[str, Any]:
    records = list(read_jsonl(predictions_path))
    candidates = []
    for record in records:
        try:
            trajectory_from_prediction(record)
        except (KeyError, TypeError, ValueError):
            continue
        if all(
            isinstance(record.get(field), str) and len(record[field]) == 64
            for field in ("model_provenance_sha256", "retriever_provenance_sha256")
        ):
            candidates.append(record)
    if not candidates:
        raise ValueError("formal predictions contain no completed replayable trajectory")
    selected = None
    trajectory = None
    report = None
    failures = []
    for candidate in sorted(candidates, key=lambda record: _candidate_rank(record, seed)):
        candidate_trajectory = trajectory_from_prediction(candidate)
        try:
            candidate_report = replay_frozen_trajectory(candidate_trajectory)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(
                {
                    "identity": candidate_trajectory["source_identity"],
                    "error": str(error),
                }
            )
            continue
        if candidate_report["exact"]:
            selected = candidate
            trajectory = candidate_trajectory
            report = candidate_report
            break
        failures.append(
            {
                "identity": candidate_trajectory["source_identity"],
                "error": "behavioral divergence",
                "divergences": candidate_report["divergences"],
            }
        )
    if selected is None or trajectory is None or report is None:
        raise ValueError(
            "formal predictions contain no trajectory that replays exactly under the "
            f"current harness; examined={len(candidates)} first_failures={failures[:3]}"
        )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = artifact_dir / "trajectory.json"
    report_path = artifact_dir / "replay_report.json"
    trajectory_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "kind": "formal_f8192_frozen_environment_replay",
        "exact": True,
        "selection_seed": seed,
        "candidate_count": len(candidates),
        "skipped_before_exact": len(failures),
        "selected_identity": trajectory["source_identity"],
        "model_provenance_sha256": trajectory["model_provenance_sha256"],
        "retriever_provenance_sha256": trajectory["retriever_provenance_sha256"],
        "run_config_sha256": trajectory["run_config_sha256"],
        "source_predictions": _evidence(predictions_path),
        "artifacts": {
            "trajectory": _evidence(trajectory_path),
            "report": _evidence(report_path),
        },
        "replay_command": f"python scripts/replay_formal_demo.py --trajectory {trajectory_path}",
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or replay a final F/8192 frozen-environment demo"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--predictions", type=Path)
    mode.add_argument("--trajectory", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    if args.predictions:
        if args.artifact_dir is None:
            parser.error("--artifact-dir is required with --predictions")
        result = build_formal_replay_artifact(args.predictions, args.artifact_dir, args.seed)
    else:
        trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
        result = replay_frozen_trajectory(trajectory)
        if not result["exact"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
