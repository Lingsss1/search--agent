#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.phase_b import select_phase_b_hard


def _read(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"non-object record at {path}:{line_number}")
                records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the frozen GRPO Phase-B hard set")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--phase-a",
        type=Path,
        default=Path("data/processed/full_context_seed36/rl_train.jsonl"),
    )
    parser.add_argument("--per-dataset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    selected, manifest = select_phase_b_hard(
        _read([args.pool]),
        _read(args.trajectories),
        per_dataset=args.per_dataset,
        seed=args.seed,
    )
    selected_ids = {
        (str(record.get("dataset", "")), str(record.get("id", ""))) for record in selected
    }
    phase_a_ids = {
        (str(record.get("dataset", "")), str(record.get("id", "")))
        for record in _read([args.phase_a])
    }
    overlap = sorted(selected_ids & phase_a_ids)
    manifest["phase_a_identity_overlap"] = len(overlap)
    if overlap:
        raise ValueError(f"Phase-B hard set overlaps Phase A: {overlap[:3]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in selected
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest_path = args.manifest or Path(f"{args.output}.manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
