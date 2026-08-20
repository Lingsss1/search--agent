#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.training_log import parse_training_log, summarize_training_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse an AReaL GRPO log into auditable per-step and window metrics"
    )
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument(
        "--through-step",
        type=int,
        help="include completed rows only through this exact checkpoint step",
    )
    args = parser.parse_args()
    rows, metadata = parse_training_log(args.log)
    if args.through_step is not None:
        if args.through_step <= 0:
            parser.error("--through-step must be positive")
        rows = [row for row in rows if int(row["step"]) <= args.through_step]
        if not rows or int(rows[-1]["step"]) != args.through_step:
            parser.error(f"training log does not contain completed step {args.through_step}")
        observed = {int(row["step"]) for row in rows}
        metadata.update(
            {
                "first_completed_step": int(rows[0]["step"]),
                "last_completed_step": int(rows[-1]["step"]),
                "completed_steps": len(rows),
                "missing_steps_in_observed_range": [
                    step
                    for step in range(int(rows[0]["step"]), int(rows[-1]["step"]) + 1)
                    if step not in observed
                ],
                "through_step": args.through_step,
            }
        )
    payload = {
        "schema_version": 1,
        **metadata,
        "window_size": args.window_size,
        "windows": summarize_training_rows(rows, args.window_size),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    output.write_text(rendered, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "output": str(output),
        "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "source": metadata["source"],
        "source_sha256": metadata["source_sha256"],
        "completed_steps": metadata["completed_steps"],
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
