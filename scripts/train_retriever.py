#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.training import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and launch LRAT retriever training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pairs")
    parser.add_argument("--output", default="artifacts/retriever_lrat")
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--per-device-batch", type=int, default=16)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config, "retriever")
    # Pinning is intentionally a hard gate before downloading or training.
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print("Retriever config and frozen revision are valid.")
    if not args.validate_only:
        if not args.pairs:
            raise SystemExit("--pairs is required unless --validate-only is used")
        command = [
            "accelerate",
            "launch",
            "--multi_gpu",
            "--num_processes",
            str(args.num_processes),
            "-m",
            "openstatesearch.training.retriever_train",
            "--config",
            args.config,
            "--pairs",
            args.pairs,
            "--output",
            args.output,
            "--per-device-batch",
            str(args.per_device_batch),
            "--save-every",
            str(args.save_every),
        ]
        if args.max_steps is not None:
            command.extend(["--max-steps", str(args.max_steps)])
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
