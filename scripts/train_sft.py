#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.training import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Policy SFT frozen config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--output", default="artifacts/policy_sft_adapter")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    errors = validate_config(load_config(args.config), "sft")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print("SFT config valid: 8K BF16 FSDP LoRA, assistant-only loss, fixed gate.")
    if not args.validate_only:
        if not args.data:
            raise SystemExit("--data is required unless --validate-only is used")
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=8",
            "-m",
            "openstatesearch.training.sft_train",
            "--config",
            args.config,
            "--data",
            args.data,
            "--output",
            args.output,
            "--max-steps",
            str(args.max_steps),
        ]
        if args.resume_from:
            command.extend(["--resume-from", args.resume_from])
        raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
