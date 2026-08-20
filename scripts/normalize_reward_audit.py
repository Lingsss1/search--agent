#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.reward_audit import normalize_reward_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an exact, deterministic per-step GRPO reward audit sample"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    summary = normalize_reward_audit(
        args.source,
        args.output,
        args.summary,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
