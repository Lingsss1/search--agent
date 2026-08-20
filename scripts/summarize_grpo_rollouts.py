#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.grpo_rollout_trend import summarize_phase_a_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize every complete Phase-A rollout version")
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--through-version", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-episodes", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=25)
    args = parser.parse_args()
    result = summarize_phase_a_rollouts(
        rollout_root=args.rollout_root,
        start_version=args.start_version,
        through_version=args.through_version,
        output_path=args.output,
        expected_episodes_per_version=args.expected_episodes,
        window_size=args.window_size,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "start_version": result["start_version"],
                "through_version": result["through_version"],
                "windows": result["windows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
