#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.full_reward_audit import build_full_reward_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Link a GRPO reward audit to full AReaL rollout trajectories"
    )
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--reward-audit", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--model-provenance-manifest")
    parser.add_argument("--checkpoint-manifest")
    args = parser.parse_args()
    summary = build_full_reward_audit(
        rollout_root=args.rollout_root,
        reward_audit_path=args.reward_audit,
        prompts_path=args.prompts,
        output_path=args.output,
        summary_path=args.summary,
        version=args.version,
        sample_size=args.sample_size,
        seed=args.seed,
        model_provenance_path=args.model_provenance_manifest,
        checkpoint_manifest_path=args.checkpoint_manifest,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
