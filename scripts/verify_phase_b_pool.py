#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.phase_b_audit import verify_phase_b_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the complete frozen Phase-B pool and hard-set evidence chain"
    )
    parser.add_argument("--pool", required=True)
    parser.add_argument("--merged", required=True)
    parser.add_argument("--hard", required=True)
    parser.add_argument("--phase-a", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = verify_phase_b_pool(
        pool_path=args.pool,
        merged_path=args.merged,
        hard_path=args.hard,
        phase_a_path=args.phase_a,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
