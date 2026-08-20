#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.matrix_summary import summarize_eval_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize formal A-F runs, cost curves, comparisons, and failures"
    )
    parser.add_argument("--matrix-root", default="artifacts/eval/matrix")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--failure-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    result = summarize_eval_matrix(
        matrix_root=args.matrix_root,
        tag=args.tag,
        output_dir=args.output_dir,
        failure_limit=args.failure_limit,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
