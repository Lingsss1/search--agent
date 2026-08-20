#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.contamination import find_contamination, write_report
from openstatesearch.data.corpus import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove exact/near-duplicate train questions against tests"
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()
    train = [str(record["question"]) for record in read_jsonl(args.train)]
    tests: list[str] = []
    source_counts: dict[str, int] = {}
    for path in args.test:
        records = list(read_jsonl(path))
        source_counts[path] = len(records)
        tests.extend(str(record["question"]) for record in records)
    matches = find_contamination(train, tests, args.threshold)
    write_report(args.report, matches, source_counts)
    print(json.dumps({"train": len(train), "tests": len(tests), "removed": len(matches)}))


if __name__ == "__main__":
    main()
