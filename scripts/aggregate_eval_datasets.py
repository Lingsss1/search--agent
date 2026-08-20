#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.dataset_aggregate import aggregate_dataset_runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate formal dataset predictions and recompute weighted metrics"
    )
    parser.add_argument("--input", action="append", default=[], metavar="NAME=MANIFEST")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    manifests = {}
    for value in args.input:
        if "=" not in value:
            parser.error("--input must use NAME=MANIFEST")
        name, path = value.split("=", 1)
        if not name or not path or name in manifests:
            parser.error("input names/paths must be non-empty and unique")
        manifests[name] = path
    manifest = aggregate_dataset_runs(
        manifests=manifests,
        output_path=args.output,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
