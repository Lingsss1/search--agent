#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.retriever_benchmark import evaluate_retriever
from openstatesearch.retriever.http import HttpRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a running frozen retriever service")
    parser.add_argument("--url", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="R4")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    metrics = evaluate_retriever(
        HttpRetriever(args.url, timeout=args.timeout), list(read_jsonl(args.queries))
    )
    result = {args.name: metrics}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
