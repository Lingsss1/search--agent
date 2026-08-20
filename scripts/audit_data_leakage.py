#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.normalize import normalize_question


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode()).hexdigest()


def jsonl_questions(path: Path) -> Iterable[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            value = row.get("question") or row.get("prompt")
            if value:
                yield str(value)


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact normalized train/test leakage")
    parser.add_argument("--output", default="artifacts/audits/data_leakage.json")
    args = parser.parse_args()
    train: dict[str, set[str]] = {
        "rl_train": {
            fingerprint(value)
            for value in jsonl_questions(Path("data/processed/full_context_seed36/rl_train.jsonl"))
        },
        "openseeker": set(),
        "openresearcher_all": set(),
    }
    with Path("data/raw/OpenSeeker-v1-Data/openseeker_v1_data.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("question"):
                train["openseeker"].add(fingerprint(str(row["question"])))
    for path in sorted(Path("data/raw/OpenResearcher-Dataset").glob("**/*.parquet")):
        for row in pq.read_table(path, columns=["messages"]).to_pylist():
            first_user = next(
                (
                    message_text(message)
                    for message in row["messages"]
                    if message.get("role") == "user"
                ),
                "",
            )
            if first_user:
                train["openresearcher_all"].add(fingerprint(first_user))

    test_paths = {
        "in_domain_dev": Path("data/processed/full_context_seed36/rl_dev.jsonl"),
        "browsecomp_plus": Path("data/processed/browsecomp_plus/eval_prompts.jsonl"),
        "xbench_deepsearch": Path("data/processed/chinese_eval/xbench_deepsearch.jsonl"),
        "browsecomp_zh": Path("data/processed/chinese_eval/browsecomp_zh.jsonl"),
    }
    tests = {
        name: {fingerprint(value) for value in jsonl_questions(path)}
        for name, path in test_paths.items()
    }
    overlaps = []
    for train_name, train_values in train.items():
        for test_name, test_values in tests.items():
            for digest in sorted(train_values & test_values):
                overlaps.append({"train": train_name, "test": test_name, "question_sha256": digest})
    report = {
        "schema_version": 1,
        "method": "exact match after normalize_question; reports hashes only",
        "train_unique_questions": {name: len(values) for name, values in train.items()},
        "test_unique_questions": {name: len(values) for name, values in tests.items()},
        "overlap_count": len(overlaps),
        "passed": not overlaps,
        "overlaps": overlaps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if overlaps:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
