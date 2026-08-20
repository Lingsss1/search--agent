#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def repeated_xor(value: str, key: bytes) -> str:
    encrypted = base64.b64decode(value)
    repeated = key * (len(encrypted) // len(key)) + key[: len(encrypted) % len(key)]
    return bytes(left ^ right for left, right in zip(encrypted, repeated)).decode()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return {"path": str(path), "rows": len(records), "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Decrypt Chinese zero-shot tests locally")
    parser.add_argument("--xbench", default="data/raw/xbench-DeepSearch/DeepSearch.csv")
    parser.add_argument("--browsecomp-zh", default="data/raw/BrowseComp-ZH/test.parquet")
    parser.add_argument("--output-dir", default="data/processed/chinese_eval")
    args = parser.parse_args()

    xbench: list[dict[str, object]] = []
    with Path(args.xbench).open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = row["canary"].encode()
            question = repeated_xor(row["prompt"], key)
            answer = repeated_xor(row["answer"], key)
            xbench.append(
                {
                    "id": str(row["id"]),
                    "dataset": "xbench_deepsearch",
                    "question": question,
                    "answer": answer,
                    "answers": [answer],
                    "messages": [{"role": "user", "content": question}],
                    "constraints": [],
                }
            )

    browsecomp: list[dict[str, object]] = []
    for index, row in enumerate(pq.read_table(args.browsecomp_zh).to_pylist()):
        key = hashlib.sha256(str(row["canary"]).encode()).digest()
        question = repeated_xor(str(row["Question"]), key)
        answer = repeated_xor(str(row["Answer"]), key)
        browsecomp.append(
            {
                "id": str(index),
                "dataset": "browsecomp_zh",
                "topic": repeated_xor(str(row["Topic"]), key),
                "question": question,
                "answer": answer,
                "answers": [answer],
                "messages": [{"role": "user", "content": question}],
                "constraints": [],
            }
        )

    if len(xbench) != 100 or len(browsecomp) != 289:
        raise ValueError(f"unexpected Chinese counts: {len(xbench)}, {len(browsecomp)}")
    output = Path(args.output_dir)
    manifest = {
        "schema_version": 1,
        "plaintext_policy": "local only; do not publish benchmark fields",
        "xbench_revision": "436bbed79aef5b19c857047650ab528be33c6680",
        "xbench_decrypt_revision": "17c5621925ff8bc67a113b89ac41fb798b044264",
        "browsecomp_zh_revision": "15b7bffc8af684c9b012843fb4f6353838ea3357",
        "outputs": [
            write_jsonl(output / "xbench_deepsearch.jsonl", xbench),
            write_jsonl(output / "browsecomp_zh.jsonl", browsecomp),
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
