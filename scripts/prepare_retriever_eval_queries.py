#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import read_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_retriever_queries(
    prompts_path: str | Path,
    output_path: str | Path,
    relevance_field: str = "evidence_doc_ids",
) -> dict[str, Any]:
    prompts_path = Path(prompts_path)
    output_path = Path(output_path)
    if prompts_path.resolve() == output_path.resolve():
        raise ValueError("prompts and output paths must differ")
    identities: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for prompt in read_jsonl(prompts_path):
        dataset = str(prompt.get("dataset", ""))
        identity = str(prompt.get("id", ""))
        question = str(prompt.get("question", "")).strip()
        if not dataset or not identity or not question:
            raise ValueError("every prompt requires non-empty dataset, id, and question")
        key = (dataset, identity)
        if key in identities:
            raise ValueError(f"duplicate prompt identity: {key}")
        identities.add(key)
        raw_relevant = prompt.get(relevance_field)
        if not isinstance(raw_relevant, list) or not raw_relevant:
            raise ValueError(f"prompt {key} has no relevance in {relevance_field}")
        relevant = sorted({str(value) for value in raw_relevant if str(value)})
        if not relevant:
            raise ValueError(f"prompt {key} has empty normalized relevance")
        rows.append(
            {
                "dataset": dataset,
                "id": identity,
                "query": question,
                "relevant_doc_ids": relevant,
            }
        )
    if not rows:
        raise ValueError("prompts file is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
    manifest = {
        "schema_version": 1,
        "source": str(prompts_path),
        "source_sha256": _sha256(prompts_path),
        "output": str(output_path),
        "output_sha256": digest.hexdigest(),
        "rows": len(rows),
        "relevance_field": relevance_field,
        "datasets": sorted({row["dataset"] for row in rows}),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create frozen retriever queries/qrels from prepared evaluation prompts"
    )
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--relevance-field", default="evidence_doc_ids")
    args = parser.parse_args()
    manifest = prepare_retriever_queries(
        args.prompts, args.output, relevance_field=args.relevance_field
    )
    # The manifest contains hashes/counts only; decrypted benchmark text remains local.
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
