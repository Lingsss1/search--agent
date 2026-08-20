#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream the released LRAT JSONL into compact Parquet"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--batch-rows", type=int, default=512)
    args = parser.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required (it is installed with datasets)") from exc

    source = Path(args.input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    schema = pa.schema(
        [
            ("record_index", pa.int64()),
            ("query", pa.string()),
            ("positive", pa.string()),
            ("negative", pa.string()),
            ("positive_id", pa.string()),
            ("negative_id", pa.string()),
            ("reweight_rate", pa.float32()),
            ("reasoning_len", pa.int32()),
        ]
    )
    writer = pq.ParquetWriter(partial, schema, compression="zstd")
    batch: list[dict[str, object]] = []
    kept = skipped = 0
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for index, raw in enumerate(handle):
                digest.update(raw)
                value = json.loads(raw)
                positives = value.get("pos") or []
                negatives = value.get("neg") or []
                if not value.get("satisfied", True) or not positives or not negatives:
                    skipped += 1
                    continue
                # Stable selection avoids retaining hundreds of long negative passages
                # while still drawing from each example's released hard-negative set.
                selector = hashlib.sha256(f"{args.seed}:{index}".encode()).digest()
                negative_index = int.from_bytes(selector[:8], "big") % len(negatives)
                positive_ids = value.get("pos_id") or [""]
                negative_ids = value.get("neg_id") or []
                batch.append(
                    {
                        "record_index": index,
                        "query": str(value["query"]),
                        "positive": str(positives[0]),
                        "negative": str(negatives[negative_index]),
                        "positive_id": str(positive_ids[0]),
                        "negative_id": str(negative_ids[negative_index])
                        if negative_index < len(negative_ids)
                        else "",
                        "reweight_rate": float(value.get("reweight_rate", 1.0)),
                        "reasoning_len": int(value.get("reasoning_len", 0)),
                    }
                )
                kept += 1
                if len(batch) >= args.batch_rows:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
                if kept % 10000 == 0:
                    print(json.dumps({"kept": kept, "skipped": skipped}), flush=True)
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    finally:
        writer.close()
    partial.replace(destination)
    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "source_sha256": digest.hexdigest(),
        "output": str(destination.resolve()),
        "rows": kept,
        "skipped": skipped,
        "seed": args.seed,
        "negative_selection": "sha256(seed:record_index) modulo released neg list length",
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
