#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_in_domain import (  # noqa: E402
    Builder,
    hotpot_example,
    hotpot_rows,
    json_array,
    json_lines,
    musique_example,
    rank,
    wiki_example,
    write_records,
)


def ranked_slice(values, start: int, count: int, seed: int, dataset: str):
    ranked = sorted(
        (
            rank(seed, dataset, "train", str(value.get("id") or value.get("_id") or index)),
            index,
            value,
        )
        for index, value in enumerate(values)
    )
    return [value for _, _, value in ranked[start : start + count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the disjoint Phase-B hard-selection pool")
    parser.add_argument(
        "--output", default="data/processed/full_context_seed36/rl_phase_b_pool.jsonl"
    )
    parser.add_argument("--pool-per-dataset", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    specifications = {
        "hotpotqa": (
            hotpot_rows(Path("data/raw/HotpotQA/hf_dataset"), "train"),
            2500,
            hotpot_example,
        ),
        "2wiki": (
            json_array(Path("data/raw/2WikiMultiHopQA/data/train.json")),
            2500,
            wiki_example,
        ),
        "musique": (
            json_lines(Path("data/raw/MuSiQue-Ans/data/musique_ans_v1.0_train.jsonl")),
            1000,
            musique_example,
        ),
    }
    records = []
    identities: set[tuple[str, str]] = set()
    counts = {}
    builder = Builder()
    for dataset, (values, phase_a_count, convert) in specifications.items():
        selected = ranked_slice(values, phase_a_count, args.pool_per_dataset, args.seed, dataset)
        counts[dataset] = len(selected)
        for value in selected:
            record, _ = convert(builder, value)
            identity = (dataset, str(record["id"]))
            if identity in identities:
                raise ValueError(f"duplicate Phase-B identity: {identity}")
            identities.add(identity)
            records.append(record)
    digest = write_records(Path(args.output), records)
    phase_a = {
        (str(row["dataset"]), str(row["id"]))
        for row in (
            json.loads(line)
            for line in Path("data/processed/full_context_seed36/rl_train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    }
    overlap = sorted(identities & phase_a)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "selection": "next hashes after Phase-A slice; hard filtering occurs after SFT rollout",
        "counts": counts,
        "rows": len(records),
        "phase_a_identity_overlap": len(overlap),
        "sha256": digest,
    }
    Path(f"{args.output}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if overlap:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
