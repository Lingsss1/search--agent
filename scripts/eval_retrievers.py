#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.retriever_benchmark import evaluate_retriever
from openstatesearch.retriever import HybridRetriever, TransformerDenseRetriever
from openstatesearch.retriever.bm25 import BM25Retriever
from openstatesearch.retriever.service import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mandatory R0-R4 retriever comparison")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True, help="JSONL: query, relevant_doc_ids")
    parser.add_argument("--base-model", required=True, help="R1 checkpoint/path")
    parser.add_argument("--official-model", required=True, help="R2 official LRAT checkpoint/path")
    parser.add_argument(
        "--trained-model", required=True, help="R3 self-trained LRAT checkpoint/path"
    )
    parser.add_argument("--base-index", required=True)
    parser.add_argument("--official-index", required=True)
    parser.add_argument("--trained-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--only",
        default="R0,R1,R2,R3,R4",
        help="Comma-separated variants; useful for checkpointed full-corpus evaluation",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    documents = load_corpus(args.corpus)
    examples = list(read_jsonl(args.queries))
    wanted = {value.strip() for value in args.only.split(",") if value.strip()}
    unknown = wanted - {"R0", "R1", "R2", "R3", "R4"}
    if unknown:
        raise ValueError(f"unknown retriever variants: {sorted(unknown)}")
    output = Path(args.output)
    results = json.loads(output.read_text()) if args.resume and output.exists() else {}

    def checkpoint() -> None:
        rendered = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(rendered, end="", flush=True)

    if "R0" in wanted and "R0" not in results:
        results["R0"] = evaluate_retriever(BM25Retriever(documents), examples)
        checkpoint()
    specifications = [
        ("R1", args.base_model, args.base_index),
        ("R2", args.official_model, args.official_index),
        ("R3", args.trained_model, args.trained_index),
    ]
    for name, model, index in specifications:
        needs_dense = name in wanted or (name == "R3" and "R4" in wanted)
        if not needs_dense:
            continue
        dense = TransformerDenseRetriever(documents, model, index)
        if name in wanted and name not in results:
            results[name] = evaluate_retriever(dense, examples)
            checkpoint()
        if name == "R3" and "R4" in wanted and "R4" not in results:
            results["R4"] = evaluate_retriever(HybridRetriever(documents, dense=dense), examples)
            checkpoint()
        del dense
        gc.collect()
    checkpoint()


if __name__ == "__main__":
    main()
