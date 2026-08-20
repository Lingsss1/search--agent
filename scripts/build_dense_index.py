#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.retriever.service import load_corpus
from openstatesearch.retriever.transformer_dense import TransformerDenseRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode and freeze a LRAT dense corpus index")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    documents = load_corpus(args.corpus)
    retriever = TransformerDenseRetriever(
        documents, args.model, batch_size=args.batch_size, dtype=args.dtype
    )
    retriever.save_index(args.output)
    print(f"indexed {len(documents)} documents at {args.output}")


if __name__ == "__main__":
    main()
