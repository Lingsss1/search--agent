#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import build_documents, read_jsonl, write_jsonl, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deduplicated frozen RL corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()
    documents = build_documents(read_jsonl(args.input), args.chunk_size, args.overlap)
    sha256 = write_jsonl(args.output, documents)
    write_manifest(args.manifest, args.output, documents, sha256)
    print(f"documents={len(documents)} sha256={sha256}")


if __name__ == "__main__":
    main()
