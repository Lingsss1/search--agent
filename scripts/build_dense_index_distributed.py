#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.retriever.service import load_corpus
from openstatesearch.retriever.transformer_dense import (
    QUERY_INSTRUCTION,
    last_token_pool,
)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed frozen dense corpus encoding")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--keep-shards", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import torch
    import torch.distributed as dist
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("distributed index construction requires CUDA")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)

    documents = load_corpus(args.corpus)
    by_id = {document.doc_id: document for document in documents}
    doc_ids = sorted(by_id)
    if len(by_id) != len(documents):
        raise ValueError("duplicate doc_id in frozen corpus")
    start = len(doc_ids) * rank // world_size
    end = len(doc_ids) * (rank + 1) // world_size
    local_ids = doc_ids[start:end]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(args.model, dtype=torch.bfloat16, local_files_only=True)
        .to(device)
        .eval()
    )
    vectors: list[np.ndarray] = []
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(local_ids), args.batch_size):
            ids = local_ids[offset : offset + args.batch_size]
            texts = [f"{by_id[doc_id].title}\n{by_id[doc_id].text}" for doc_id in ids]
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            embedding = last_token_pool(hidden, batch["attention_mask"])
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
            vectors.append(embedding.to(torch.float16).cpu().numpy())
            completed = min(offset + args.batch_size, len(local_ids))
            if completed % (args.batch_size * 100) == 0 or completed == len(local_ids):
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "completed": completed,
                            "assigned": len(local_ids),
                            "elapsed_seconds": round(time.monotonic() - started, 2),
                        }
                    ),
                    flush=True,
                )
    local_vectors = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype=np.float16)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shard = Path(f"{output}.rank{rank:05d}.npz")
    np.savez(shard, start=np.asarray(start), doc_ids=np.asarray(local_ids), vectors=local_vectors)
    del model, local_vectors, vectors
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        all_ids: list[np.ndarray] = []
        all_vectors: list[np.ndarray] = []
        dimensions = 0
        for shard_rank in range(world_size):
            shard_path = Path(f"{output}.rank{shard_rank:05d}.npz")
            with np.load(shard_path, allow_pickle=False) as stored:
                all_ids.append(stored["doc_ids"])
                all_vectors.append(stored["vectors"])
                dimensions = int(stored["vectors"].shape[1])
        merged_ids = np.concatenate(all_ids)
        merged_vectors = np.concatenate(all_vectors)
        if [str(value) for value in merged_ids] != doc_ids:
            raise ValueError("distributed shards do not preserve frozen doc-id order")
        np.savez(output, doc_ids=merged_ids, vectors=merged_vectors)
        sha256 = digest_file(output)
        manifest = {
            "schema_version": 1,
            "index": str(output),
            "sha256": sha256,
            "document_count": len(doc_ids),
            "doc_ids_sha256": hashlib.sha256("\n".join(doc_ids).encode()).hexdigest(),
            "dimensions": dimensions,
            "pooling": "last_token",
            "query_instruction": QUERY_INSTRUCTION,
            "world_size": world_size,
            "dtype": "float16",
            "model": str(Path(args.model).resolve()),
        }
        Path(f"{output}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not args.keep_shards:
            for shard_rank in range(world_size):
                Path(f"{output}.rank{shard_rank:05d}.npz").unlink()
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
