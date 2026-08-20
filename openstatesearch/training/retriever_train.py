from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from openstatesearch.retriever.lrat import weighted_contrastive_loss
from openstatesearch.retriever.transformer_dense import format_query, last_token_pool
from openstatesearch.training import load_config, validate_config


def _normalize_record(value: dict[str, Any]) -> dict[str, Any] | None:
    """Accept both the released LRAT schema and the compact prepared schema."""
    if not value.get("satisfied", True):
        return None
    positives = value.get("pos") or []
    negatives = value.get("neg") or value.get("negatives") or []
    positive = value.get("positive") or (positives[0] if positives else None)
    negative = value.get("negative") or (negatives[0] if negatives else None)
    if not value.get("query") or not positive or not negative:
        return None
    return {
        "query": str(value["query"]),
        "positive": str(positive),
        "negative": str(negative),
        "reweight_rate": float(value.get("reweight_rate", 1.0)),
    }


def _records(path: str) -> Any:
    if Path(path).suffix == ".parquet":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("datasets is required to read prepared LRAT parquet") from exc
        values = load_dataset("parquet", data_files=path, split="train")
        if not len(values):
            raise ValueError("no LRAT pairs found")
        return values
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                normalized = _normalize_record(value)
                if normalized is not None:
                    values.append(normalized)
    if not values:
        raise ValueError("no satisfied LRAT pairs found")
    return values


def main() -> None:  # pragma: no cover - requires GPU stack
    parser = argparse.ArgumentParser(
        description="Train Qwen3-Embedding-0.6B on exported LRAT pairs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pairs", required=True, help="JSONL: query, positive, negatives[], reweight_rate"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-device-batch", type=int, default=16)
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Debug/smoke limit; omit for full epoch"
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume-from", help="Accelerate checkpoint directory")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config, "retriever")
    if errors:
        raise SystemExit("; ".join(errors))
    try:
        import torch
        from accelerate import Accelerator, DataLoaderConfiguration
        from torch.utils.data import DataLoader
        from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("install openstatesearch[retriever] before training") from exc

    if (
        args.per_device_batch * int(__import__("os").environ.get("WORLD_SIZE", "1"))
        != config["global_batch_size"]
    ):
        raise SystemExit(
            "per-device batch * world size must equal configured global_batch_size "
            f"({config['global_batch_size']})"
        )
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, even_batches=True),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"],
        revision=config["model_revision"],
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        config["model"], revision=config["model_revision"], local_files_only=True
    )
    records = _records(args.pairs)
    loader = DataLoader(
        records,
        batch_size=args.per_device_batch,
        shuffle=True,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    total_steps = len(loader) * config["epochs"]
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * config["warmup_ratio"])), total_steps
    )
    # total_steps is computed after dataloader sharding, so this scheduler must be
    # stepped exactly once per optimizer update. AcceleratedScheduler would advance
    # it num_processes times when batches are not split.
    completed_steps = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        state_path = Path(args.resume_from) / "trainer_state.json"
        completed_steps = int(json.loads(state_path.read_text(encoding="utf-8"))["completed_steps"])
    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "run_config.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "pairs": str(Path(args.pairs).resolve()),
                    "world_size": accelerator.num_processes,
                    "per_device_batch": args.per_device_batch,
                    "total_steps": total_steps,
                    "resume_from": args.resume_from,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    model.train()
    started = time.monotonic()
    stop = completed_steps >= total_steps
    for epoch in range(config["epochs"]):
        epoch_start = epoch * len(loader)
        if completed_steps >= epoch_start + len(loader):
            continue
        epoch_loader = loader
        if completed_steps > epoch_start:
            epoch_loader = accelerator.skip_first_batches(loader, completed_steps - epoch_start)
        for batch in epoch_loader:
            queries = [format_query(str(item["query"])) for item in batch]
            positives = [str(item["positive"]) for item in batch]
            hard_negatives = [str(item["negative"]) for item in batch]
            query_tokens = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=config["query_length"],
                return_tensors="pt",
            )
            doc_tokens = tokenizer(
                positives + hard_negatives,
                padding=True,
                truncation=True,
                max_length=config["document_length"],
                return_tensors="pt",
            )
            query_tokens = {
                key: value.to(accelerator.device) for key, value in query_tokens.items()
            }
            doc_tokens = {key: value.to(accelerator.device) for key, value in doc_tokens.items()}
            query_embeddings = last_token_pool(
                model(**query_tokens).last_hidden_state, query_tokens["attention_mask"]
            )
            doc_embeddings = last_token_pool(
                model(**doc_tokens).last_hidden_state, doc_tokens["attention_mask"]
            )
            # Every query sees the complete global batch (128 positives + 128 original
            # hard negatives). Restore the local slice so document gradients still flow;
            # DDP supplies the remaining slices on their owning ranks.
            gathered_documents = accelerator.gather(doc_embeddings.detach())
            local_docs = doc_embeddings.shape[0]
            local_start = accelerator.process_index * local_docs
            document_embeddings = torch.cat(
                (
                    gathered_documents[:local_start],
                    doc_embeddings,
                    gathered_documents[local_start + local_docs :],
                ),
                dim=0,
            )
            targets = local_start + torch.arange(len(batch), device=accelerator.device)
            weights = torch.tensor(
                [item.get("reweight_rate", 1.0) for item in batch], device=accelerator.device
            )
            loss = weighted_contrastive_loss(
                query_embeddings, document_embeddings, targets, weights, config["temperature"]
            )
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
            mean_loss = accelerator.gather(loss.detach()).float().mean().item()
            if accelerator.is_main_process and (
                completed_steps == 1 or completed_steps % 10 == 0 or completed_steps >= total_steps
            ):
                elapsed = time.monotonic() - started
                row = {
                    "step": completed_steps,
                    "total_steps": total_steps,
                    "loss": mean_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed
                    / max(1, completed_steps)
                    * (total_steps - completed_steps),
                }
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                accelerator.print(json.dumps(row))
            if args.save_every > 0 and completed_steps % args.save_every == 0:
                checkpoint = output / f"checkpoint-{completed_steps:06d}"
                accelerator.save_state(str(checkpoint))
                if accelerator.is_main_process:
                    (checkpoint / "trainer_state.json").write_text(
                        json.dumps({"completed_steps": completed_steps, "total_steps": total_steps})
                        + "\n",
                        encoding="utf-8",
                    )
            if completed_steps >= total_steps:
                stop = True
                break
        if stop:
            break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        (output / "trainer_state.json").write_text(
            json.dumps({"completed_steps": completed_steps, "total_steps": total_steps}) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
