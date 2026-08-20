from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from openstatesearch.training import load_config, validate_config


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict) or not isinstance(value.get("messages"), list):
                    raise ValueError("each SFT record requires a messages list")
                records.append(value)
    if not records:
        raise ValueError("SFT data is empty")
    return records


def encode_assistant_only(
    record: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, list[int]]:
    """Encode chat messages and mask every token except assistant spans."""
    messages = record["messages"]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    input_ids = list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)
    labels = [-100] * len(input_ids)
    assistant_header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    index = 0
    while index <= len(input_ids) - len(assistant_header):
        if input_ids[index : index + len(assistant_header)] != assistant_header:
            index += 1
            continue
        end = index + len(assistant_header)
        while end < len(input_ids) and input_ids[end] != end_id:
            end += 1
        if end >= len(input_ids):
            raise ValueError("unterminated assistant span in rendered chat")
        labels[index : end + 1] = input_ids[index : end + 1]
        index = end + 1
    return {"input_ids": input_ids[:max_length], "labels": labels[:max_length]}


def main() -> None:  # pragma: no cover - requires 8-GPU training stack
    parser = argparse.ArgumentParser(description="Qwen3.6-27B assistant-only LoRA SFT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-device-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Debug/smoke limit; -1 runs the epoch"
    )
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    config = load_config(args.config)
    errors = validate_config(config, "sft")
    if errors:
        raise SystemExit("; ".join(errors))
    try:
        import torch
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForMultimodalLM, AutoProcessor, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit("install openstatesearch[training] before SFT") from exc

    model_source = Path(config["model"])
    revision_args = {} if model_source.exists() else {"revision": config["model_revision"]}
    processor = AutoProcessor.from_pretrained(
        config["model"], trust_remote_code=True, local_files_only=True, **revision_args
    )
    tokenizer = processor.tokenizer
    if Path(args.data).suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=args.data, split="train")
        required = {"input_ids", "labels"}
        if not required.issubset(dataset.column_names):
            raise ValueError(f"prepared SFT parquet must contain {sorted(required)}")
    else:
        records = read_jsonl(args.data)
        dataset = Dataset.from_list(records).map(
            lambda record: encode_assistant_only(record, tokenizer, config["context_length"]),
            remove_columns=list(records[0]),
        )
    model = AutoModelForMultimodalLM.from_pretrained(
        config["model"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        **revision_args,
    )
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            target_modules=lora["targets"],
            task_type="CAUSAL_LM",
        ),
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        longest = max(len(feature["input_ids"]) for feature in features)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        ids, labels, masks = [], [], []
        for feature in features:
            padding = longest - len(feature["input_ids"])
            ids.append(feature["input_ids"] + [pad_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            masks.append([1] * len(feature["input_ids"]) + [0] * padding)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks),
        }

    arguments = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=config["optimizer"]["learning_rate"],
        warmup_ratio=config["optimizer"]["warmup_ratio"],
        weight_decay=config["optimizer"]["weight_decay"],
        lr_scheduler_type=config["optimizer"]["schedule"],
        num_train_epochs=config["epochs"],
        bf16=True,
        save_steps=config["save_steps"],
        logging_steps=10,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fsdp="full_shard auto_wrap",
        # Qwen3.6's hybrid linear-attention blocks retain selected FP32 states.
        # FSDP activation checkpointing recomputes those in BF16 and fails the
        # PyTorch metadata check; model-level checkpointing is verified instead.
        fsdp_config={"use_orig_params": False},
        seed=config["seed"],
        data_seed=config["seed"],
        include_num_input_tokens_seen=True,
        report_to="none",
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset, data_collator=collate)
    if int(os.environ.get("RANK", "0")) == 0:
        data_digest = hashlib.sha256()
        with Path(args.data).open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                data_digest.update(chunk)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "run_config.json").write_text(
            json.dumps(
                {
                    "config": config,
                    "data": str(Path(args.data).resolve()),
                    "data_sha256": data_digest.hexdigest(),
                    "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                    "per_device_batch": args.per_device_batch,
                    "gradient_accumulation": args.gradient_accumulation,
                    "max_steps_override": args.max_steps,
                    "resume_from": args.resume_from,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)


if __name__ == "__main__":
    main()
