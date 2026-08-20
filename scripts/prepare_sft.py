#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import statistics
from collections.abc import Mapping
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterable


_WORKER_TOKENIZER: Any = None


def init_worker(model: str, revision: str) -> None:
    global _WORKER_TOKENIZER
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=True
    )


def text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("text"):
            texts.append(str(item["text"]))
        elif isinstance(item, str):
            texts.append(item)
    return "\n".join(texts).strip()


def normalize_messages(raw_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    system_parts: list[str] = []
    conversation: list[dict[str, str]] = []
    for message in raw_messages:
        role = str(message.get("role", ""))
        content = text_content(message)
        if not content:
            continue
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role in {"user", "assistant", "tool"}:
            conversation.append({"role": role, "content": content})
    first_user = next((i for i, item in enumerate(conversation) if item["role"] == "user"), None)
    if first_user is None:
        return []
    conversation = conversation[first_user:]
    if system_parts:
        conversation.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return conversation


def render_ids(messages: list[dict[str, str]], tokenizer: Any) -> list[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    return list(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)


def assistant_only(messages: list[dict[str, str]], tokenizer: Any) -> tuple[list[int], list[int]]:
    input_ids = render_ids(messages, tokenizer)
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
    if not input_ids or not any(label != -100 for label in labels):
        raise ValueError("window contains no assistant training tokens")
    return input_ids, labels


def window_at_boundary(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    selector: int,
    force_final: bool,
) -> list[dict[str, str]]:
    assistant_indices = [i for i, item in enumerate(messages) if item["role"] == "assistant"]
    if not assistant_indices:
        raise ValueError("trajectory contains no assistant message")
    target = (
        assistant_indices[-1]
        if force_final
        else assistant_indices[selector % len(assistant_indices)]
    )
    system = [messages[0]] if messages and messages[0]["role"] == "system" else []
    first_user = next(i for i, item in enumerate(messages) if item["role"] == "user")
    header = system + [messages[first_user]]
    body = messages[first_user + 1 : target + 1]
    if not body:
        raise ValueError("assistant boundary precedes user prompt")

    header_ids = render_ids(header, tokenizer)
    full_ids = render_ids(header + body, tokenizer)
    if len(full_ids) <= max_length:
        return header + body
    if full_ids[: len(header_ids)] != header_ids:
        raise ValueError("chat template is not prefix-stable after the user header")
    body_ids = full_ids[len(header_ids) :]
    start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    starts = [index for index, token in enumerate(body_ids) if token == start_id]
    if len(starts) != len(body):
        raise ValueError(f"rendered message boundary mismatch: {len(starts)} != {len(body)}")
    budget = max_length - len(header_ids)
    start = next(
        (index for index, position in enumerate(starts) if len(body_ids) - position <= budget), None
    )
    if start is not None:
        return header + body[start:]

    # A single unusually long assistant turn still needs a syntactically valid
    # assistant boundary. Keep its tail (where final answers normally occur).
    target_message = dict(body[-1])
    content = target_message["content"]
    left, right = 0, len(content)
    best: list[dict[str, str]] | None = None
    while left <= right:
        keep = (left + right) // 2
        target_message["content"] = content[-keep:] if keep else ""
        trial = header + [target_message]
        if len(render_ids(trial, tokenizer)) <= max_length:
            best = [dict(item) for item in trial]
            left = keep + 1
        else:
            right = keep - 1
    if best is None:
        raise ValueError("system and user header alone exceed context length")
    return best


def selected_openresearcher(
    files: list[Path], per_file: int, seed: int
) -> Iterable[tuple[str, dict[str, Any]]]:
    import pyarrow.parquet as pq

    for source in files:
        metadata = pq.read_table(source, columns=["qid", "status"]).to_pylist()
        ranked: list[tuple[bytes, int]] = []
        for index, row in enumerate(metadata):
            if row.get("status") != "success":
                continue
            key = hashlib.sha256(
                f"{seed}:{source.parent.name}:{source.name}:{row['qid']}".encode()
            ).digest()
            ranked.append((key, index))
        chosen = {index for _, index in heapq.nsmallest(per_file, ranked)}
        if len(chosen) != per_file:
            raise ValueError(
                f"{source} has only {len(chosen)} successful rows; expected {per_file}"
            )
        table = pq.read_table(source, columns=["qid", "messages", "status"])
        for index in sorted(chosen):
            yield (
                f"{source.parent.name}/{source.name}:{index}",
                table.slice(index, 1).to_pylist()[0],
            )


def encode_openseeker_task(
    task: tuple[int, dict[str, Any], int],
) -> tuple[list[int], list[int], str]:
    index, value, max_length = task
    correctness = str(value.get("trajectory correctness", "missing"))
    messages = normalize_messages(value["trajectory"])
    window = window_at_boundary(messages, _WORKER_TOKENIZER, max_length, index, force_final=True)
    ids, labels = assistant_only(window, _WORKER_TOKENIZER)
    if len(ids) > max_length:
        raise AssertionError("OpenSeeker window overflow")
    return ids, labels, correctness


def encode_openresearcher_file_task(
    task: tuple[str, int, int, int],
) -> list[tuple[list[int], list[int]]]:
    source_text, per_file, seed, max_length = task
    source = Path(source_text)
    values = []
    for identity, value in selected_openresearcher([source], per_file, seed):
        messages = normalize_messages(value["messages"])
        selector = int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "big")
        window = window_at_boundary(
            messages, _WORKER_TOKENIZER, max_length, selector, force_final=(selector % 2 == 0)
        )
        ids, labels = assistant_only(window, _WORKER_TOKENIZER)
        if len(ids) > max_length:
            raise AssertionError("OpenResearcher window overflow")
        values.append((ids, labels))
    return values


class PackedWriter:
    def __init__(self, output: Path, max_length: int, batch_rows: int = 64):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.partial = output.with_suffix(output.suffix + ".partial")
        self.schema = pa.schema(
            [("input_ids", pa.list_(pa.int32())), ("labels", pa.list_(pa.int32()))]
        )
        self.writer = pq.ParquetWriter(self.partial, self.schema, compression="zstd")
        self.max_length = max_length
        self.batch_rows = batch_rows
        self.ids: list[int] = []
        self.labels: list[int] = []
        self.rows: list[dict[str, list[int]]] = []
        self.blocks = 0

    def add(self, ids: list[int], labels: list[int]) -> None:
        self.ids.extend(ids)
        self.labels.extend(labels)
        while len(self.ids) >= self.max_length:
            self.rows.append(
                {"input_ids": self.ids[: self.max_length], "labels": self.labels[: self.max_length]}
            )
            del self.ids[: self.max_length]
            del self.labels[: self.max_length]
            self.blocks += 1
            self.flush_if_needed()

    def flush_if_needed(self, force: bool = False) -> None:
        if self.rows and (force or len(self.rows) >= self.batch_rows):
            self.writer.write_table(self.pa.Table.from_pylist(self.rows, schema=self.schema))
            self.rows.clear()

    def close(self, output: Path) -> int:
        if self.ids:
            self.rows.append({"input_ids": self.ids, "labels": self.labels})
            self.blocks += 1
        self.flush_if_needed(force=True)
        self.writer.close()
        self.partial.replace(output)
        return self.blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen assistant-only packed SFT data")
    parser.add_argument("--openseeker", required=True)
    parser.add_argument("--openresearcher-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--openresearcher-count", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = PackedWriter(output, args.max_length)
    token_lengths: list[int] = []
    source_counts = {"OpenSeeker-v1": 0, "OpenResearcher-8k-windows": 0}
    openseeker_correctness: dict[str, int] = {}

    pool = None
    if args.workers > 1:
        pool = get_context("spawn").Pool(
            args.workers, initializer=init_worker, initargs=(args.model, args.revision)
        )
    else:
        init_worker(args.model, args.revision)
    try:

        def openseeker_tasks() -> Iterable[tuple[int, dict[str, Any], int]]:
            with Path(args.openseeker).open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    yield index, json.loads(line), args.max_length

        encoded_openseeker = (
            pool.imap(encode_openseeker_task, openseeker_tasks(), chunksize=2)
            if pool
            else map(encode_openseeker_task, openseeker_tasks())
        )
        for index, (ids, labels, correctness) in enumerate(encoded_openseeker):
            openseeker_correctness[correctness] = openseeker_correctness.get(correctness, 0) + 1
            writer.add(ids, labels)
            token_lengths.append(len(ids))
            source_counts["OpenSeeker-v1"] += 1
            if (index + 1) % 1000 == 0:
                print(json.dumps({"source": "OpenSeeker-v1", "records": index + 1}), flush=True)

        files = sorted(Path(args.openresearcher_root).glob("**/*.parquet"))
        if not files or args.openresearcher_count % len(files):
            raise ValueError(
                "OpenResearcher count must divide evenly across available Parquet files"
            )
        per_file = args.openresearcher_count // len(files)
        file_tasks = [(str(path), per_file, args.seed, args.max_length) for path in files]
        encoded_files = (
            pool.imap(encode_openresearcher_file_task, file_tasks, chunksize=1)
            if pool
            else map(encode_openresearcher_file_task, file_tasks)
        )
        for examples in encoded_files:
            for ids, labels in examples:
                writer.add(ids, labels)
                token_lengths.append(len(ids))
                source_counts["OpenResearcher-8k-windows"] += 1
                count = source_counts["OpenResearcher-8k-windows"]
                if count % 500 == 0:
                    print(json.dumps({"source": "OpenResearcher", "records": count}), flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()

    if source_counts != {"OpenSeeker-v1": 11677, "OpenResearcher-8k-windows": 12000}:
        raise ValueError(f"frozen SFT source counts do not match config: {source_counts}")
    blocks = writer.close(output)
    digest = hashlib.sha256()
    with output.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.revision,
        "seed": args.seed,
        "context_length": args.max_length,
        "packing": True,
        "assistant_only_loss": True,
        "source_records": source_counts,
        "openseeker_correctness": openseeker_correctness,
        "packed_blocks": blocks,
        "record_token_length": {
            "min": min(token_lengths),
            "median": statistics.median(token_lengths),
            "max": max(token_lengths),
            "mean": statistics.fmean(token_lengths),
        },
        "output_sha256": digest.hexdigest(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
