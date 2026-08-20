#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.runner import evaluate_by_dataset
from openstatesearch.eval.sft_gate import passes_sft_gate, sft_gate_metrics
from openstatesearch.training import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic SFT gate shards")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--config", default="configs/policy_sft.yaml")
    parser.add_argument("--expected-experiment")
    parser.add_argument("--expected-state-mode", choices=("external_state", "transcript"))
    parser.add_argument("--expected-retriever")
    parser.add_argument("--expected-memory-token-budget", type=int)
    parser.add_argument("--require-retriever-provenance", action="store_true")
    parser.add_argument("--require-model-provenance", action="store_true")
    parser.add_argument("--require-run-config", action="store_true")
    args = parser.parse_args()

    records = {}
    retriever_provenance_values: set[str] = set()
    model_provenance_values: set[str] = set()
    source_stats = []
    for source in args.inputs:
        source_path = Path(source)
        run_config_path = source_path.with_suffix(source_path.suffix + ".run_config.json")
        run_config = None
        if run_config_path.is_file():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        elif args.require_run_config:
            raise ValueError(f"source {source_path} is missing its run-config sidecar")
        source_rows = 0
        for record in read_jsonl(source_path):
            source_rows += 1
            identity = (str(record.get("dataset", "")), str(record["id"]))
            if identity in records and records[identity] != record:
                raise ValueError(f"conflicting records for {identity}")
            expected = {
                "experiment": args.expected_experiment,
                "state_mode": args.expected_state_mode,
                "retriever": args.expected_retriever,
                "memory_token_budget": args.expected_memory_token_budget,
            }
            for field, value in expected.items():
                if value is not None and record.get(field) != value:
                    raise ValueError(
                        f"record {identity} has {field}={record.get(field)!r}; expected {value!r}"
                    )
            provenance = record.get("retriever_provenance_sha256")
            if provenance is not None:
                if not isinstance(provenance, str) or len(provenance) != 64:
                    raise ValueError(f"record {identity} has invalid retriever provenance")
                retriever_provenance_values.add(provenance)
            elif args.require_retriever_provenance:
                raise ValueError(f"record {identity} is missing retriever provenance")
            model_provenance = record.get("model_provenance_sha256")
            if model_provenance is not None:
                if not isinstance(model_provenance, str) or len(model_provenance) != 64:
                    raise ValueError(f"record {identity} has invalid model provenance")
                model_provenance_values.add(model_provenance)
            elif args.require_model_provenance:
                raise ValueError(f"record {identity} is missing model provenance")
            if args.require_run_config:
                if record.get("run_config_sha256") != run_config.get("run_config_sha256"):
                    raise ValueError(f"record {identity} does not match its run-config")
            records[identity] = record
        source_stat = {
            "path": str(source_path),
            "rows": source_rows,
            "sha256": _sha256(source_path),
        }
        if run_config_path.is_file():
            source_stat["run_config"] = str(run_config_path)
            source_stat["run_config_sha256"] = _sha256(run_config_path)
        source_stats.append(source_stat)
    available = list(read_jsonl(args.prompts))
    datasets = sorted({str(prompt.get("dataset", "")) for prompt in available})
    if datasets and len(datasets) > 1:
        by_dataset = {
            dataset: [prompt for prompt in available if str(prompt.get("dataset", "")) == dataset]
            for dataset in datasets
        }
        selected = [
            by_dataset[datasets[index % len(datasets)]][index // len(datasets)]
            for index in range(args.count)
        ]
    else:
        selected = available[: args.count]
    prompt_ids = [(str(item.get("dataset", "")), str(item["id"])) for item in selected]
    missing = [identity for identity in prompt_ids if identity not in records]
    extras = sorted(set(records) - set(prompt_ids))
    if missing or extras:
        raise ValueError(f"missing={len(missing)} extras={len(extras)}")
    if args.require_retriever_provenance and len(retriever_provenance_values) != 1:
        raise ValueError("merged shards must use exactly one retriever provenance")
    if args.require_model_provenance and len(model_provenance_values) != 1:
        raise ValueError("merged shards must use exactly one model provenance")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for identity in prompt_ids:
            handle.write(json.dumps(records[identity], ensure_ascii=False, sort_keys=True) + "\n")
    ordered = [records[identity] for identity in prompt_ids]
    metrics = sft_gate_metrics(ordered)
    metrics["passed"] = passes_sft_gate(metrics, load_config(args.config)["go_gate"])
    output.with_suffix(output.suffix + ".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_metrics = evaluate_by_dataset(ordered)
    output.with_suffix(output.suffix + ".eval_metrics.json").write_text(
        json.dumps(evaluation_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    identity_digest = hashlib.sha256(
        "\n".join(f"{dataset}\t{identity}" for dataset, identity in prompt_ids).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "prompts": {
            "path": args.prompts,
            "sha256": _sha256(Path(args.prompts)),
        },
        "sources": source_stats,
        "output": {
            "path": str(output),
            "rows": len(ordered),
            "sha256": _sha256(output),
            "identity_sha256": identity_digest,
        },
        "expected": {
            "experiment": args.expected_experiment,
            "state_mode": args.expected_state_mode,
            "retriever": args.expected_retriever,
            "memory_token_budget": args.expected_memory_token_budget,
            "retriever_provenance_sha256": (
                next(iter(retriever_provenance_values))
                if len(retriever_provenance_values) == 1
                else None
            ),
            "model_provenance_sha256": (
                next(iter(model_provenance_values)) if len(model_provenance_values) == 1 else None
            ),
        },
        "metrics": {
            "path": str(output.with_suffix(output.suffix + ".eval_metrics.json")),
            "sha256": _sha256(output.with_suffix(output.suffix + ".eval_metrics.json")),
        },
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
