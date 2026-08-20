from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED: dict[str, set[str]] = {
    "retriever": {
        "model",
        "model_revision",
        "dataset",
        "dataset_revision",
        "official_checkpoint_revision",
        "temperature",
        "global_batch_size",
        "seed",
    },
    "sft": {"model", "model_revision", "context_length", "lora", "datasets", "go_gate", "seed"},
    "grpo": {
        "phase",
        "framework_revision",
        "rollouts_per_prompt",
        "prompts_per_step",
        "trajectories_per_step",
        "reward",
        "seed",
    },
    "evaluation": {"experiments", "retrievers", "in_domain", "ood", "seed"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML configs without making PyYAML a runtime dependency."""
    source = Path(path)
    try:
        config = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} must be JSON-compatible YAML: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")
    return config


def validate_config(config: dict[str, Any], expected_kind: str | None = None) -> list[str]:
    errors: list[str] = []
    kind = config.get("kind")
    if expected_kind and kind != expected_kind:
        errors.append(f"expected kind={expected_kind!r}, got {kind!r}")
    if kind not in REQUIRED:
        errors.append(f"unknown config kind: {kind!r}")
        return errors
    for key in sorted(REQUIRED[kind] - set(config)):
        errors.append(f"missing required field: {key}")
    if config.get("seed") != 36:
        errors.append("all frozen configs must use seed 36")
    if kind == "retriever":
        if config.get("temperature") != 0.02:
            errors.append("LRAT temperature must be 0.02")
        if config.get("dataset_revision") in {None, "", "REPLACE_WITH_FROZEN_REVISION"}:
            errors.append("dataset_revision must be pinned before training")
        if config.get("model_revision") in {None, "", "REPLACE_WITH_FROZEN_REVISION"}:
            errors.append("model_revision must be pinned before training")
    if kind == "sft" and config.get("context_length") != 8192:
        errors.append("main SFT context_length must be 8192")
    if kind == "sft" and config.get("model_revision") in {None, "", "REPLACE_WITH_FROZEN_REVISION"}:
        errors.append("Policy model_revision must be pinned before training")
    if kind == "grpo":
        rollouts = config.get("rollouts_per_prompt")
        prompts = config.get("prompts_per_step")
        if isinstance(rollouts, int) and isinstance(prompts, int):
            if config.get("trajectories_per_step") != rollouts * prompts:
                errors.append(
                    "trajectories_per_step must equal prompts_per_step * rollouts_per_prompt"
                )
        credit_assignment = config.get("credit_assignment", "terminal")
        if credit_assignment not in {"terminal", "abc"}:
            errors.append("credit_assignment must be terminal or abc")
        abc_beta = config.get("abc_beta", 1.0)
        if not isinstance(abc_beta, (int, float)) or abc_beta < 0:
            errors.append("abc_beta must be a non-negative number")
        if credit_assignment == "abc":
            if config.get("abc_process_positive_cap") != 0.25:
                errors.append("ABC positive process cap must be pinned to 0.25")
            if config.get("abc_process_negative_cap") != 0.10:
                errors.append("ABC negative process cap must be pinned to 0.10")
            if config.get("actor_adv_norm", "missing") is not None:
                errors.append("ABC actor_adv_norm must be null")
            if config.get("actor_loss_reduction") != "sequence_mean":
                errors.append("ABC actor_loss_reduction must be sequence_mean")
    return errors
