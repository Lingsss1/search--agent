from __future__ import annotations

import os
import sys


def main(args: list[str] | None = None) -> None:
    """Thin official AReaL GRPO integration using the proxy workflow API."""
    try:
        from areal import PPOTrainer
        from areal.api.cli_args import GRPOConfig, load_expr_config
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - cluster dependency
        raise SystemExit(
            "AReaL is not installed; install the pinned AReaL environment first"
        ) from exc

    config, _ = load_expr_config(args or sys.argv[1:], GRPOConfig)
    dataset = load_dataset("json", data_files=config.train_dataset.path, split="train")
    valid_dataset = (
        load_dataset("json", data_files=config.valid_dataset.path, split="train")
        if config.valid_dataset is not None
        else None
    )
    workflow_kwargs = {
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "export_style": "individual",
        "turn_discount": config.rollout.agent.turn_discount,
        "credit_assignment": os.environ.get("OSS36_CREDIT_ASSIGNMENT", "terminal"),
        "abc_beta": float(os.environ.get("OSS36_ABC_BETA", "1.0")),
        "sampling_seed": config.seed,
    }
    eval_workflow_kwargs = {
        **workflow_kwargs,
        "gconfig": config.eval_gconfig,
    }
    workflow_path = "openstatesearch.training.areal_agent.OpenStateSearchWorkflow"
    with PPOTrainer(config, train_dataset=dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow=workflow_path,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=workflow_path if valid_dataset is not None else None,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main()
