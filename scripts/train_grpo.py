#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.training import load_config, validate_config


_ROLLOUT_SOURCE_FILES = (
    "openstatesearch/training/areal_agent.py",
    "openstatesearch/rewards/credit.py",
    "vendor/AReaL/areal/api/cli_args.py",
    "vendor/AReaL/areal/engine/vllm_remote.py",
    "vendor/AReaL/areal/infra/remote_inf_engine.py",
    "vendor/AReaL/areal/infra/workflow_executor.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_retriever_provenance(url: str, expected_name: str) -> dict[str, object]:
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{url.rstrip('/')}/provenance", timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("retriever provenance endpoint did not return an object")
    claimed = value.pop("provenance_sha256", None)
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    value["provenance_sha256"] = claimed
    if claimed != actual:
        raise ValueError("retriever provenance SHA does not match its payload")
    if value.get("name") != expected_name:
        raise ValueError(f"retriever provenance name {value.get('name')!r} != {expected_name!r}")
    return value


def _file_evidence(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"evidence file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _verify_remote_source_parity(source: str, remote_workspace: str) -> dict[str, object]:
    """Fail before model loading if rollout-node Python sources are stale."""
    local_workspace = Path.cwd().resolve()
    local_hashes = {
        relative: _sha256(local_workspace / relative) for relative in _ROLLOUT_SOURCE_FILES
    }
    remote_paths = [f"{remote_workspace.rstrip('/')}/{p}" for p in _ROLLOUT_SOURCE_FILES]
    completed = subprocess.run(
        ["ssh", source, "sha256sum", *remote_paths],
        check=True,
        capture_output=True,
        text=True,
    )
    remote_hashes: dict[str, str] = {}
    workspace_prefix = f"{remote_workspace.rstrip('/')}/"
    for line in completed.stdout.splitlines():
        digest, separator, path = line.partition("  ")
        if not separator or not path.startswith(workspace_prefix):
            raise RuntimeError(f"unexpected remote sha256sum output: {line!r}")
        remote_hashes[path.removeprefix(workspace_prefix)] = digest
    missing = sorted(set(local_hashes) - set(remote_hashes))
    mismatched = sorted(
        path for path, digest in local_hashes.items() if remote_hashes.get(path) != digest
    )
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if mismatched:
            details.append(f"mismatched={mismatched}")
        raise RuntimeError(f"rollout source parity failed for {source}: " + ", ".join(details))
    return {
        "source": source,
        "remote_workspace": remote_workspace,
        "files": [{"path": path, "sha256": local_hashes[path]} for path in _ROLLOUT_SOURCE_FILES],
    }


def _collect_rollout_artifacts(source: str, experiment_name: str, trial_name: str) -> Path:
    log_dir = Path(f"artifacts/areal/logs/root/{experiment_name}/{trial_name}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    remote_log_dir = str(log_dir)
    subprocess.run(
        [
            "rsync",
            "-a",
            f"{source}:{remote_log_dir}/rollout/",
            str(log_dir / "rollout"),
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            f"{source}:{remote_log_dir}/rollout.log",
            str(log_dir / "rollout.log"),
        ],
        check=True,
    )
    files = sorted(path for path in (log_dir / "rollout").rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"no rollout artifacts collected from {source}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "remote_log_dir": remote_log_dir,
        "files": [_file_evidence(path) for path in files],
    }
    manifest_path = log_dir / "rollout_collection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Policy GRPO frozen config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--areal-config", default="configs/areal_grpo_lora.yaml")
    parser.add_argument(
        "--initialization",
        help=(
            "Override the frozen actor initialization with an audited merged "
            "checkpoint, for example when continuing a gated stability segment."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=("ray", "local"),
        default="ray",
        help="Use Ray for the formal multi-node run; local is only for diagnostics.",
    )
    parser.add_argument("--corpus")
    parser.add_argument("--dense-model")
    parser.add_argument("--dense-index")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after this many optimizer steps (intended for isolated smoke runs).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        help=(
            "Override prompts per optimizer step. Intended for isolated integration "
            "smokes; formal runs should keep the frozen config value."
        ),
    )
    parser.add_argument(
        "--max-tokens-per-mb",
        type=int,
        help=(
            "Override actor.mb_spec.max_tokens_per_mb. Use this to lower the "
            "per-rank memory ceiling when a recovered stability run observes "
            "long-sequence load imbalance."
        ),
    )
    parser.add_argument(
        "--rollout-enforce-eager",
        action="store_true",
        help=(
            "Skip vLLM torch.compile and CUDA-graph capture. This materially "
            "reduces cold-start time for short smoke runs, at the cost of lower "
            "steady-state rollout throughput."
        ),
    )
    parser.add_argument(
        "--recover-every-steps",
        type=int,
        help=(
            "Force recovery checkpoints at a step cadence. Intended to test the "
            "full DCP model+optimizer save path in an isolated max-step run."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        help="Override the AReaL experiment name so smoke and full runs stay isolated.",
    )
    parser.add_argument("--trial-name", help="Override the AReaL trial name.")
    parser.add_argument(
        "--rollout-artifact-source",
        help=(
            "SSH host/alias whose non-shared fileroot contains rollout dumps. "
            "After training, collect and hash its rollout directory locally."
        ),
    )
    parser.add_argument(
        "--rollout-remote-workspace",
        default="/code/openstatesearch",
        help=(
            "Workspace root on --rollout-artifact-source. For Ray runs, key "
            "rollout sources are hash-checked before any model is loaded."
        ),
    )
    parser.add_argument(
        "--retriever-url",
        default=os.environ.get("OSS36_RETRIEVER_URL", "http://127.0.0.1:8036"),
    )
    parser.add_argument("--retriever-name")
    parser.add_argument("--require-retriever-provenance", action="store_true")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Disable the dev evaluator for an infrastructure smoke run.",
    )
    parser.add_argument(
        "--credit-assignment",
        choices=("terminal", "abc"),
        help="Override the frozen reward credit-assignment mode.",
    )
    parser.add_argument(
        "--abc-beta",
        type=float,
        help="Override the frozen weight on deterministic ABC process rewards.",
    )
    parser.add_argument(
        "--rejection-lower",
        type=float,
        help=(
            "Set actor.rejection_sampling.lower for an isolated rollout-policy "
            "mismatch ablation (AReaL IcePop uses 0.5)."
        ),
    )
    parser.add_argument(
        "--rejection-upper",
        type=float,
        help="Override actor.rejection_sampling.upper for a stability ablation.",
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.train_batch_size is not None and args.train_batch_size < 8:
        parser.error(
            "--train-batch-size must be at least 8 so every FSDP data-parallel "
            "rank receives a prompt group"
        )
    if args.max_tokens_per_mb is not None and args.max_tokens_per_mb < 1024:
        parser.error("--max-tokens-per-mb must be at least 1024")
    if args.recover_every_steps is not None:
        if args.recover_every_steps < 1:
            parser.error("--recover-every-steps must be positive")
        if args.max_steps is None:
            parser.error("--recover-every-steps requires --max-steps")
    if args.rejection_lower is not None and args.rejection_lower <= 0.0:
        parser.error("--rejection-lower must be positive")
    if args.rejection_upper is not None and args.rejection_upper <= 0.0:
        parser.error("--rejection-upper must be positive")
    if (
        args.rejection_lower is not None
        and args.rejection_upper is not None
        and args.rejection_lower >= args.rejection_upper
    ):
        parser.error("--rejection-lower must be smaller than --rejection-upper")
    if args.require_retriever_provenance and not args.retriever_name:
        parser.error("--require-retriever-provenance requires --retriever-name")
    config = load_config(args.config)
    areal_config_path = Path(args.areal_config)
    try:
        areal_config = yaml.safe_load(areal_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"invalid AReaL config {areal_config_path}: {exc}") from exc
    if not isinstance(areal_config, dict):
        raise SystemExit(f"AReaL config root must be an object: {areal_config_path}")
    errors = validate_config(config, "grpo")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    credit_assignment = args.credit_assignment or str(config.get("credit_assignment", "terminal"))
    initialization = str(Path(args.initialization or str(config["initialization"])).resolve())
    if credit_assignment not in {"terminal", "abc"}:
        parser.error("config credit_assignment must be 'terminal' or 'abc'")
    abc_beta = args.abc_beta if args.abc_beta is not None else float(config.get("abc_beta", 1.0))
    if abc_beta < 0.0:
        parser.error("--abc-beta must be non-negative")
    print(f"GRPO-{config['phase']} config valid: 4 rollouts, 8-GPU FSDP actor + TP4 vLLM rollout.")
    if not args.validate_only:
        if not args.retriever_url and not all((args.corpus, args.dense_model, args.dense_index)):
            raise SystemExit("local retrieval requires --corpus, --dense-model and --dense-index")
        environment = os.environ.copy()
        environment.setdefault("HF_HOME", "/code/hf_cache")
        environment.setdefault("HF_DATASETS_CACHE", "/code/hf_cache/datasets")
        environment.setdefault("SWANLAB_SAVE_DIR", str(Path("artifacts/runtime/swanlab").resolve()))
        environment.setdefault("SWANLAB_LOG_DIR", str(Path("artifacts/runtime/swanlog").resolve()))
        environment.setdefault("SWANLAB_MODE", "disabled")
        environment.setdefault(
            "FLASHINFER_WORKSPACE_BASE",
            str(Path("artifacts/runtime/flashinfer").resolve()),
        )
        if args.scheduler == "ray":
            environment.setdefault("RAY_ADDRESS", "10.82.123.139:26379")
        # Dense Qwen3.6 rollout is served by vLLM; the legacy SGLang
        # environment is retained only for reproducing the rejected backend.
        areal_venv_bin = Path("vendor/AReaL/.venv-vllm/bin").resolve()
        environment["PATH"] = f"{areal_venv_bin}{os.pathsep}{environment.get('PATH', '')}"
        python_paths = [str(Path.cwd()), str(Path("vendor/AReaL").resolve())]
        if environment.get("PYTHONPATH"):
            python_paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["OSS36_RL_PHASE"] = str(config["phase"])
        environment["OSS36_CREDIT_ASSIGNMENT"] = credit_assignment
        environment["OSS36_ABC_BETA"] = str(abc_beta)
        if args.corpus:
            environment["OSS36_CORPUS"] = str(Path(args.corpus).resolve())
        if args.dense_model:
            environment["OSS36_DENSE_MODEL"] = str(Path(args.dense_model).resolve())
        if args.dense_index:
            environment["OSS36_DENSE_INDEX"] = str(Path(args.dense_index).resolve())
        retriever_url = args.retriever_url.rstrip("/")
        try:
            opener = build_opener(ProxyHandler({}))
            with opener.open(f"{retriever_url}/health", timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as exc:
            raise SystemExit(f"retriever health check failed at {retriever_url}: {exc}") from exc
        retriever_provenance = None
        if args.require_retriever_provenance:
            try:
                retriever_provenance = _fetch_retriever_provenance(
                    retriever_url, str(args.retriever_name)
                )
            except Exception as exc:
                raise SystemExit(
                    f"retriever provenance check failed at {retriever_url}: {exc}"
                ) from exc
        rollout_source_parity = None
        if args.scheduler == "ray" and args.rollout_artifact_source:
            try:
                rollout_source_parity = _verify_remote_source_parity(
                    args.rollout_artifact_source, args.rollout_remote_workspace
                )
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                raise SystemExit(f"rollout source parity check failed: {exc}") from exc
            print(
                "Rollout source parity verified: "
                f"{len(_ROLLOUT_SOURCE_FILES)} files on {args.rollout_artifact_source}."
            )
        environment["OSS36_RETRIEVER_URL"] = retriever_url
        if retriever_provenance is not None:
            environment["OSS36_RETRIEVER_PROVENANCE_SHA256"] = str(
                retriever_provenance["provenance_sha256"]
            )
        environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        areal_python = areal_venv_bin / "python"
        if not areal_python.exists():
            raise SystemExit(f"AReaL Python is missing: {areal_python}")
        phase = str(config["phase"]).lower()
        experiment_name = args.experiment_name or f"oss36-grpo-{phase}"
        trial_name = args.trial_name or f"phase{phase}"
        environment["OSS36_REWARD_AUDIT"] = str(
            Path(f"artifacts/runtime/{experiment_name}_{trial_name}_reward_audit.jsonl").resolve()
        )
        command = [
            str(areal_python),
            "-m",
            "openstatesearch.training.areal_train",
            "--config",
            args.areal_config,
            f"scheduler.type={args.scheduler}",
            f"actor.path={initialization}",
            f"experiment_name={experiment_name}",
            f"trial_name={trial_name}",
            f"train_dataset.path={config['train_dataset']}",
        ]
        if args.max_steps is not None:
            command.append(f"+total_train_steps={args.max_steps}")
            command.append("saver.freq_steps=1")
            if args.recover_every_steps is None:
                command.append("recover.mode=disabled")
            else:
                command.extend(
                    [
                        "recover.mode=auto",
                        f"recover.freq_steps={args.recover_every_steps}",
                        "recover.freq_secs=null",
                    ]
                )
        else:
            command.extend(["recover.mode=auto", "recover.freq_secs=3600"])
        if args.train_batch_size is not None:
            command.append(f"train_dataset.batch_size={args.train_batch_size}")
        if args.max_tokens_per_mb is not None:
            command.append(f"actor.mb_spec.max_tokens_per_mb={args.max_tokens_per_mb}")
        if args.rejection_lower is not None:
            command.append(f"+actor.rejection_sampling.lower={args.rejection_lower}")
        if args.rejection_upper is not None:
            command.append(f"actor.rejection_sampling.upper={args.rejection_upper}")
        if args.rollout_enforce_eager:
            # ``enforce_eager`` is an optional VLLMConfig field and is not
            # materialized in older experiment YAMLs. Hydra struct mode needs
            # append syntax in that case; a plain override fails before any
            # worker is launched.
            command.append("+vllm.enforce_eager=true")
        if args.skip_eval:
            command.append("valid_dataset=null")
        initialization_path = Path(initialization)
        initialization_manifest = initialization_path / "merge_manifest.json"
        if not initialization_manifest.is_file():
            initialization_manifest = initialization_path / "model_provenance.json"
        launch_manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "phase": config["phase"],
            "experiment": experiment_name,
            "trial": trial_name,
            "command": command,
            "scheduler": args.scheduler,
            "credit_assignment": {
                "mode": credit_assignment,
                "abc_beta": abc_beta,
            },
            "cuda_visible_devices": environment["CUDA_VISIBLE_DEVICES"],
            "inputs": {
                "grpo_config": _file_evidence(Path(args.config)),
                "areal_config": _file_evidence(Path(args.areal_config)),
                "train_dataset": _file_evidence(Path(str(config["train_dataset"]))),
                "corpus": (_file_evidence(Path(str(args.corpus))) if args.corpus else None),
                "dense_index": (
                    _file_evidence(Path(str(args.dense_index))) if args.dense_index else None
                ),
                "initialization_manifest": (
                    _file_evidence(initialization_manifest)
                    if initialization_manifest.is_file()
                    else None
                ),
                "dense_model": (str(Path(str(args.dense_model))) if args.dense_model else None),
                "initialization": str(initialization_path),
            },
            "retriever": {
                "url": retriever_url,
                "name": args.retriever_name,
                "provenance": retriever_provenance,
            },
            "recovery": {
                "max_steps": args.max_steps,
                "recover_every_steps": args.recover_every_steps,
                "mode": "disabled" if args.max_steps and not args.recover_every_steps else "auto",
            },
            "train_batch_size_override": args.train_batch_size,
            "max_tokens_per_mb_override": args.max_tokens_per_mb,
            "rollout_enforce_eager": args.rollout_enforce_eager,
            "rollout_artifact_source": args.rollout_artifact_source,
            "rollout_source_parity": rollout_source_parity,
        }
        launch_manifest_path = Path(
            f"artifacts/runtime/{experiment_name}_{trial_name}_launch_manifest.json"
        )
        launch_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        launch_manifest_path.write_text(
            json.dumps(launch_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return_code = subprocess.call(command, env=environment)
        if args.rollout_artifact_source:
            try:
                manifest_path = _collect_rollout_artifacts(
                    args.rollout_artifact_source,
                    experiment_name,
                    trial_name,
                )
                print(f"Collected remote rollout artifacts: {manifest_path}")
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                print(f"ERROR: failed to collect remote rollout artifacts: {exc}")
                if return_code == 0:
                    return_code = 3
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
