#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.agent.harness import SearchHarness
from openstatesearch.agent.schemas import ActionValidationError, parse_action
from openstatesearch.agent.state import Budget, SearchState
from openstatesearch.data.corpus import read_jsonl
from openstatesearch.eval.observations import transcript_policy_messages
from openstatesearch.eval.runner import evaluate_by_dataset
from openstatesearch.eval.sft_gate import passes_sft_gate, sft_gate_metrics
from openstatesearch.retriever import HybridRetriever, TransformerDenseRetriever
from openstatesearch.retriever.http import HttpRetriever
from openstatesearch.retriever.service import load_corpus
from openstatesearch.training import load_config
from openstatesearch.training.areal_agent import (
    _policy_messages,
    build_policy_input,
    legal_action_space,
)


def _fetch_retriever_provenance(url: str, expected_name: str) -> dict[str, Any]:
    opener = build_opener(ProxyHandler({}))
    with opener.open(f"{url.rstrip('/')}/provenance", timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("retriever provenance endpoint did not return an object")
    claimed = value.pop("provenance_sha256", None)
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    actual = hashlib.sha256(canonical).hexdigest()
    value["provenance_sha256"] = claimed
    if claimed != actual:
        raise ValueError("retriever provenance SHA is invalid")
    if expected_name != "unspecified" and value.get("name") != expected_name:
        raise ValueError(f"retriever provenance name {value.get('name')!r} != {expected_name!r}")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generate_http(
    url: str,
    backend: str,
    model: str,
    input_ids: list[int],
    *,
    max_completion_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    stop_token_ids: list[int],
) -> tuple[str, int]:
    sampling_params = {
        "max_new_tokens": max_completion_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "stop_token_ids": stop_token_ids,
        "skip_special_tokens": True,
    }
    if backend == "sglang":
        payload = {
            # Match AReaL's SGLang path: render the chat template client-side
            # and send token IDs to /generate.
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "stream": False,
        }
        endpoint = "/generate"
    elif backend == "vllm":
        payload = {
            "model": model,
            "prompt": input_ids,
            "max_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "stop_token_ids": stop_token_ids,
            "skip_special_tokens": True,
            "stream": False,
        }
        endpoint = "/v1/completions"
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unsupported generation backend: {backend}")
    request = Request(
        f"{url.rstrip('/')}{endpoint}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        result = json.loads(response.read().decode("utf-8"))
    if backend == "sglang":
        text = str(result.get("text") or "")
        completion_tokens = int((result.get("meta_info") or {}).get("completion_tokens", 0))
    else:
        text = str(result["choices"][0].get("text") or "")
        completion_tokens = int((result.get("usage") or {}).get("completion_tokens", 0))
    return text, completion_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out SFT trajectories and enforce Go Gate"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--dense-model")
    parser.add_argument("--dense-index")
    parser.add_argument("--retriever-url")
    parser.add_argument(
        "--generation-url",
        help="Optional OpenAI-compatible generation server; otherwise load the model locally.",
    )
    parser.add_argument(
        "--generation-model",
        help=(
            "Optional model name sent to a remote generation server. Defaults to "
            "--model, which remains the local tokenizer/provenance path."
        ),
    )
    parser.add_argument(
        "--generation-backend",
        choices=("sglang", "vllm"),
        default="sglang",
        help="HTTP generation protocol used by --generation-url.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/policy_sft.yaml")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=16)
    parser.add_argument(
        "--state-mode",
        choices=("external_state", "transcript"),
        default="external_state",
    )
    parser.add_argument("--memory-token-budget", type=int, default=8192)
    parser.add_argument("--generation-token-budget", type=int, default=8192)
    parser.add_argument("--max-action-tokens", type=int, default=256)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 keeps the historical greedy gate.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling cutoff; Qwen checkpoint default is 20.",
    )
    parser.add_argument("--sampling-seed", type=int, default=36)
    parser.add_argument(
        "--sampling-scheme",
        choices=("legacy_shard_stream", "per_prompt_turn_v1"),
        default="per_prompt_turn_v1",
        help="Use per-action seeds for shard/scheduling invariance; legacy is only for old checkpoint parity.",
    )
    parser.add_argument("--experiment-name", default="unspecified")
    parser.add_argument("--retriever-name", default="unspecified")
    parser.add_argument("--require-retriever-provenance", action="store_true")
    parser.add_argument("--model-provenance-manifest")
    parser.add_argument("--require-model-provenance", action="store_true")
    parser.add_argument(
        "--disable-flash-linear-attention",
        action="store_true",
        help=(
            "Force the Transformers Qwen3.5 torch fallback. Useful for multi-process "
            "evaluation where independent FLA/Triton autotuning dominates startup."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument(
        "--skip-gate-enforcement",
        action="store_true",
        help="Keep rollout records/metrics without treating this dataset as the held-out Go Gate.",
    )
    args = parser.parse_args()
    if args.memory_token_budget <= args.max_action_tokens:
        parser.error("memory-token-budget must exceed max-action-tokens")
    if args.generation_token_budget <= 0 or args.max_action_tokens <= 0:
        parser.error("generation-token-budget and max-action-tokens must be positive")
    if args.temperature < 0:
        parser.error("temperature must be non-negative")
    if not (0 < args.top_p <= 1):
        parser.error("top-p must be in (0, 1]")
    if args.top_k <= 0:
        parser.error("top-k must be positive")
    model_provenance_path = (
        Path(args.model_provenance_manifest) if args.model_provenance_manifest else None
    )
    if model_provenance_path is not None and not model_provenance_path.is_file():
        parser.error("--model-provenance-manifest does not exist")
    if args.require_model_provenance and model_provenance_path is None:
        parser.error("--require-model-provenance requires --model-provenance-manifest")
    model_provenance_sha = _sha256_path(model_provenance_path) if model_provenance_path else None

    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    if args.disable_flash_linear_attention:
        # This must happen before AutoModel imports modeling_qwen3_5.  Merely
        # setting a generation flag later is too late: optional FLA kernels are
        # selected at module import time.
        import transformers.utils.import_utils as transformers_import_utils

        transformers_import_utils.is_flash_linear_attention_available = lambda: False

    if not args.generation_url and not torch.cuda.is_available():
        raise RuntimeError("SFT gate generation requires CUDA")
    torch.manual_seed(args.sampling_seed)
    torch.cuda.manual_seed_all(args.sampling_seed)
    available = list(read_jsonl(args.prompts))
    datasets = sorted({str(prompt.get("dataset", "")) for prompt in available})
    if datasets and len(datasets) > 1:
        by_dataset = {
            dataset: [prompt for prompt in available if str(prompt.get("dataset", "")) == dataset]
            for dataset in datasets
        }
        prompts = []
        for index in range(args.count):
            dataset = datasets[index % len(datasets)]
            offset = index // len(datasets)
            if offset < len(by_dataset[dataset]):
                prompts.append(by_dataset[dataset][offset])
    else:
        prompts = available[: args.count]
    if len(prompts) != args.count:
        raise ValueError(f"requested {args.count} prompts, found {len(prompts)}")
    stop_index = args.stop_index if args.stop_index is not None else len(prompts)
    if not (0 <= args.start_index <= stop_index <= len(prompts)):
        raise ValueError("require 0 <= start-index <= stop-index <= count")
    prompts = prompts[args.start_index : stop_index]
    if args.retriever_url:
        retriever = HttpRetriever(args.retriever_url)
        try:
            retriever_provenance = _fetch_retriever_provenance(
                args.retriever_url, args.retriever_name
            )
        except Exception:
            if args.require_retriever_provenance:
                raise
            retriever_provenance = None
    else:
        if not all((args.corpus, args.dense_model, args.dense_index)):
            parser.error("provide --retriever-url or all of --corpus/--dense-model/--dense-index")
        documents = load_corpus(args.corpus)
        dense = TransformerDenseRetriever(
            documents, args.dense_model, args.dense_index, device="cuda", dtype="bfloat16"
        )
        retriever = HybridRetriever(documents, dense=dense)
        retriever_provenance = None
        if args.require_retriever_provenance:
            parser.error("--require-retriever-provenance currently requires --retriever-url")
    output = Path(args.output)
    run_config = {
        "schema_version": 1,
        "model": args.model,
        "model_provenance_manifest": (
            str(model_provenance_path) if model_provenance_path else None
        ),
        "model_provenance_sha256": model_provenance_sha,
        "prompts": args.prompts,
        "prompts_sha256": _sha256_path(Path(args.prompts)),
        "retriever_url": args.retriever_url,
        "generation_url": args.generation_url,
        "generation_model": args.generation_model,
        "generation_backend": args.generation_backend,
        "retriever_name": args.retriever_name,
        "retriever_provenance_sha256": (
            retriever_provenance.get("provenance_sha256") if retriever_provenance else None
        ),
        "experiment": args.experiment_name,
        "state_mode": args.state_mode,
        "memory_token_budget": args.memory_token_budget,
        "generation_token_budget": args.generation_token_budget,
        "max_action_tokens": args.max_action_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "sampling_seed": args.sampling_seed,
        "sampling_scheme": args.sampling_scheme,
        "max_turns": args.max_turns,
        "disable_flash_linear_attention": args.disable_flash_linear_attention,
        "count": args.count,
        "start_index": args.start_index,
        "stop_index": stop_index,
    }
    canonical_config = json.dumps(
        run_config, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    run_config["run_config_sha256"] = hashlib.sha256(canonical_config).hexdigest()
    run_config_path = output.with_suffix(output.suffix + ".run_config.json")
    if output.exists():
        if not run_config_path.is_file():
            raise ValueError(f"refusing to resume {output} without a frozen run-config sidecar")
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing_config != run_config:
            raise ValueError(f"refusing to resume {output} with different configuration")
    else:
        if run_config_path.is_file():
            existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
            if existing_config != run_config:
                raise ValueError(f"stale run-config conflicts with {output}")
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        run_config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    # The OpenStateSearch policy path is text-only.  Loading AutoProcessor here
    # unnecessarily couples evaluation to image/video preprocessor discovery;
    # some otherwise valid merged Qwen checkpoints only carry the nested
    # processor_config.json and fail that discovery on a fresh host.  The
    # tokenizer owns the chat template and is the only processor used below.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    model = None
    if not args.generation_url:
        model = (
            AutoModelForMultimodalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
            )
            .to("cuda")
            .eval()
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[tuple[str, str]] = set()
    if output.exists():
        completed_ids = {
            (str(record.get("dataset", "")), str(record["id"]))
            for record in read_jsonl(output)
            if isinstance(record.get("id"), str)
        }
    mode = "a" if completed_ids else "w"
    with output.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for prompt_index, prompt in enumerate(prompts, start=args.start_index):
            identity = str(prompt.get("id", prompt_index))
            identity_key = (str(prompt.get("dataset", "")), identity)
            if identity_key in completed_ids:
                continue
            question = str(prompt["question"])
            state = SearchState(
                question,
                constraints=list(prompt.get("constraints", [])),
                budget=Budget(
                    search_left=4,
                    open_left=4,
                    token_left=args.generation_token_budget,
                ),
            )
            harness = SearchHarness(state, retriever, top_k=5)
            last_result: dict[str, Any] | None = None
            actions: list[dict[str, Any]] = []
            input_tokens_total = 0
            generated_tokens_total = 0
            transcript_history: list[dict[str, Any]] = []
            initial_budget = asdict(state.budget)
            initial_legal = legal_action_space(state.observation(), [], args.max_turns, None)
            for turn_index in range(args.max_turns):
                remaining_turns = args.max_turns - turn_index
                max_completion_tokens = min(args.max_action_tokens, state.budget.token_left)
                if max_completion_tokens <= 0:
                    break
                max_prompt_tokens = args.memory_token_budget - max_completion_tokens
                if args.state_mode == "external_state":
                    policy_input = build_policy_input(harness, remaining_turns, last_result)
                    messages = _policy_messages(policy_input, tokenizer, max_prompt_tokens)
                    dropped_transcript_turns = 0
                else:
                    messages, dropped_transcript_turns = transcript_policy_messages(
                        question=question,
                        constraints=list(prompt.get("constraints", [])),
                        initial_budget=initial_budget,
                        initial_remaining_turns=args.max_turns,
                        initial_legal_action_space=initial_legal,
                        history=transcript_history,
                        tokenizer=tokenizer,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(rendered, return_tensors="pt")
                input_tokens = int(inputs["input_ids"].shape[1])
                if input_tokens > max_prompt_tokens:
                    raise ValueError(
                        f"rendered policy input has {input_tokens} tokens; "
                        f"budget permits {max_prompt_tokens}"
                    )
                if args.generation_url:
                    action_seed = args.sampling_seed
                    if args.sampling_scheme == "per_prompt_turn_v1":
                        action_seed += prompt_index * args.max_turns + turn_index
                    text, generated_tokens = _generate_http(
                        args.generation_url,
                        args.generation_backend,
                        args.generation_model or args.model,
                        inputs["input_ids"][0].tolist(),
                        max_completion_tokens=max_completion_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        seed=action_seed,
                        stop_token_ids=list(
                            {
                                int(tokenizer.eos_token_id),
                                int(tokenizer.pad_token_id),
                            }
                        ),
                    )
                    text = text.strip()
                else:
                    assert model is not None
                    if args.sampling_scheme == "per_prompt_turn_v1":
                        action_seed = (
                            args.sampling_seed + prompt_index * args.max_turns + turn_index
                        )
                        torch.manual_seed(action_seed)
                        torch.cuda.manual_seed_all(action_seed)
                    inputs = inputs.to("cuda")
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": max_completion_tokens,
                        "do_sample": args.temperature > 0,
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                    }
                    if args.temperature > 0:
                        generation_kwargs.update(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                        )
                    generated = model.generate(**inputs, **generation_kwargs)
                    text = tokenizer.decode(
                        generated[0, inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    ).strip()
                    generated_tokens = int(generated.shape[1] - inputs["input_ids"].shape[1])
                input_tokens_total += input_tokens
                generated_tokens_total += generated_tokens
                # Every model completion consumes budget, including malformed or
                # terminal ANSWER actions.  Accounting before dispatch prevents a
                # successful final action from disappearing from the cost record.
                harness.consume_tokens(generated_tokens)
                visible = [candidate.doc_id for candidate in state.candidate_pool]
                try:
                    raw_action: Any = json.loads(text)
                except json.JSONDecodeError:
                    raw_action = text
                actions.append(
                    {
                        "action": raw_action,
                        "visible_doc_ids": visible,
                        "opened_doc_ids": sorted(harness.opened),
                        "raw": text,
                        "input_tokens": input_tokens,
                        "generated_tokens": generated_tokens,
                        "dropped_transcript_turns": dropped_transcript_turns,
                    }
                )
                try:
                    action = parse_action(raw_action)
                    result = harness.apply(action)
                    last_result = result.to_dict()
                    actions[-1]["tool_result"] = last_result
                    if harness.finished:
                        break
                except (ActionValidationError, TypeError):
                    last_result = {"ok": False, "action": "INVALID", "error": "invalid action JSON"}
                    actions[-1]["tool_result"] = last_result
                if args.state_mode == "transcript":
                    next_remaining_turns = max(0, remaining_turns - 1)
                    current_state = state.observation()
                    transcript_history.append(
                        {
                            "assistant": text,
                            "observation": {
                                "tool_result": last_result,
                                "budget": asdict(state.budget),
                                "remaining_turns": next_remaining_turns,
                                "legal_action_space": legal_action_space(
                                    current_state,
                                    sorted(harness.opened),
                                    next_remaining_turns,
                                    last_result,
                                ),
                            },
                        }
                    )
                if state.budget.token_left <= 0:
                    break
            final_answer = asdict(harness.answer) if harness.answer else None
            evidence = [asdict(item) for item in state.evidence]
            record = {
                "id": identity,
                "dataset": prompt.get("dataset"),
                "question": question,
                "actions": actions,
                "trajectory_completed": harness.finished,
                "final_answer": final_answer,
                # Canonical prediction fields make this trajectory directly
                # consumable by openstatesearch.eval.runner while retaining the
                # full replay/audit trace above.
                "prediction": str(final_answer.get("answer", "")) if final_answer else "",
                "answers": prompt.get("answers", [prompt.get("answer", "")]),
                "evidence": evidence,
                "legal_evidence": evidence,
                "citations": final_answer.get("citations", []) if final_answer else [],
                "gold_evidence": prompt.get("gold_evidence", []),
                "search_count": 4 - state.budget.search_left,
                "open_count": 4 - state.budget.open_left,
                "input_tokens": input_tokens_total,
                "generated_tokens": generated_tokens_total,
                "state_mode": args.state_mode,
                "memory_token_budget": args.memory_token_budget,
                "generation_token_budget": args.generation_token_budget,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "sampling_seed": args.sampling_seed,
                "experiment": args.experiment_name,
                "retriever": args.retriever_name,
                "generation_url": args.generation_url,
                "generation_model": args.generation_model,
                "retriever_provenance_sha256": (
                    retriever_provenance.get("provenance_sha256") if retriever_provenance else None
                ),
                "model": args.model,
                "model_provenance_sha256": model_provenance_sha,
                "run_config_sha256": run_config["run_config_sha256"],
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "completed": prompt_index + 1,
                        "id": identity,
                        "turns": len(actions),
                        "finished": harness.finished,
                    }
                ),
                flush=True,
            )

    records = list(read_jsonl(output))
    metrics = sft_gate_metrics(records)
    config = load_config(args.config)
    metrics["passed"] = passes_sft_gate(metrics, config["go_gate"])
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evaluation_metrics = evaluate_by_dataset(records)
    evaluation_path = output.with_suffix(output.suffix + ".eval_metrics.json")
    evaluation_path.write_text(
        json.dumps(evaluation_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not metrics["passed"] and not args.skip_gate_enforcement:
        raise SystemExit("SFT Go Gate failed; RL must not start")


if __name__ == "__main__":
    main()
