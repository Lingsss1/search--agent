from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from openstatesearch.data.corpus import read_jsonl


USER_START = "<|im_start|>user\n"
MESSAGE_END = "<|im_end|>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _completion_action_name(completion: str) -> str:
    """Recover the action label stored in a rollout completion.

    Reward-audit records identify a prompt rather than a unique sampled trajectory.
    The ordered action labels are therefore useful for disambiguating trajectories
    that share the same prompt and terminal reward.
    """
    decoder = json.JSONDecoder()
    for start, character in enumerate(completion):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(completion[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            return str(value["type"]).upper()
    return "INVALID"


def _trajectory_action_signature(trajectory: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        _completion_action_name(str(interaction.get("completion", "")))
        for interaction in trajectory["interactions"]
    )


def _reward_action_signature(record: dict[str, Any]) -> tuple[str, ...] | None:
    transitions = record.get("credit_transitions")
    if not isinstance(transitions, list) or not transitions:
        return None
    actions: list[str] = []
    for transition in transitions:
        if not isinstance(transition, dict) or not isinstance(transition.get("action"), str):
            return None
        actions.append(str(transition["action"]).upper())
    return tuple(actions)


def extract_policy_input(rendered_prompt: str) -> dict[str, Any]:
    if USER_START not in rendered_prompt:
        raise ValueError("rollout prompt has no user message")
    content = rendered_prompt.rsplit(USER_START, 1)[1].split(MESSAGE_END, 1)[0]
    value = json.loads(content)
    if not isinstance(value, dict) or not isinstance(value.get("state"), dict):
        raise ValueError("rollout user message is not an OpenStateSearch policy input")
    return value


def load_rollout_trajectories(
    rollout_root: str | Path,
    version: int,
    expected_rollouts_per_task: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if version < 0:
        raise ValueError("version must be non-negative")
    root = Path(rollout_root)
    trajectories: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    version_root = root / str(version)
    paths = sorted(version_root.glob("*.jsonl")) if version_root.is_dir() else []
    if not paths:
        paths = sorted(root.glob("*/*.jsonl"))
    for path in paths:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            continue
        interactions = [json.loads(line) for line in lines]
        if int(interactions[0].get("head_version", -1)) != version:
            continue
        if any(int(item.get("head_version", -1)) != version for item in interactions):
            raise ValueError(f"rollout file mixes head versions: {path}")
        task_ids = {int(item["task_id"]) for item in interactions}
        if len(task_ids) != 1:
            raise ValueError(f"rollout file mixes task ids: {path}")
        boundaries = []
        policy_inputs = []
        for index, interaction in enumerate(interactions):
            policy_input = extract_policy_input(str(interaction["prompt"]))
            policy_inputs.append(policy_input)
            if policy_input.get("last_tool_result") is None:
                boundaries.append(index)
        if not boundaries or len(boundaries) % expected_rollouts_per_task:
            raise ValueError(
                f"{path} has {len(boundaries)} rollout starts; "
                f"expected a positive multiple of {expected_rollouts_per_task}"
            )
        boundaries.append(len(interactions))
        task_id = task_ids.pop()
        file_sha = _sha256(path)
        file_mtime_ns = path.stat().st_mtime_ns
        batch_count = (len(boundaries) - 1) // expected_rollouts_per_task
        for batch_index in range(batch_count):
            first_boundary = batch_index * expected_rollouts_per_task
            batch_boundaries = boundaries[
                first_boundary : first_boundary + expected_rollouts_per_task + 1
            ]
            source_key = f"{path}#batch={batch_index}"
            source_files.append(
                {
                    "path": str(path),
                    "sha256": file_sha,
                    "task_id": task_id,
                    "batch_index": batch_index,
                    "source_key": source_key,
                    "mtime_ns": file_mtime_ns,
                    "interaction_start": batch_boundaries[0],
                    "interaction_stop": batch_boundaries[-1],
                }
            )
            for rollout_index, (start, stop) in enumerate(
                zip(batch_boundaries[:-1], batch_boundaries[1:], strict=True)
            ):
                segment = interactions[start:stop]
                rewards = {float(item["original_reward"]) for item in segment}
                if len(rewards) != 1:
                    raise ValueError(
                        f"trajectory reward is inconsistent in {path}:{batch_index}:{rollout_index}"
                    )
                state = policy_inputs[start]["state"]
                question = str(state.get("question", ""))
                if not question:
                    raise ValueError(
                        f"trajectory has no question in {path}:{batch_index}:{rollout_index}"
                    )
                trajectories.append(
                    {
                        "step": version,
                        "task_id": task_id,
                        "rollout_batch_index": batch_index,
                        "rollout_index": rollout_index,
                        "trajectory_key": (f"{version}:{task_id}:{batch_index}:{rollout_index}"),
                        "question": question,
                        "original_reward": rewards.pop(),
                        "interaction_count": len(segment),
                        "rollout_file": str(path),
                        "rollout_source_key": source_key,
                        "rollout_file_sha256": file_sha,
                        "interactions": segment,
                    }
                )
    if not trajectories:
        raise ValueError(f"no rollout trajectories found for version {version}")
    return trajectories, source_files


def _prompt_identity_map(prompts: Iterable[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    result: dict[str, tuple[str, str]] = {}
    for prompt in prompts:
        question = str(prompt.get("question", ""))
        identity = (str(prompt.get("dataset", "")), str(prompt.get("id", "")))
        if not question or not all(identity):
            raise ValueError("prompt requires non-empty question, dataset, and id")
        if question in result and result[question] != identity:
            raise ValueError(f"question maps to multiple prompt identities: {question[:80]!r}")
        result[question] = identity
    return result


def attach_reward_breakdowns(
    trajectories: list[dict[str, Any]],
    reward_records: list[dict[str, Any]],
    prompts: Iterable[dict[str, Any]],
    tolerance: float = 1e-4,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    identities = _prompt_identity_map(prompts)
    full_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        question = trajectory["question"]
        if question not in identities:
            raise ValueError(f"rollout question is absent from prompt dataset: {question[:80]!r}")
        dataset, prompt_id = identities[question]
        trajectory["dataset"] = dataset
        trajectory["prompt_id"] = prompt_id
        full_by_id[prompt_id].append(trajectory)

    raw_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in reward_records:
        raw_by_id[str(record.get("trajectory_id", ""))].append(record)
    matched: list[dict[str, Any]] = []
    unmatched_raw = 0
    ambiguous_reward_matches = 0
    ambiguous_reward_matches_before_action_signature = 0
    action_signature_disambiguated = 0
    action_signature_unmatched = 0
    max_reward_delta = 0.0
    for prompt_id, raw_values in sorted(raw_by_id.items()):
        candidates = full_by_id.get(prompt_id, [])
        available = {value["trajectory_key"]: value for value in candidates}
        for raw in sorted(raw_values, key=_canonical):
            if not available:
                unmatched_raw += 1
                continue
            raw_total = float(raw["total"])
            reward_matches = [
                item
                for item in available.items()
                if abs(float(item[1]["original_reward"]) - raw_total) <= tolerance
            ]
            if not reward_matches:
                unmatched_raw += 1
                continue
            if len(reward_matches) > 1:
                ambiguous_reward_matches_before_action_signature += 1
            action_signature = _reward_action_signature(raw)
            signature_matches = (
                [
                    item
                    for item in reward_matches
                    if _trajectory_action_signature(item[1]) == action_signature
                ]
                if action_signature is not None
                else []
            )
            if len(reward_matches) > 1 and len(signature_matches) == 1:
                action_signature_disambiguated += 1
            if action_signature is not None and not signature_matches:
                action_signature_unmatched += 1
            narrowed_matches = signature_matches or reward_matches
            if len(narrowed_matches) > 1:
                ambiguous_reward_matches += 1
            key, trajectory = min(
                narrowed_matches,
                key=lambda item: (
                    abs(float(item[1]["original_reward"]) - raw_total),
                    item[0],
                ),
            )
            delta = abs(float(trajectory["original_reward"]) - raw_total)
            max_reward_delta = max(max_reward_delta, delta)
            enriched = dict(trajectory)
            enriched["reward_breakdown"] = raw
            matched.append(enriched)
            del available[key]
    return matched, {
        "raw_records": len(reward_records),
        "full_trajectories": len(trajectories),
        "matched_records": len(matched),
        "unmatched_raw_records": unmatched_raw,
        "ambiguous_reward_matches": ambiguous_reward_matches,
        "ambiguous_reward_matches_before_action_signature": (
            ambiguous_reward_matches_before_action_signature
        ),
        "action_signature_disambiguated": action_signature_disambiguated,
        "action_signature_unmatched": action_signature_unmatched,
        "max_reward_delta": max_reward_delta,
    }


def summarize_selected_trajectories(
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the exact sampled trajectories embedded in a full audit."""
    if not selected:
        raise ValueError("selected reward audit trajectories must not be empty")
    component_names = (
        "total",
        "answer_f1",
        "support_recall",
        "citation_precision",
        "duplicate_rate",
        "search_cost",
        "open_cost",
        "token_cost",
        "protocol_penalty",
    )
    components: dict[str, dict[str, float]] = {}
    for name in component_names:
        values = [float(item["reward_breakdown"][name]) for item in selected]
        components[name] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    interaction_counts = [int(item["interaction_count"]) for item in selected]
    valid_count = sum(bool(item["reward_breakdown"]["valid"]) for item in selected)
    dataset_counts: dict[str, int] = defaultdict(int)
    dataset_valid_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        dataset = str(item["dataset"])
        dataset_counts[dataset] += 1
        dataset_valid_counts[dataset] += int(bool(item["reward_breakdown"]["valid"]))
    return {
        "trajectories": len(selected),
        "valid_count": valid_count,
        "valid_rate": valid_count / len(selected),
        "interaction_count": {
            "mean": sum(interaction_counts) / len(interaction_counts),
            "min": min(interaction_counts),
            "max": max(interaction_counts),
        },
        "reward_components": components,
        "datasets": {
            dataset: {
                "trajectories": count,
                "valid_count": dataset_valid_counts[dataset],
                "valid_rate": dataset_valid_counts[dataset] / count,
            }
            for dataset, count in sorted(dataset_counts.items())
        },
    }


def build_full_reward_audit(
    *,
    rollout_root: str | Path,
    reward_audit_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    version: int,
    sample_size: int = 50,
    seed: int = 36,
    model_provenance_path: str | Path | None = None,
    checkpoint_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    reward_path = Path(reward_audit_path)
    prompt_path = Path(prompts_path)
    reward_records = list(read_jsonl(reward_path))
    if any(record.get("step") != version for record in reward_records):
        reward_records = [record for record in reward_records if record.get("step") == version]
    trajectories, source_files = load_rollout_trajectories(rollout_root, version)
    matched, matching = attach_reward_breakdowns(
        trajectories, reward_records, read_jsonl(prompt_path)
    )
    ranked = sorted(
        matched,
        key=lambda item: (
            hashlib.sha256(f"{seed}:{item['trajectory_key']}".encode()).hexdigest(),
            item["trajectory_key"],
        ),
    )
    selected = ranked[:sample_size]
    if len(selected) != sample_size:
        raise ValueError(
            f"only {len(selected)} reward records could be linked to full rollouts; "
            f"need {sample_size}"
        )
    payload = "".join(
        json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in selected
    )
    destination = Path(output_path)
    summary = {
        "schema_version": 1,
        "step": version,
        "seed": seed,
        "sample_size": sample_size,
        "output": str(destination),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "raw_reward_audit": {
            "path": str(reward_path),
            "sha256": _sha256(reward_path),
        },
        "prompts": {"path": str(prompt_path), "sha256": _sha256(prompt_path)},
        "rollout_files": source_files,
        "matching": matching,
        "selected_metrics": summarize_selected_trajectories(selected),
        "selected_trajectory_keys": [item["trajectory_key"] for item in selected],
    }
    if model_provenance_path is not None:
        model_path = Path(model_provenance_path)
        if not model_path.is_file():
            raise ValueError(f"model provenance is missing: {model_path}")
        model_sha = _sha256(model_path)
        summary["model_provenance"] = {
            "path": str(model_path),
            "sha256": model_sha,
        }
        summary["model_provenance_sha256"] = model_sha
    if checkpoint_manifest_path is not None:
        checkpoint_path = Path(checkpoint_manifest_path)
        if not checkpoint_path.is_file():
            raise ValueError(f"checkpoint manifest is missing: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(checkpoint.get("step", -1)) != version:
            raise ValueError(
                f"checkpoint step {checkpoint.get('step')!r} does not match audit {version}"
            )
        summary["checkpoint_manifest"] = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    summary_destination = Path(summary_path)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
