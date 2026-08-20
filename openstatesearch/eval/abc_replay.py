from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from openstatesearch.agent.schemas import (
    Action,
    ActionValidationError,
    KeepAction,
    OpenAction,
    SearchAction,
    VerifyAction,
    parse_action,
)
from openstatesearch.rewards import (
    ABCCreditConfig,
    EvidenceCreditTracker,
    combine_group_advantages,
)


@dataclass(frozen=True)
class ReplayedEpisode:
    question: str
    terminal_reward: float
    process_rewards: tuple[float, ...]
    actions: tuple[str, ...]
    final_transition_unobserved: bool


def parse_policy_input(prompt: str) -> dict[str, Any]:
    """Extract the JSON user observation from an AReaL HF chat prompt."""

    marker = "<|im_start|>user\n"
    start = prompt.rfind(marker)
    if start < 0:
        raise ValueError("prompt has no Qwen user-message marker")
    start += len(marker)
    end = prompt.find("<|im_end|>", start)
    if end < 0:
        raise ValueError("prompt has no closing Qwen message marker")
    value = json.loads(prompt[start:end])
    if not isinstance(value, dict) or "state" not in value:
        raise ValueError("prompt user message is not an OpenStateSearch observation")
    return value


def _completion_action(completion: str) -> Action | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(completion):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(completion[start:])
            return parse_action(value)
        except (json.JSONDecodeError, ActionValidationError, TypeError):
            continue
    return None


def _action_from_result(completion: str, result: dict[str, Any]) -> Action | None:
    parsed = _completion_action(completion)
    if parsed is not None:
        return parsed
    if not result.get("ok"):
        return None
    name = result.get("action")
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if name == "SEARCH":
        return SearchAction(
            str(payload.get("query", "replayed")), str(payload.get("target", "replayed"))
        )
    if name == "VERIFY":
        return VerifyAction(
            str(payload.get("target", "replayed")), str(payload.get("query", "replayed"))
        )
    if name == "OPEN" and payload.get("doc_id"):
        return OpenAction(str(payload["doc_id"]))
    if name == "KEEP" and payload.get("doc_id") and payload.get("sent_ids"):
        return KeepAction(
            str(payload["doc_id"]),
            tuple(int(value) for value in payload["sent_ids"]),
            str(payload.get("claim", "replayed")),
            str(payload.get("constraint_id", "replayed")),
        )
    return None


def segment_rollout_rows(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split one grouped dump file into its per-rollout interaction sequences."""

    episodes: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["sample_idx"])):
        policy_input = parse_policy_input(str(row["prompt"]))
        if policy_input.get("last_tool_result") is None and current:
            episodes.append(current)
            current = []
        current.append({**row, "_policy_input": policy_input})
    if current:
        episodes.append(current)
    return episodes


def replay_episode(
    rows: list[dict[str, Any]],
    gold_evidence: Iterable[tuple[str, int]],
    config: ABCCreditConfig | None = None,
) -> ReplayedEpisode:
    if not rows:
        raise ValueError("cannot replay an empty episode")
    tracker = EvidenceCreditTracker(gold_evidence, config)
    rewards: list[float] = []
    actions: list[str] = []
    for index, row in enumerate(rows[:-1]):
        next_input = rows[index + 1]["_policy_input"]
        result = next_input.get("last_tool_result")
        if not isinstance(result, dict):
            raise ValueError("non-final interaction has no observable tool result")
        action = _action_from_result(str(row.get("completion", "")), result)
        transition = tracker.score(action, result)
        rewards.append(transition.process_reward)
        actions.append(transition.action)
    first_input = rows[0]["_policy_input"]
    question = str(first_input.get("state", {}).get("question", ""))
    terminal_reward = float(rows[-1].get("original_reward", rows[-1].get("reward", 0.0)))
    return ReplayedEpisode(
        question=question,
        terminal_reward=terminal_reward,
        process_rewards=tuple(rewards),
        actions=tuple(actions),
        final_transition_unobserved=True,
    )


def load_gold_by_question(dataset_path: str | Path) -> dict[str, tuple[tuple[str, int], ...]]:
    result: dict[str, tuple[tuple[str, int], ...]] = {}
    with Path(dataset_path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            refs = tuple(
                (str(item["doc_id"]), int(sent_id))
                for item in row.get("gold_evidence", [])
                for sent_id in item.get("sent_ids", [])
            )
            result[str(row["question"])] = refs
    return result


def build_abc_replay_report(
    rollout_root: str | Path,
    dataset_path: str | Path,
    config: ABCCreditConfig | None = None,
) -> dict[str, Any]:
    config = config or ABCCreditConfig()
    gold_by_question = load_gold_by_question(dataset_path)
    groups: list[list[ReplayedEpisode]] = []
    missing_questions: set[str] = set()
    source_files = sorted(Path(rollout_root).glob("*/*.jsonl"))
    for path in source_files:
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        group: list[ReplayedEpisode] = []
        for episode_rows in segment_rollout_rows(rows):
            question = str(episode_rows[0]["_policy_input"]["state"]["question"])
            gold = gold_by_question.get(question)
            if gold is None:
                missing_questions.add(question)
                continue
            group.append(replay_episode(episode_rows, gold, config))
        if group:
            groups.append(group)

    episodes = [episode for group in groups for episode in group]
    process_values = [value for episode in episodes for value in episode.process_rewards]
    nonzero_episodes = [episode for episode in episodes if any(episode.process_rewards)]
    failed = [episode for episode in episodes if episode.terminal_reward < 0.0]
    failed_with_positive_prefix = [
        episode for episode in failed if any(value > 0.0 for value in episode.process_rewards)
    ]
    complete_groups = [group for group in groups if len(group) == 4]
    identical_groups = [
        group
        for group in complete_groups
        if max(episode.terminal_reward for episode in group)
        - min(episode.terminal_reward for episode in group)
        < 1e-8
    ]
    advantages: list[float] = []
    action_advantages: dict[str, list[float]] = {}
    for group in complete_groups:
        group_advantages = combine_group_advantages(
            [episode.terminal_reward for episode in group],
            [episode.process_rewards for episode in group],
            beta=config.beta,
        )
        for episode, row in zip(group, group_advantages, strict=True):
            advantages.extend(row)
            for action, value in zip(episode.actions, row, strict=True):
                action_advantages.setdefault(action, []).append(value)
    action_values: dict[str, list[float]] = {}
    for episode in episodes:
        for action, value in zip(episode.actions, episode.process_rewards):
            action_values.setdefault(action, []).append(value)

    process_returns = [sum(episode.process_rewards) for episode in episodes]

    def correlation(left: list[float], right: list[float]) -> float | None:
        if len(left) != len(right) or not left:
            return None
        left_mean = sum(left) / len(left)
        right_mean = sum(right) / len(right)
        numerator = sum(
            (left_value - left_mean) * (right_value - right_mean)
            for left_value, right_value in zip(left, right, strict=True)
        )
        left_scale = sum((value - left_mean) ** 2 for value in left) ** 0.5
        right_scale = sum((value - right_mean) ** 2 for value in right) ** 0.5
        denominator = left_scale * right_scale
        return numerator / denominator if denominator else None

    def mean_or_none(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "schema_version": 1,
        "config": asdict(config),
        "sources": {
            "rollout_root": str(rollout_root),
            "dataset": str(dataset_path),
            "files": len(source_files),
        },
        "groups": {
            "observed": len(groups),
            "complete_size_4": len(complete_groups),
            "all_identical_terminal": len(identical_groups),
            "all_identical_terminal_ratio": ratio(len(identical_groups), len(complete_groups)),
        },
        "episodes": {
            "count": len(episodes),
            "with_nonzero_process": len(nonzero_episodes),
            "with_nonzero_process_ratio": ratio(len(nonzero_episodes), len(episodes)),
            "terminal_negative": len(failed),
            "failed_with_positive_prefix": len(failed_with_positive_prefix),
            "failed_with_positive_prefix_ratio": ratio(
                len(failed_with_positive_prefix), len(failed)
            ),
            "unobserved_final_transitions": len(episodes),
            "mean_process_return": mean_or_none(process_returns),
            "terminal_process_return_correlation": correlation(
                [episode.terminal_reward for episode in episodes], process_returns
            ),
            "mean_process_return_terminal_negative": mean_or_none(
                [
                    process_return
                    for episode, process_return in zip(episodes, process_returns, strict=True)
                    if episode.terminal_reward < 0.0
                ]
            ),
            "mean_process_return_terminal_nonnegative": mean_or_none(
                [
                    process_return
                    for episode, process_return in zip(episodes, process_returns, strict=True)
                    if episode.terminal_reward >= 0.0
                ]
            ),
        },
        "interactions": {
            "observed": len(process_values),
            "nonzero": sum(value != 0.0 for value in process_values),
            "positive": sum(value > 0.0 for value in process_values),
            "negative": sum(value < 0.0 for value in process_values),
            "nonzero_ratio": ratio(
                sum(value != 0.0 for value in process_values), len(process_values)
            ),
            "training_advantage_nonzero": sum(abs(value) > 1e-8 for value in advantages),
        },
        "by_action": {
            action: {
                "count": len(values),
                "mean_process_reward": sum(values) / len(values),
                "nonzero": sum(value != 0.0 for value in values),
                "training_advantage_count": len(action_advantages.get(action, [])),
                "mean_training_advantage": mean_or_none(action_advantages.get(action, [])),
                "training_advantage_nonzero": sum(
                    abs(value) > 1e-8 for value in action_advantages.get(action, [])
                ),
            }
            for action, values in sorted(action_values.items())
        },
        "missing_dataset_questions": len(missing_questions),
    }
