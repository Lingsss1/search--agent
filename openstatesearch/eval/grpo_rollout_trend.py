from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openstatesearch.eval.full_reward_audit import load_rollout_trajectories


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_metrics(
    version: int,
    trajectories: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    *,
    observed_episodes: int,
    observed_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = [float(item["original_reward"]) for item in trajectories]
    interactions = [int(item["interaction_count"]) for item in trajectories]
    # Phase-A invalid trajectories are exactly -1.0 minus a [0, 0.5]
    # protocol penalty. A valid Phase-A trajectory is bounded below by -0.65,
    # so reward > -1 is an exact validity classifier for this phase.
    valid_count = sum(reward > -1.0 for reward in rewards)
    single_invalid = sum(
        reward <= -1.0 and count == 1 for reward, count in zip(rewards, interactions, strict=True)
    )
    source_payload = json.dumps(
        sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    observed_source_payload = json.dumps(
        observed_sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "version": version,
        "episodes": len(trajectories),
        "observed_episodes": observed_episodes,
        "discarded_retry_episodes": observed_episodes - len(trajectories),
        "valid_count": valid_count,
        "valid_rate": valid_count / len(trajectories),
        "positive_reward_count": sum(reward > 0 for reward in rewards),
        "mean_reward": sum(rewards) / len(rewards),
        "mean_interactions": sum(interactions) / len(interactions),
        "single_interaction_invalid_count": single_invalid,
        "single_interaction_invalid_rate": single_invalid / len(trajectories),
        "source_files": len(sources),
        "source_manifest_sha256": hashlib.sha256(source_payload).hexdigest(),
        "observed_source_files": len(observed_sources),
        "observed_source_manifest_sha256": hashlib.sha256(observed_source_payload).hexdigest(),
    }


def select_latest_complete_rollout_batch(
    trajectories: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    expected_episodes: int,
    rollouts_per_source: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the latest complete retry batch without double-counting recovery dumps."""
    if len(trajectories) < expected_episodes:
        raise ValueError(f"observed {len(trajectories)} episodes; expected {expected_episodes}")
    if expected_episodes % rollouts_per_source:
        raise ValueError(
            f"expected {expected_episodes} episodes is not divisible by "
            f"{rollouts_per_source} rollouts per source"
        )

    def source_key(source: dict[str, Any]) -> str:
        return str(source.get("source_key", source["path"]))

    def trajectory_source_key(trajectory: dict[str, Any]) -> str:
        return str(trajectory.get("rollout_source_key", trajectory["rollout_file"]))

    def source_order(source: dict[str, Any]) -> tuple[int, int, str, str]:
        path = Path(str(source["path"]))
        return (
            int(source.get("mtime_ns", path.stat().st_mtime_ns)),
            int(source.get("batch_index", 0)),
            str(path),
            source_key(source),
        )

    by_source: dict[str, list[dict[str, Any]]] = {}
    for trajectory in trajectories:
        by_source.setdefault(trajectory_source_key(trajectory), []).append(trajectory)
    complete_sources = [
        source
        for source in sources
        if len(by_source.get(source_key(source), [])) == rollouts_per_source
    ]
    required_sources = expected_episodes // rollouts_per_source
    # A retry may append another four-sample batch to the same task file. Keep
    # only the newest complete virtual source for each task before selecting the
    # latest cross-task batch, otherwise an older and newer batch for one prompt
    # could both be counted.
    latest_by_task: dict[int, dict[str, Any]] = {}
    for source in complete_sources:
        task_id = int(source["task_id"])
        previous = latest_by_task.get(task_id)
        if previous is None or source_order(source) > source_order(previous):
            latest_by_task[task_id] = source
    candidate_sources = list(latest_by_task.values())
    if len(candidate_sources) < required_sources:
        raise ValueError(
            f"only {len(candidate_sources)} latest complete task sources; need {required_sources}"
        )
    selected_sources = sorted(
        candidate_sources,
        key=source_order,
    )[-required_sources:]
    selected_keys = {source_key(source) for source in selected_sources}
    selected = [
        trajectory
        for trajectory in trajectories
        if trajectory_source_key(trajectory) in selected_keys
    ]
    if len(selected) != expected_episodes:
        raise ValueError(
            f"latest complete rollout batch has {len(selected)} episodes; "
            f"expected {expected_episodes}"
        )
    return selected, selected_sources


def _window(values: list[dict[str, Any]], first: int, last: int) -> dict[str, Any]:
    selected = [item for item in values if first <= int(item["version"]) <= last]
    episodes = sum(int(item["episodes"]) for item in selected)
    valid = sum(int(item["valid_count"]) for item in selected)
    single_invalid = sum(int(item["single_interaction_invalid_count"]) for item in selected)
    return {
        "first_version": first,
        "last_version": last,
        "versions": len(selected),
        "episodes": episodes,
        "valid_count": valid,
        "valid_rate": valid / episodes,
        "single_interaction_invalid_count": single_invalid,
        "single_interaction_invalid_rate": single_invalid / episodes,
        "mean_reward": sum(float(item["mean_reward"]) * int(item["episodes"]) for item in selected)
        / episodes,
        "mean_interactions": sum(
            float(item["mean_interactions"]) * int(item["episodes"]) for item in selected
        )
        / episodes,
    }


def summarize_phase_a_rollouts(
    *,
    rollout_root: str | Path,
    through_version: int,
    output_path: str | Path,
    start_version: int = 1,
    expected_episodes_per_version: int = 64,
    window_size: int = 25,
) -> dict[str, Any]:
    if (
        start_version < 0
        or through_version < start_version
        or expected_episodes_per_version < 1
        or window_size < 1
    ):
        raise ValueError("versions, episode count, and window size must be positive")
    values = []
    for version in range(start_version, through_version + 1):
        observed, observed_sources = load_rollout_trajectories(rollout_root, version)
        if len(observed) < expected_episodes_per_version:
            raise ValueError(
                f"version {version} has {len(observed)} episodes; "
                f"expected {expected_episodes_per_version}"
            )
        trajectories, sources = select_latest_complete_rollout_batch(
            observed,
            observed_sources,
            expected_episodes_per_version,
        )
        values.append(
            _version_metrics(
                version,
                trajectories,
                sources,
                observed_episodes=len(observed),
                observed_sources=observed_sources,
            )
        )
    windows = [
        _window(values, first, min(first + window_size - 1, through_version))
        for first in range(start_version, through_version + 1, window_size)
    ]
    result = {
        "schema_version": 1,
        "phase": "A",
        "start_version": start_version,
        "through_version": through_version,
        "expected_episodes_per_version": expected_episodes_per_version,
        "retry_selection": (
            "when a recovered policy version contains multiple dump batches, "
            "select the latest complete source-file batch by preserved mtime and "
            "record all discarded retry episodes"
        ),
        "validity_method": (
            "reward > -1.0; exact for the frozen Phase-A reward because invalid "
            "is [-1.5,-1.0] and valid is >= -0.65"
        ),
        "rollout_root": str(rollout_root),
        "versions": values,
        "windows": windows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "output": str(destination),
        "sha256": _sha256(destination),
        "start_version": start_version,
        "through_version": through_version,
        "episodes": (through_version - start_version + 1) * expected_episodes_per_version,
    }
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result
