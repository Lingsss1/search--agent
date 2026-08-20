from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
STEP_RE = re.compile(
    r"(?P<timestamp>\d{8}-\d{2}:\d{2}:\d{2}\.\d+) .* "
    r"Step (?P<step>\d+)/(?P<total>\d+) Train step"
)
DEFAULT_METRICS = (
    "timeperf/train_step",
    "ppo_actor/update/actor_loss/avg",
    "ppo_actor/update/grad_norm",
    "ppo_actor/update/rs_filtered_fraction",
    "ppo_actor/update/update_successful",
    "ppo_actor/update/behave_approx_kl/avg",
    "ppo_actor/update/behave_filtered_ratio/avg",
    "ppo_actor/update/behave_imp_weight/avg",
    "ppo_actor/update/behave_imp_weight/min",
    "ppo_actor/update/behave_imp_weight/max",
    "ppo_actor/update/behave_imp_weight_applied/avg",
    "ppo_actor/update/behave_imp_weight_sq/avg",
    "ppo_actor/update/behave_ratio_below_0_5/avg",
    "ppo_actor/update/behave_ratio_above_2/avg",
    "ppo_actor/update/behave_ratio_above_5/avg",
    "ppo_actor/update/behave_k3/avg",
    "ppo_actor/update/compute_logp/kl_div_direct/avg",
    "ppo_actor/update/logp_abs_diff/avg",
    "ppo_actor/update/logp_abs_diff/max",
    "ppo_actor/update/entropy/avg",
    "ppo_actor/update/clip_ratio/avg",
    "ppo_actor/update/masked_token_ratio",
    "ppo_actor/prompt_len/avg",
    "ppo_actor/seq_len/avg",
    "ppo_actor/no_eos_ratios/avg",
    "rollout/answer_f1",
    "rollout/citation_precision",
    "rollout/process_reward_nonzero_ratio",
    "rollout/process_reward_total",
    "rollout/support_recall",
    "rollout/valid_reward",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_training_log(
    path: str | Path, metrics: tuple[str, ...] = DEFAULT_METRICS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(path)
    wanted = set(metrics)
    rows: dict[int, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    total_steps: int | None = None
    with source.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            match = STEP_RE.search(line)
            if match:
                step = int(match.group("step"))
                declared_total = int(match.group("total"))
                if total_steps is not None and total_steps != declared_total:
                    raise ValueError("training log declares inconsistent total step counts")
                total_steps = declared_total
                current = {
                    "step": step,
                    "timestamp": match.group("timestamp"),
                    "metrics": {},
                }
                rows[step] = current
                continue
            if current is None or not line.startswith("│"):
                continue
            cells = [cell.strip() for cell in line.split("│")[1:-1]]
            for index in range(0, len(cells) - 1, 2):
                name, raw_value = cells[index], cells[index + 1]
                if name not in wanted or not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(f"invalid metric {name} at step {current['step']}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"non-finite metric {name} at step {current['step']}")
                current["metrics"][name] = value
    ordered = [rows[step] for step in sorted(rows)]
    if not ordered:
        raise ValueError("training log contains no completed steps")
    for row in ordered:
        row_metrics = row["metrics"]
        first_moment = row_metrics.get("ppo_actor/update/behave_imp_weight_applied/avg")
        second_moment = row_metrics.get("ppo_actor/update/behave_imp_weight_sq/avg")
        if first_moment is not None and second_moment is not None:
            normalized_ess = (
                float(first_moment) ** 2 / float(second_moment)
                if float(second_moment) > 0.0
                else 0.0
            )
            row_metrics["derived/behavior_normalized_effective_sample_size"] = min(
                1.0, max(0.0, normalized_ess)
            )
    gaps = [step for step in range(ordered[0]["step"], ordered[-1]["step"] + 1) if step not in rows]
    metadata = {
        "source": str(source),
        "source_sha256": _sha256(source),
        "declared_total_steps": total_steps,
        "first_completed_step": ordered[0]["step"],
        "last_completed_step": ordered[-1]["step"],
        "completed_steps": len(ordered),
        "missing_steps_in_observed_range": gaps,
    }
    return ordered, metadata


def summarize_training_rows(
    rows: list[dict[str, Any]], window_size: int = 25
) -> list[dict[str, Any]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    windows: list[dict[str, Any]] = []
    for start in range(0, len(rows), window_size):
        group = rows[start : start + window_size]
        names = sorted({name for row in group for name in row.get("metrics", {})})
        aggregate: dict[str, Any] = {}
        for name in names:
            values = [
                float(row["metrics"][name]) for row in group if name in row.get("metrics", {})
            ]
            aggregate[name] = {
                "count": len(values),
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
                "nonzero": sum(value != 0 for value in values),
            }
        windows.append(
            {
                "first_step": group[0]["step"],
                "last_step": group[-1]["step"],
                "steps": len(group),
                "metrics": aggregate,
            }
        )
    return windows
