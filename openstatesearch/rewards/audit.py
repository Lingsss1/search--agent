from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .aggregate import RewardBreakdown


class RewardAuditor:
    """Append-only reward component log used by the fixed 100-step audit gate."""

    def __init__(self, path: str | Path, every_steps: int = 100, sample_size: int = 50):
        if every_steps <= 0 or sample_size <= 0:
            raise ValueError("audit cadence and sample size must be positive")
        self.path = Path(path)
        self.every_steps = every_steps
        self.sample_size = sample_size

    def should_audit(self, step: int) -> bool:
        return step > 0 and step % self.every_steps == 0

    def record(self, step: int, items: Iterable[tuple[str, RewardBreakdown]]) -> int:
        if not self.should_audit(step):
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with self.path.open("a", encoding="utf-8") as handle:
            for trajectory_id, breakdown in items:
                if count >= self.sample_size:
                    break
                payload = {"step": step, "trajectory_id": trajectory_id, **asdict(breakdown)}
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
        return count
