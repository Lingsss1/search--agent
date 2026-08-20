from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def select_reward_audit_records(
    records: Iterable[dict[str, Any]], sample_size: int = 50, seed: int = 36
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """Select a stable, bounded trajectory sample independently for each step."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
            raise ValueError(f"reward audit record has invalid step: {step!r}")
        grouped[step].append(record)

    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = {}
    for step in sorted(grouped):
        values = grouped[step]
        counts[step] = len(values)
        ranked = sorted(
            values,
            key=lambda record: (
                hashlib.sha256(f"{seed}:{_canonical(record)}".encode()).hexdigest(),
                _canonical(record),
            ),
        )
        selected.extend(ranked[:sample_size])
    return selected, counts


def normalize_reward_audit(
    source: Path,
    destination: Path,
    summary_path: Path,
    sample_size: int = 50,
    seed: int = 36,
) -> dict[str, Any]:
    """Preserve raw evidence and write an exact, reproducible audit sample."""
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ so raw evidence is preserved")
    records: list[dict[str, Any]] = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object record at {source}:{line_number}")
            records.append(record)

    selected, raw_counts = select_reward_audit_records(records, sample_size, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in selected
    )
    destination.write_text(payload, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "source": str(source),
        "destination": str(destination),
        "seed": seed,
        "sample_size_per_step": sample_size,
        "raw_records": len(records),
        "selected_records": len(selected),
        "steps": {
            str(step): {
                "raw_count": count,
                "selected_count": min(count, sample_size),
            }
            for step, count in sorted(raw_counts.items())
        },
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
