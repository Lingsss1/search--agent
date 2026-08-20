#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.agent.schemas import parse_action
from openstatesearch.training.areal_agent import legal_action_space


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt train-only recovery SFT to legal_action_space"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()

    source = Path(args.input)
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in source.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        payload = json.loads(row["messages"][1]["content"])
        action = json.loads(row["messages"][2]["content"])
        parse_action(action)
        payload["legal_action_space"] = legal_action_space(
            payload["state"],
            payload["opened_doc_ids"],
            int(payload["remaining_turns"]),
            payload.get("last_tool_result"),
        )
        row["messages"][1]["content"] = json.dumps(payload, ensure_ascii=False)
        groups[action["type"]].append(row)

    rng = random.Random(args.seed)
    limits = {"SEARCH": 750, "OPEN": 1500, "KEEP": 1500, "ANSWER": 1500}
    selected = []
    counts = {}
    for action_type, limit in limits.items():
        values = groups[action_type]
        rng.shuffle(values)
        picked = values[: min(limit, len(values))]
        selected.extend(picked)
        counts[action_type] = len(picked)
    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "held_out_used": False,
        "seed": args.seed,
        "records": len(selected),
        "actions": counts,
        "sha256": digest,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
