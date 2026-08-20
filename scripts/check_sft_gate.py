#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.sft_gate import passes_sft_gate, sft_gate_metrics
from openstatesearch.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce all four SFT go-gate thresholds")
    parser.add_argument("--records", required=True)
    parser.add_argument("--config", default="configs/policy_sft.yaml")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line
    ]
    config = load_config(args.config)
    metrics = sft_gate_metrics(records)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if not passes_sft_gate(metrics, config["go_gate"]):
        raise SystemExit("SFT Go Gate failed; RL must not start")


if __name__ == "__main__":
    main()
