#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.acceptance import audit_acceptance
from openstatesearch.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit final research results against every acceptance criterion"
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--thresholds", default="configs/acceptance.yaml")
    args = parser.parse_args()
    report = audit_acceptance(load_config(args.results), load_config(args.thresholds))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
