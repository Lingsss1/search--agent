#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.training import load_config, validate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a frozen OpenStateSearch config")
    parser.add_argument("config")
    parser.add_argument("--kind")
    args = parser.parse_args()
    errors = validate_config(load_config(args.config), args.kind)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print(f"valid: {args.config}")


if __name__ == "__main__":
    main()
