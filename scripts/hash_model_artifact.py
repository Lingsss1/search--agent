#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.eval.model_provenance import write_model_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash every file in a frozen model artifact exactly once"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    output = args.output or str(Path(args.model) / "model_provenance.json")
    manifest = write_model_provenance(args.model, output)
    print(
        json.dumps(
            {
                "output": output,
                "files": len(manifest["files"]),
                "total_bytes": manifest["total_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
