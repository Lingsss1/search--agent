#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.sources import fetch_source, load_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch datasets at the commits pinned by the source manifest"
    )
    parser.add_argument("--manifest", default="data/manifests/sources.json")
    parser.add_argument("--destination", default="data/raw")
    parser.add_argument("--only", action="append", help="source name; repeat to fetch a subset")
    args = parser.parse_args()
    selected = set(args.only or [])
    sources = load_sources(args.manifest)
    if selected:
        unknown = selected - {source["name"] for source in sources}
        if unknown:
            parser.error(f"unknown source names: {sorted(unknown)}")
        sources = [source for source in sources if source["name"] in selected]
    for source in sources:
        print(f"fetching {source['name']} @ {source['revision']}")
        fetch_source(source, args.destination)


if __name__ == "__main__":
    main()
