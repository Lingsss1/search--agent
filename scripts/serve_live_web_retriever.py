#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.retriever.live_web import (
    LiveWebRetriever,
    build_live_web_provenance,
)
from openstatesearch.retriever.service import make_handler


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve an audited, disk-cached DuckDuckGo live-web retriever"
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument("--name", default="live_web_duckduckgo")
    parser.add_argument("--session-label", default="chinese_seed36")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-document-chars", type=int, default=20_000)
    parser.add_argument("--write-manifest")
    args = parser.parse_args()
    retriever = LiveWebRetriever(
        args.cache_dir,
        timeout=args.timeout,
        max_document_chars=args.max_document_chars,
    )
    provenance = build_live_web_provenance(
        name=args.name,
        session_label=args.session_label,
        cache_dir=args.cache_dir,
        timeout=args.timeout,
        max_document_chars=args.max_document_chars,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(retriever, provenance))
    print(
        json.dumps(
            {
                "status": "serving",
                "host": args.host,
                "port": args.port,
                "cache_dir": args.cache_dir,
                "provenance_sha256": provenance["provenance_sha256"],
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        if args.write_manifest:
            path = Path(args.write_manifest)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    retriever.cache_manifest(provenance),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
