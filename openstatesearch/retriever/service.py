from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from openstatesearch.data.corpus import read_jsonl

from .hybrid import HybridRetriever
from .transformer_dense import TransformerDenseRetriever
from .types import Document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *,
    name: str,
    corpus: str | Path,
    dense_model: str | Path | None,
    dense_index: str | Path | None,
    dtype: str,
) -> dict[str, object]:
    corpus_path = Path(corpus)
    model_path = Path(dense_model) if dense_model else None
    index_path = Path(dense_index) if dense_index else None
    model_files = (
        [
            {
                "path": item.name,
                "bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in sorted(model_path.iterdir())
            if item.is_file()
        ]
        if model_path
        else []
    )
    value: dict[str, object] = {
        "schema_version": 1,
        "name": name,
        "corpus": {
            "path": str(corpus_path),
            "bytes": corpus_path.stat().st_size,
            "sha256": _sha256(corpus_path),
        },
        "dense_model": ({"path": str(model_path), "files": model_files} if model_path else None),
        "dense_index": (
            {
                "path": str(index_path),
                "bytes": index_path.stat().st_size,
                "sha256": _sha256(index_path),
                "manifest": (
                    {
                        "path": f"{index_path}.manifest.json",
                        "sha256": _sha256(Path(f"{index_path}.manifest.json")),
                    }
                    if Path(f"{index_path}.manifest.json").is_file()
                    else None
                ),
            }
            if index_path
            else None
        ),
        "dtype": dtype,
    }
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    value["provenance_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def load_corpus(path: str | Path) -> list[Document]:
    documents: list[Document] = []
    for record in read_jsonl(path):
        documents.append(
            Document(
                doc_id=str(record["doc_id"]),
                title=str(record["title"]),
                text=str(record["text"]),
                source=str(record.get("source", "unknown")),
                metadata=dict(record.get("metadata", {})),
            )
        )
    return documents


def make_handler(
    retriever: HybridRetriever, provenance: dict[str, object] | None = None
) -> type[BaseHTTPRequestHandler]:
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: object) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib API
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "documents": len(retriever.documents),
                        "name": provenance.get("name") if provenance else None,
                        "provenance_sha256": (
                            provenance.get("provenance_sha256") if provenance else None
                        ),
                    },
                )
                return
            if parsed.path == "/provenance":
                if provenance is None:
                    self._send(404, {"error": "provenance unavailable"})
                else:
                    self._send(200, provenance)
                return
            if parsed.path == "/search":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0]
                try:
                    k = min(100, max(1, int(params.get("k", ["5"])[0])))
                except ValueError:
                    self._send(400, {"error": "k must be an integer"})
                    return
                with inference_lock:
                    hits = retriever.search(query, k)
                self._send(200, [hit.__dict__ for hit in hits])
                return
            if parsed.path.startswith("/documents/"):
                doc_id = unquote(parsed.path.removeprefix("/documents/"))
                document = retriever.get_document(doc_id)
                if document is None:
                    self._send(404, {"error": "document not found"})
                else:
                    self._send(200, document.__dict__)
                return
            self._send(404, {"error": "unknown endpoint"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a frozen local hybrid retrieval index")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8036)
    parser.add_argument("--name", default="unspecified")
    parser.add_argument("--dense-model", help="self-trained LRAT model/checkpoint path")
    parser.add_argument("--dense-index", help="frozen .npz index built from the same corpus")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    documents = load_corpus(args.corpus)
    if bool(args.dense_model) != bool(args.dense_index):
        parser.error("--dense-model and --dense-index must be supplied together")
    dense = (
        TransformerDenseRetriever(
            documents,
            args.dense_model,
            args.dense_index,
            device=args.device,
            dtype=args.dtype,
        )
        if args.dense_model
        else None
    )
    retriever = HybridRetriever(documents, dense=dense)
    provenance = build_provenance(
        name=args.name,
        corpus=args.corpus,
        dense_model=args.dense_model,
        dense_index=args.dense_index,
        dtype=args.dtype,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(retriever, provenance))
    print(f"serving {len(retriever.documents)} documents on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
