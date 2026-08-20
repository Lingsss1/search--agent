from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Any

from openstatesearch.retriever.types import Document


def chunk_text(text: str, size: int = 384, overlap: int = 64) -> list[str]:
    """Whitespace-token chunking matching the frozen 384/64 policy."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("require size > overlap >= 0")
    tokens = text.split()
    if len(tokens) <= size:
        return [" ".join(tokens)] if tokens else []
    chunks: list[str] = []
    step = size - overlap
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + size >= len(tokens):
            break
    return chunks


def build_documents(
    records: Iterable[Mapping[str, Any]], size: int = 384, overlap: int = 64
) -> list[Document]:
    documents: dict[str, Document] = {}
    for record in records:
        title = str(record["title"]).strip()
        text = " ".join(str(record["text"]).split())
        source = str(record.get("source", "unknown")).strip()
        if not title or not text:
            continue
        chunks = chunk_text(text, size, overlap)
        for chunk_index, chunk in enumerate(chunks):
            metadata = {"chunk_index": chunk_index, "parent_title": title}
            document = Document.create(title, chunk, source, **metadata)
            documents.setdefault(document.doc_id, document)
    return sorted(documents.values(), key=lambda document: document.doc_id)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def write_jsonl(path: str | Path, documents: Iterable[Document]) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with destination.open("w", encoding="utf-8") as handle:
        for document in documents:
            line = json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def write_manifest(
    path: str | Path, corpus_path: str | Path, documents: list[Document], sha256: str
) -> None:
    counts: dict[str, int] = {}
    for document in documents:
        counts[document.source] = counts.get(document.source, 0) + 1
    manifest = {
        "schema_version": 1,
        "corpus": str(corpus_path),
        "document_count": len(documents),
        "sha256": sha256,
        "source_counts": dict(sorted(counts.items())),
        "doc_id": "sha1(source|title|normalize(text))",
        "chunking": {"size": 384, "overlap": 64},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
