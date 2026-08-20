from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s*")


def split_sentences(text: str) -> tuple[str, ...]:
    """Split a paragraph while keeping deterministic sentence identifiers."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ()
    parts = tuple(part.strip() for part in _SENTENCE_BOUNDARY.split(cleaned) if part.strip())
    return parts or (cleaned,)


def stable_doc_id(source: str, title: str, text: str) -> str:
    normalized = " ".join(text.split())
    payload = f"{source}|{title}|{normalized}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def sentences(self) -> tuple[str, ...]:
        return split_sentences(self.text)

    @classmethod
    def create(cls, title: str, text: str, source: str = "unknown", **metadata: Any) -> "Document":
        return cls(stable_doc_id(source, title, text), title, text, source, metadata)


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    score: float
    snippet: str
    source: str = "unknown"
