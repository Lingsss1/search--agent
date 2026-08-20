from __future__ import annotations

import json
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, build_opener

from .types import Document, SearchHit


class HttpRetriever:
    """Small synchronous client for the shared frozen retriever service."""

    def __init__(self, base_url: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Frozen retriever endpoints are cluster-internal. Environment proxy
        # variables can otherwise route RFC1918 addresses to an external proxy.
        self._opener = build_opener(ProxyHandler({}))

    def _get(self, path: str) -> object:
        with self._opener.open(f"{self.base_url}{path}", timeout=self.timeout) as response:
            return json.loads(response.read())

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        values = self._get(f"/search?{urlencode({'q': query, 'k': k})}")
        if not isinstance(values, list):
            raise RuntimeError("retriever service returned a non-list search result")
        return [SearchHit(**value) for value in values]

    def get_document(self, doc_id: str) -> Document | None:
        try:
            value = self._get(f"/documents/{quote(doc_id, safe='')}")
        except Exception as exc:
            if getattr(exc, "code", None) == 404:
                return None
            raise
        if not isinstance(value, dict):
            raise RuntimeError("retriever service returned a non-object document")
        return Document(**value)
