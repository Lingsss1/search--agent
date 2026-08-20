from __future__ import annotations

import hashlib
import html
import json
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from .types import Document, SearchHit, stable_doc_id


_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/124.0 Safari/537.36 OpenStateSearch-36/1.0"
)


def build_live_web_provenance(
    *,
    name: str,
    session_label: str,
    cache_dir: str | Path,
    timeout: float,
    max_document_chars: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "backend": "duckduckgo_html",
        "session_label": session_label,
        "cache_dir": str(cache_dir),
        "timeout_seconds": timeout,
        "max_document_chars": max_document_chars,
        "user_agent": _USER_AGENT,
        "implementation_sha256": _sha256_bytes(Path(__file__).read_bytes()),
    }
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    value["provenance_sha256"] = _sha256_bytes(canonical)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def _result_url(href: str) -> str | None:
    value = html.unescape(href)
    if value.startswith("//"):
        value = "https:" + value
    parsed = urllib.parse.urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        value = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return value


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._field: str | None = None
        self._href = ""
        self._parts: list[str] = []
        self._pending: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = {key: value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if "result__a" in classes:
            self._field = "title"
        elif "result__snippet" in classes:
            self._field = "snippet"
        else:
            return
        self._href = values.get("href", "")
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._field:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._field:
            return
        text = " ".join("".join(self._parts).split())
        url = _result_url(self._href)
        if self._field == "title" and url and text:
            self._pending = {"title": text, "url": url, "snippet": ""}
        elif self._field == "snippet" and self._pending:
            if url == self._pending["url"]:
                self._pending["snippet"] = text
                self.results.append(self._pending)
            self._pending = None
        self._field = None
        self._href = ""
        self._parts = []


class _VisibleTextParser(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "canvas", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            value = " ".join(data.split())
            if value:
                self.parts.append(value)


def parse_duckduckgo_results(rendered: str) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser()
    parser.feed(rendered)
    seen: set[str] = set()
    values = []
    for result in parser.results:
        if result["url"] not in seen:
            seen.add(result["url"])
            values.append(result)
    return values


def visible_page_text(rendered: str, max_chars: int = 20_000) -> str:
    parser = _VisibleTextParser()
    parser.feed(rendered)
    return " ".join(parser.parts)[:max_chars]


class LiveWebRetriever:
    """DuckDuckGo-backed retriever with immutable raw snapshots and parsed caches."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        timeout: float = 20.0,
        max_document_chars: int = 20_000,
        min_search_interval: float = 0.5,
        opener: Callable[[urllib.request.Request, float], tuple[bytes, str, str]] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_document_chars = max_document_chars
        self.min_search_interval = min_search_interval
        self._opener = opener or self._open
        self._documents: dict[str, Document] = {}
        self._search_lock = threading.Lock()
        self._documents_lock = threading.Lock()
        self._document_locks_guard = threading.Lock()
        self._document_locks: dict[str, threading.Lock] = {}
        self._last_search = 0.0

    @property
    def documents(self) -> list[Document]:
        with self._documents_lock:
            return list(self._documents.values())

    def _open(self, request: urllib.request.Request, timeout: float) -> tuple[bytes, str, str]:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(4 * 1024 * 1024)
            content_type = response.headers.get_content_type()
            charset = response.headers.get_content_charset() or "utf-8"
            return payload, response.geturl(), f"{content_type}; charset={charset}"

    @staticmethod
    def _decode(payload: bytes, content_type: str) -> str:
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.rsplit("charset=", 1)[1].split(";", 1)[0].strip()
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            return payload.decode("utf-8", errors="replace")

    def _request(self, url: str) -> tuple[bytes, str, str]:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        return self._opener(request, self.timeout)

    def _document_from_record(self, record: dict[str, Any]) -> Document:
        document = Document(
            doc_id=str(record["doc_id"]),
            title=str(record["title"]),
            text=str(record["text"]),
            source="live_web",
            metadata={
                "url": record["url"],
                "captured_at": record.get("captured_at"),
                "raw_sha256": record.get("raw_sha256"),
            },
        )
        with self._documents_lock:
            self._documents[document.doc_id] = document
        return document

    def _document_lock(self, doc_id: str) -> threading.Lock:
        with self._document_locks_guard:
            return self._document_locks.setdefault(doc_id, threading.Lock())

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        cleaned = " ".join(query.split())
        if not cleaned:
            return []
        key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        parsed_path = self.cache_dir / "search" / f"{key}.json"
        with self._search_lock:
            if parsed_path.is_file():
                record = json.loads(parsed_path.read_text(encoding="utf-8"))
            else:
                delay = self.min_search_interval - (time.monotonic() - self._last_search)
                if delay > 0:
                    time.sleep(delay)
                url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": cleaned})
                payload, final_url, content_type = self._request(url)
                self._last_search = time.monotonic()
                raw_path = self.cache_dir / "search" / f"{key}.html"
                _atomic_write(raw_path, payload)
                rendered = self._decode(payload, content_type)
                record = {
                    "schema_version": 1,
                    "query": cleaned,
                    "captured_at": _utc_now(),
                    "request_url": url,
                    "final_url": final_url,
                    "raw_path": str(raw_path),
                    "raw_sha256": _sha256_bytes(payload),
                    "results": parse_duckduckgo_results(rendered),
                }
                _atomic_write(
                    parsed_path,
                    (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(),
                )
            hits = []
            for rank, result in enumerate(record["results"][:k]):
                doc_id = stable_doc_id("live_web", result["url"], result["url"])
                document_path = self.cache_dir / "documents" / f"{doc_id}.json"
                if document_path.is_file():
                    document_record = json.loads(document_path.read_text(encoding="utf-8"))
                else:
                    document_record = {
                        "schema_version": 1,
                        "doc_id": doc_id,
                        "title": result["title"],
                        "url": result["url"],
                        "text": result.get("snippet", ""),
                        "snippet": result.get("snippet", ""),
                        "fetched": False,
                    }
                    _atomic_write(
                        document_path,
                        (json.dumps(document_record, ensure_ascii=False, indent=2) + "\n").encode(),
                    )
                self._document_from_record(document_record)
                hits.append(
                    SearchHit(
                        doc_id=doc_id,
                        title=result["title"],
                        score=1.0 / (rank + 1),
                        snippet=result.get("snippet", ""),
                        source="live_web",
                    )
                )
            return hits

    def get_document(self, doc_id: str) -> Document | None:
        path = self.cache_dir / "documents" / f"{doc_id}.json"
        with self._document_lock(doc_id):
            if not path.is_file():
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            if not record.get("fetched"):
                try:
                    payload, final_url, content_type = self._request(str(record["url"]))
                    raw_path = self.cache_dir / "pages" / f"{doc_id}.raw"
                    _atomic_write(raw_path, payload)
                    if content_type.startswith(("text/", "application/xhtml", "application/json")):
                        rendered = self._decode(payload, content_type)
                        page_text = visible_page_text(rendered, self.max_document_chars)
                    else:
                        page_text = ""
                    record.update(
                        {
                            "text": " ".join(
                                value for value in (record.get("snippet", ""), page_text) if value
                            )[: self.max_document_chars],
                            "fetched": True,
                            "captured_at": _utc_now(),
                            "final_url": final_url,
                            "content_type": content_type,
                            "raw_path": str(raw_path),
                            "raw_sha256": _sha256_bytes(payload),
                        }
                    )
                except Exception as exc:  # Preserve the search snippet as usable evidence.
                    record.update(
                        {
                            "fetched": True,
                            "captured_at": _utc_now(),
                            "fetch_error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                _atomic_write(
                    path,
                    (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode(),
                )
            return self._document_from_record(record)

    def cache_manifest(self, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        search_records = sorted((self.cache_dir / "search").glob("*.json"))
        raw_search = sorted((self.cache_dir / "search").glob("*.html"))
        documents = sorted((self.cache_dir / "documents").glob("*.json"))
        raw_pages = sorted((self.cache_dir / "pages").glob("*.raw"))
        value = {
            "schema_version": 1,
            "backend": "duckduckgo_html",
            "cache_dir": str(self.cache_dir),
            "search_snapshots": len(search_records),
            "documents": len(documents),
            "raw_search_snapshots": len(raw_search),
            "raw_pages": len(raw_pages),
            "search_snapshot_sha256": {
                path.name: _sha256_bytes(path.read_bytes()) for path in search_records
            },
            "raw_search_sha256": {
                path.name: _sha256_bytes(path.read_bytes()) for path in raw_search
            },
            "document_sha256": {path.name: _sha256_bytes(path.read_bytes()) for path in documents},
            "raw_page_sha256": {path.name: _sha256_bytes(path.read_bytes()) for path in raw_pages},
        }
        if provenance is not None:
            value["retriever_provenance"] = provenance
        return value
