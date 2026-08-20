from __future__ import annotations

from openstatesearch.retriever.live_web import (
    LiveWebRetriever,
    build_live_web_provenance,
    parse_duckduckgo_results,
)


SEARCH_HTML = b"""
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=x">Result One</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone&amp;rut=x">Useful <b>snippet</b>.</a>
</div>
"""


def test_duckduckgo_parser_extracts_canonical_result() -> None:
    assert parse_duckduckgo_results(SEARCH_HTML.decode()) == [
        {"title": "Result One", "url": "https://example.com/one", "snippet": "Useful snippet."}
    ]


def test_live_web_retriever_caches_search_and_page(tmp_path) -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if "duckduckgo" in request.full_url:
            return SEARCH_HTML, request.full_url, "text/html; charset=utf-8"
        return (
            b"<html><body><h1>Page title</h1><script>ignore()</script><p>Page evidence.</p></body></html>",
            request.full_url,
            "text/html; charset=utf-8",
        )

    retriever = LiveWebRetriever(tmp_path, opener=opener, min_search_interval=0)
    first = retriever.search("test query", 5)
    second = retriever.search("test query", 5)
    assert first == second
    assert len(calls) == 1
    document = retriever.get_document(first[0].doc_id)
    assert document is not None
    assert "Page evidence." in document.text
    assert "ignore()" not in document.text
    assert len(calls) == 2
    cached = LiveWebRetriever(tmp_path, opener=opener, min_search_interval=0)
    assert cached.search("test query", 5)[0].doc_id == first[0].doc_id
    assert cached.get_document(first[0].doc_id).text == document.text
    assert len(calls) == 2
    manifest = cached.cache_manifest()
    assert manifest["search_snapshots"] == 1
    assert manifest["documents"] == 1
    assert manifest["raw_search_snapshots"] == 1
    assert manifest["raw_pages"] == 1


def test_live_web_provenance_is_stable_and_bound_to_configuration(tmp_path) -> None:
    first = build_live_web_provenance(
        name="live_web_duckduckgo",
        session_label="chinese_seed36",
        cache_dir=tmp_path,
        timeout=20,
        max_document_chars=20_000,
    )
    second = build_live_web_provenance(
        name="live_web_duckduckgo",
        session_label="chinese_seed36",
        cache_dir=tmp_path,
        timeout=20,
        max_document_chars=20_000,
    )
    assert first == second
    assert len(first["provenance_sha256"]) == 64
