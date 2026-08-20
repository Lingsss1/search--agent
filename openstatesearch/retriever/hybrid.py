from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from .bm25 import BM25Retriever, tokenize
from .types import Document, SearchHit


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


class InMemoryDenseRetriever:
    """Deterministic local encoder used for smoke tests; production swaps in LRAT embeddings."""

    def __init__(self, documents: list[Document], dimensions: int = 384):
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = {doc.doc_id: doc for doc in documents}
        self.dimensions = dimensions
        self._vectors = {doc.doc_id: self._encode(f"{doc.title} {doc.text}") for doc in documents}

    def _encode(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            vector[index] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(item * item for item in vector)) or 1.0
        return tuple(item / norm for item in vector)

    def search(self, query: str, k: int = 100) -> list[SearchHit]:
        query_vector = self._encode(query)
        ranked = sorted(
            (
                (sum(a * b for a, b in zip(query_vector, vector)), doc_id)
                for doc_id, vector in self._vectors.items()
            ),
            key=lambda item: (-item[0], item[1]),
        )
        return [
            SearchHit(
                doc_id,
                self.documents[doc_id].title,
                score,
                self.documents[doc_id].text[:512],
                self.documents[doc_id].source,
            )
            for score, doc_id in ranked[: max(0, k)]
            if score > 0
        ]

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def search_batch(self, queries: list[str], k: int = 5) -> list[list[SearchHit]]:
        return [self.search(query, k) for query in queries]


class HybridRetriever:
    """BM25 + Dense retrieval fused using RRF, with stable tie-breaking."""

    def __init__(
        self,
        documents: list[Document],
        dense: object | None = None,
        candidate_k: int = 100,
        rrf_k: int = 60,
    ):
        self.documents = {doc.doc_id: doc for doc in documents}
        self.bm25 = BM25Retriever(documents)
        self.dense = dense or InMemoryDenseRetriever(documents)
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        sparse = self.bm25.search(query, self.candidate_k)
        dense = self.dense.search(query, self.candidate_k)
        scores = reciprocal_rank_fusion(
            [[hit.doc_id for hit in sparse], [hit.doc_id for hit in dense]], self.rrf_k
        )
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:k]
        return [
            SearchHit(
                doc_id,
                self.documents[doc_id].title,
                scores[doc_id],
                self.documents[doc_id].text[:512],
                self.documents[doc_id].source,
            )
            for doc_id in ranked
        ]

    def search_batch(self, queries: list[str], k: int = 5) -> list[list[SearchHit]]:
        """Batch both retrieval branches before applying per-query deterministic RRF."""
        sparse_rankings = self.bm25.search_batch(queries, self.candidate_k)
        if hasattr(self.dense, "search_batch"):
            dense_rankings = self.dense.search_batch(queries, self.candidate_k)
        else:
            dense_rankings = [self.dense.search(query, self.candidate_k) for query in queries]
        outputs = []
        for sparse, dense in zip(sparse_rankings, dense_rankings, strict=True):
            scores = reciprocal_rank_fusion(
                [[hit.doc_id for hit in sparse], [hit.doc_id for hit in dense]], self.rrf_k
            )
            ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:k]
            outputs.append(
                [
                    SearchHit(
                        doc_id,
                        self.documents[doc_id].title,
                        scores[doc_id],
                        self.documents[doc_id].text[:512],
                        self.documents[doc_id].source,
                    )
                    for doc_id in ranked
                ]
            )
        return outputs

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)
