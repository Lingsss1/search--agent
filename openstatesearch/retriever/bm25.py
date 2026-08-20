from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from .types import Document, SearchHit


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


class BM25Retriever:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = {doc.doc_id: doc for doc in documents}
        if len(self.documents) != len(documents):
            raise ValueError("duplicate doc_id in corpus")
        self.k1 = k1
        self.b = b
        self._tf: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        document_frequency: Counter[str] = Counter()
        for doc in documents:
            tokens = tokenize(f"{doc.title} {doc.text}")
            frequencies = Counter(tokens)
            self._tf[doc.doc_id] = frequencies
            self._lengths[doc.doc_id] = len(tokens)
            document_frequency.update(frequencies.keys())
        self._avgdl = sum(self._lengths.values()) / len(documents)
        self._doc_ids = sorted(self.documents)
        self._doc_index = {doc_id: index for index, doc_id in enumerate(self._doc_ids)}
        self._length_array = np.asarray(
            [self._lengths[doc_id] for doc_id in self._doc_ids], dtype=np.float64
        )
        n = len(documents)
        self._idf = {
            token: math.log(1 + (n - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for doc_id, frequencies in self._tf.items():
            for token, frequency in frequencies.items():
                self._postings[token].append((doc_id, frequency))
        self._posting_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for token, postings in self._postings.items():
            indices = np.fromiter(
                (self._doc_index[doc_id] for doc_id, _ in postings),
                dtype=np.int32,
                count=len(postings),
            )
            frequencies = np.fromiter(
                (frequency for _, frequency in postings),
                dtype=np.int32,
                count=len(postings),
            )
            if len(indices) < 2 or bool(np.all(indices[:-1] <= indices[1:])):
                self._posting_arrays[token] = (indices, frequencies)
            else:
                order = np.argsort(indices)
                self._posting_arrays[token] = (indices[order], frequencies[order])
        # The compact arrays are the serving representation. Releasing the
        # per-document Counter graph cuts multi-million-document RSS sharply.
        self._tf.clear()
        self._postings.clear()

    def score(self, query: str, doc_id: str) -> float:
        query_tokens = tokenize(query)
        doc_index = self._doc_index[doc_id]
        doc_length = self._length_array[doc_index]
        score = 0.0
        for token in query_tokens:
            posting = self._posting_arrays.get(token)
            if posting is None:
                continue
            indices, frequencies = posting
            position = int(np.searchsorted(indices, doc_index))
            if position >= len(indices) or int(indices[position]) != doc_index:
                continue
            frequency = float(frequencies[position])
            denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / self._avgdl)
            score += self._idf.get(token, 0.0) * frequency * (self.k1 + 1) / denominator
        return score

    def search(self, query: str, k: int = 100) -> list[SearchHit]:
        if not query.strip() or k <= 0:
            return []
        scores = np.zeros(len(self._doc_ids), dtype=np.float64)
        # This is algebraically identical to score(), but visits only documents
        # containing at least one query token. Duplicate query terms retain the
        # original implementation's contribution multiplicity.
        for token in tokenize(query):
            idf = self._idf.get(token, 0.0)
            posting = self._posting_arrays.get(token)
            if posting is None:
                continue
            indices, frequencies = posting
            denominator = frequencies + self.k1 * (
                1 - self.b + self.b * self._length_array[indices] / self._avgdl
            )
            scores[indices] += idf * frequencies * (self.k1 + 1) / denominator
        candidates = np.flatnonzero(scores > 0)
        limit = min(k, len(candidates))
        if limit == len(candidates):
            selected = candidates
        else:
            candidate_scores = scores[candidates]
            partition = np.argpartition(candidate_scores, -limit)[-limit:]
            threshold = float(candidate_scores[partition].min())
            better = candidates[candidate_scores > threshold]
            # candidates are already in lexicographic doc_id order, so taking
            # the prefix of a score tie exactly matches (-score, doc_id).
            tied = candidates[candidate_scores == threshold]
            selected = np.concatenate((better, tied[: limit - len(better)]))
        order = np.lexsort((selected, -scores[selected]))
        ranked = selected[order]
        hits: list[SearchHit] = []
        for index in ranked:
            doc_id = self._doc_ids[int(index)]
            score = float(scores[index])
            doc = self.documents[doc_id]
            hits.append(SearchHit(doc_id, doc.title, score, doc.text[:512], doc.source))
        return hits

    def get_document(self, doc_id: str) -> Document | None:
        return self.documents.get(doc_id)

    def search_batch(self, queries: list[str], k: int = 100) -> list[list[SearchHit]]:
        return [self.search(query, k) for query in queries]
