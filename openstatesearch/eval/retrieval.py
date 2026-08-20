from __future__ import annotations

import math
from typing import Iterable


def recall_at_k(ranking: Iterable[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    return len(set(list(ranking)[:k]) & relevant_set) / len(relevant_set)


def ndcg_at_k(ranking: Iterable[str], relevance: dict[str, float], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    ranking = list(ranking)[:k]
    dcg = sum(
        (2 ** relevance.get(doc_id, 0.0) - 1) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranking)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 1.0
