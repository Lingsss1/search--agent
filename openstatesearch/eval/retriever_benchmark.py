from __future__ import annotations

from statistics import mean
from typing import Any

from .retrieval import ndcg_at_k, recall_at_k


def evaluate_retriever(retriever: Any, examples: list[dict[str, Any]]) -> dict[str, float | int]:
    if not examples:
        raise ValueError("retrieval benchmark cannot be empty")
    recalls: dict[int, list[float]] = {5: [], 20: [], 100: []}
    ndcgs: list[float] = []
    if hasattr(retriever, "search_batch"):
        rankings = retriever.search_batch([str(example["query"]) for example in examples], 100)
    else:
        rankings = [retriever.search(str(example["query"]), 100) for example in examples]
    for example, hits in zip(examples, rankings):
        relevant = [str(value) for value in example["relevant_doc_ids"]]
        ranking = [hit.doc_id for hit in hits]
        for k in recalls:
            recalls[k].append(recall_at_k(ranking, relevant, k))
        ndcgs.append(ndcg_at_k(ranking, {doc_id: 1.0 for doc_id in relevant}, 10))
    return {
        "examples": len(examples),
        "recall_at_5": mean(recalls[5]),
        "recall_at_20": mean(recalls[20]),
        "recall_at_100": mean(recalls[100]),
        "ndcg_at_10": mean(ndcgs),
    }
