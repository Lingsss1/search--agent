from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    lowered = text.casefold()
    without_punctuation = "".join(" " if char in string.punctuation else char for char in lowered)
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def answer_exact_match(prediction: str, references: Iterable[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(reference) for reference in references))


def _single_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_f1(prediction: str, references: Iterable[str]) -> float:
    references = list(references)
    if not references:
        raise ValueError("at least one reference answer is required")
    return max(_single_f1(prediction, reference) for reference in references)


EvidenceRef = tuple[str, int]


def evidence_precision_recall_f1(
    predicted: Iterable[EvidenceRef], gold: Iterable[EvidenceRef]
) -> tuple[float, float, float]:
    predicted_set, gold_set = set(predicted), set(gold)
    true_positive = len(predicted_set & gold_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(gold_set) if gold_set else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def duplicate_rate(queries: Iterable[str], returned_doc_ids: Iterable[Iterable[str]]) -> float:
    query_list = [" ".join(query.casefold().split()) for query in queries]
    result_list = [tuple(items) for items in returned_doc_ids]
    total = max(len(query_list), len(result_list))
    if total == 0:
        return 0.0
    duplicates = 0
    seen_queries: set[str] = set()
    seen_results: set[tuple[str, ...]] = set()
    for index in range(total):
        query = query_list[index] if index < len(query_list) else ""
        result = result_list[index] if index < len(result_list) else ()
        if (query and query in seen_queries) or (result and result in seen_results):
            duplicates += 1
        seen_queries.add(query)
        seen_results.add(result)
    return duplicates / total
