from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from openstatesearch.rewards.metrics import (
    answer_exact_match,
    answer_f1,
    evidence_precision_recall_f1,
)


def _evidence(items: list[dict[str, Any]]) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for item in items:
        for sent_id in item.get("sent_ids", []):
            result.add((str(item["doc_id"]), int(sent_id)))
    return result


def _number(record: dict[str, Any], key: str) -> float:
    value = record.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return float(value)


def evaluate_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("predictions file is empty")
    exact_matches: list[float] = []
    answer_f1s: list[float] = []
    citation_precisions: list[float] = []
    evidence_precisions: list[float] = []
    support_recalls: list[float] = []
    support_f1s: list[float] = []
    searches: list[float] = []
    opens: list[float] = []
    input_tokens: list[float] = []
    generated_tokens: list[float] = []
    legal_citations = 0
    correct_citations = 0
    total_citations = 0
    for record in records:
        references = record["answers"]
        if isinstance(references, str):
            references = [references]
        if not isinstance(references, list) or not references:
            raise ValueError("answers must be a non-empty list")
        exact_matches.append(answer_exact_match(record["prediction"], references))
        answer_f1s.append(answer_f1(record["prediction"], references))
        predicted = _evidence(record.get("evidence", []))
        citations = _evidence(record.get("citations", []))
        gold = _evidence(record.get("gold_evidence", []))
        # Citation validity is a protocol property: every cited sentence must be
        # in the mechanically maintained legal evidence set.  It is deliberately
        # distinct from citation precision, which compares citations to gold.
        legal = _evidence(record.get("legal_evidence", record.get("evidence", [])))
        citation_precision, _, _ = evidence_precision_recall_f1(citations, gold)
        evidence_precision, support_recall, support_f1 = evidence_precision_recall_f1(
            predicted, gold
        )
        citation_precisions.append(citation_precision)
        evidence_precisions.append(evidence_precision)
        support_recalls.append(support_recall)
        support_f1s.append(support_f1)
        legal_citations += len(citations & legal)
        correct_citations += len(citations & gold)
        total_citations += len(citations)
        searches.append(_number(record, "search_count"))
        opens.append(_number(record, "open_count"))
        input_tokens.append(_number(record, "input_tokens"))
        generated_tokens.append(_number(record, "generated_tokens"))
    return {
        "examples": len(records),
        "answer_em": mean(exact_matches),
        "answer_f1": mean(answer_f1s),
        "evidence_precision": mean(evidence_precisions),
        "support_recall": mean(support_recalls),
        "support_f1": mean(support_f1s),
        "citation_precision_macro": mean(citation_precisions),
        "citation_precision_micro": (
            correct_citations / total_citations if total_citations else 1.0
        ),
        "citation_validity_micro": (legal_citations / total_citations if total_citations else 1.0),
        "total_citations": total_citations,
        "avg_search": mean(searches),
        "avg_open": mean(opens),
        "avg_input_tokens": mean(input_tokens),
        "avg_generated_tokens": mean(generated_tokens),
        "avg_total_tokens": mean(
            input_value + output_value
            for input_value, output_value in zip(input_tokens, generated_tokens, strict=True)
        ),
    }


def evaluate_by_dataset(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Return overall and per-dataset metrics without hiding missing dataset labels."""
    if not records:
        raise ValueError("predictions file is empty")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset = record.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("every prediction must have a non-empty dataset")
        grouped[dataset].append(record)
    return {
        "overall": evaluate_records(records),
        **{name: evaluate_records(grouped[name]) for name in sorted(grouped)},
    }


def read_predictions(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at line {line_number}")
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen OpenStateSearch trajectories")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--by-dataset",
        action="store_true",
        help="Emit overall and per-dataset metrics.",
    )
    args = parser.parse_args()
    records = read_predictions(args.predictions)
    metrics = evaluate_by_dataset(records) if args.by_dataset else evaluate_records(records)
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
