from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .normalize import jaccard_similarity, normalize_question


@dataclass(frozen=True)
class ContaminationMatch:
    train_index: int
    test_index: int
    reason: str
    similarity: float


def find_contamination(
    train_questions: Iterable[str], test_questions: Iterable[str], threshold: float = 0.8
) -> list[ContaminationMatch]:
    tests = list(test_questions)
    normalized_tests = [normalize_question(question) for question in tests]
    matches: list[ContaminationMatch] = []
    for train_index, train in enumerate(train_questions):
        normalized_train = normalize_question(train)
        for test_index, normalized_test in enumerate(normalized_tests):
            if normalized_train == normalized_test:
                matches.append(ContaminationMatch(train_index, test_index, "exact", 1.0))
                break
            similarity = jaccard_similarity(train, tests[test_index])
            if similarity >= threshold:
                matches.append(
                    ContaminationMatch(train_index, test_index, "near_duplicate", similarity)
                )
                break
    return matches


def write_report(
    path: str | Path, matches: list[ContaminationMatch], source_counts: dict[str, int]
) -> None:
    report = {
        "removed_total": len(matches),
        "source_counts": source_counts,
        "matches": [asdict(match) for match in matches],
    }
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
