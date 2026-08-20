from __future__ import annotations

import re
import unicodedata


_PUNCTUATION = re.compile(r"[^\w\s\u3400-\u9fff]", re.UNICODE)


def normalize_question(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _PUNCTUATION.sub(" ", text)
    return " ".join(text.split())


def shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    tokens = normalize_question(text).split()
    if not tokens:
        return set()
    if len(tokens) < width:
        return {tuple(tokens)}
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def jaccard_similarity(left: str, right: str) -> float:
    left_set, right_set = shingles(left), shingles(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0
