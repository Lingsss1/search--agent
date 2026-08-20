#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.data.corpus import chunk_text, write_jsonl, write_manifest
from openstatesearch.retriever.types import Document


def rank(seed: int, dataset: str, split: str, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{dataset}:{split}:{identity}".encode()).digest()


def select(
    values: Iterable[dict[str, Any]], count: int, seed: int, dataset: str, split: str
) -> list[dict[str, Any]]:
    ranked = []
    for index, value in enumerate(values):
        identity = str(value.get("id") or value.get("_id") or index)
        ranked.append((rank(seed, dataset, split, identity), index, value))
    if len(ranked) < count:
        raise ValueError(f"{dataset}/{split}: requested {count}, found {len(ranked)}")
    return [value for _, _, value in heapq.nsmallest(count, ranked)]


def hotpot_rows(root: Path, split: str) -> list[dict[str, Any]]:
    paths = sorted((root / "distractor").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no HotpotQA {split} parquet files in {root}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def json_array(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON array")
    return value


def json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    return " ".join(str(text).split())


class Builder:
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}

    def add_paragraph(self, source: str, title: str, text: str) -> list[Document]:
        result = []
        for chunk_index, chunk in enumerate(chunk_text(text, 384, 64)):
            document = Document.create(
                normalize(title),
                chunk,
                source,
                chunk_index=chunk_index,
                parent_title=normalize(title),
            )
            self.documents.setdefault(document.doc_id, document)
            result.append(document)
        return result

    @staticmethod
    def matching_chunks(documents: list[Document], sentence: str | None) -> list[Document]:
        if not documents:
            return []
        if not sentence:
            return documents
        needle = normalize(sentence)
        exact = [document for document in documents if needle in normalize(document.text)]
        if exact:
            return exact
        prefix = " ".join(needle.split()[:12])
        approximate = [
            document for document in documents if prefix and prefix in normalize(document.text)
        ]
        return approximate or documents[:1]


def hotpot_example(builder: Builder, value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = "hotpotqa"
    contexts = value["context"]
    by_title: dict[str, list[Document]] = {}
    sentences_by_title: dict[str, list[str]] = {}
    for title, sentences in zip(contexts["title"], contexts["sentences"]):
        clean_title = normalize(title)
        sentences_by_title[clean_title] = [normalize(sentence) for sentence in sentences]
        by_title.setdefault(clean_title, []).extend(
            builder.add_paragraph(source, clean_title, " ".join(sentences))
        )
    relevant: set[str] = set()
    evidence: dict[str, set[int]] = {}
    facts = value["supporting_facts"]
    for title, sentence_id in zip(facts["title"], facts["sent_id"]):
        clean_title = normalize(title)
        original = sentences_by_title.get(clean_title, [])
        sentence = original[int(sentence_id)] if int(sentence_id) < len(original) else None
        for document in builder.matching_chunks(by_title.get(clean_title, []), sentence):
            relevant.add(document.doc_id)
            local = 0
            if sentence:
                for index, candidate in enumerate(document.sentences):
                    if normalize(sentence) in normalize(candidate) or normalize(
                        candidate
                    ) in normalize(sentence):
                        local = index
                        break
            evidence.setdefault(document.doc_id, set()).add(local)
    record = {
        "id": str(value["id"]),
        "dataset": source,
        "question": str(value["question"]),
        "answer": str(value["answer"]),
        "answers": [str(value["answer"])],
        "messages": [{"role": "user", "content": str(value["question"])}],
        "constraints": [],
        "gold_evidence": [
            {"doc_id": doc_id, "sent_ids": sorted(ids)} for doc_id, ids in sorted(evidence.items())
        ],
    }
    return record, sorted(relevant)


def add_hotpot_context(builder: Builder, value: dict[str, Any]) -> None:
    contexts = value["context"]
    for title, sentences in zip(contexts["title"], contexts["sentences"]):
        builder.add_paragraph("hotpotqa", str(title), " ".join(sentences))


def wiki_example(builder: Builder, value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = "2wiki"
    by_title: dict[str, list[Document]] = {}
    sentences_by_title: dict[str, list[str]] = {}
    for title, sentences in value["context"]:
        clean_title = normalize(title)
        sentences_by_title[clean_title] = [normalize(sentence) for sentence in sentences]
        by_title.setdefault(clean_title, []).extend(
            builder.add_paragraph(source, clean_title, " ".join(sentences))
        )
    relevant: set[str] = set()
    evidence: dict[str, set[int]] = {}
    for title, sentence_id in value["supporting_facts"]:
        clean_title = normalize(title)
        original = sentences_by_title.get(clean_title, [])
        sentence = original[int(sentence_id)] if int(sentence_id) < len(original) else None
        for document in builder.matching_chunks(by_title.get(clean_title, []), sentence):
            relevant.add(document.doc_id)
            local = 0
            if sentence:
                for index, candidate in enumerate(document.sentences):
                    if normalize(sentence) in normalize(candidate) or normalize(
                        candidate
                    ) in normalize(sentence):
                        local = index
                        break
            evidence.setdefault(document.doc_id, set()).add(local)
    record = {
        "id": str(value["_id"]),
        "dataset": source,
        "question": str(value["question"]),
        "answer": str(value["answer"]),
        "answers": [str(value["answer"])],
        "messages": [{"role": "user", "content": str(value["question"])}],
        "constraints": [],
        "gold_evidence": [
            {"doc_id": doc_id, "sent_ids": sorted(ids)} for doc_id, ids in sorted(evidence.items())
        ],
    }
    return record, sorted(relevant)


def add_wiki_context(builder: Builder, value: dict[str, Any]) -> None:
    for title, sentences in value["context"]:
        builder.add_paragraph("2wiki", str(title), " ".join(sentences))


def musique_example(builder: Builder, value: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    source = "musique"
    relevant: set[str] = set()
    evidence: dict[str, set[int]] = {}
    for paragraph in value["paragraphs"]:
        documents = builder.add_paragraph(
            source, str(paragraph["title"]), str(paragraph["paragraph_text"])
        )
        if paragraph.get("is_supporting"):
            for document in documents:
                relevant.add(document.doc_id)
                evidence.setdefault(document.doc_id, set()).add(0)
    answers = [str(value["answer"])] + [str(item) for item in value.get("answer_aliases", [])]
    answers = list(dict.fromkeys(item for item in answers if item))
    record = {
        "id": str(value["id"]),
        "dataset": source,
        "question": str(value["question"]),
        "answer": str(value["answer"]),
        "answers": answers,
        "messages": [{"role": "user", "content": str(value["question"])}],
        "constraints": [],
        "gold_evidence": [
            {"doc_id": doc_id, "sent_ids": sorted(ids)} for doc_id, ids in sorted(evidence.items())
        ],
    }
    return record, sorted(relevant)


def add_musique_context(builder: Builder, value: dict[str, Any]) -> None:
    for paragraph in value["paragraphs"]:
        builder.add_paragraph("musique", str(paragraph["title"]), str(paragraph["paragraph_text"]))


def write_records(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze in-domain RL and retriever dev datasets")
    parser.add_argument("--hotpot-root", default="data/raw/HotpotQA/hf_dataset")
    parser.add_argument("--wiki-root", default="data/raw/2WikiMultiHopQA/data")
    parser.add_argument("--musique-root", default="data/raw/MuSiQue-Ans/data")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=36)
    args = parser.parse_args()
    output = Path(args.output_dir)
    builder = Builder()

    sources = {
        "hotpotqa": {
            "train": hotpot_rows(Path(args.hotpot_root), "train"),
            "dev": hotpot_rows(Path(args.hotpot_root), "validation"),
            "train_count": 2500,
            "convert": hotpot_example,
            "add_context": add_hotpot_context,
        },
        "2wiki": {
            "train": json_array(Path(args.wiki_root) / "train.json"),
            "dev": json_array(Path(args.wiki_root) / "dev.json"),
            "train_count": 2500,
            "convert": wiki_example,
            "add_context": add_wiki_context,
        },
        "musique": {
            "train": json_lines(Path(args.musique_root) / "musique_ans_v1.0_train.jsonl"),
            "dev": json_lines(Path(args.musique_root) / "musique_ans_v1.0_dev.jsonl"),
            "train_count": 1000,
            "convert": musique_example,
            "add_context": add_musique_context,
        },
    }
    rl_train: list[dict[str, Any]] = []
    rl_dev: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    # The frozen retrieval universe is the deduplicated union of every
    # train/dev context. It is deliberately built before selecting RL prompts;
    # per-question candidate contexts must never define the search universe.
    for dataset, specification in sources.items():
        add_context = specification["add_context"]
        seen = 0
        for split in ("train", "dev"):
            for value in specification[split]:
                add_context(builder, value)
                seen += 1
                if seen % 25000 == 0:
                    print(
                        json.dumps(
                            {
                                "dataset": dataset,
                                "contexts_processed": seen,
                                "unique_documents": len(builder.documents),
                            }
                        ),
                        flush=True,
                    )
    for dataset, specification in sources.items():
        train = select(
            specification["train"], specification["train_count"], args.seed, dataset, "train"
        )
        dev = select(specification["dev"], 500, args.seed, dataset, "dev")
        counts[dataset] = {
            "available_train": len(specification["train"]),
            "available_dev": len(specification["dev"]),
            "selected_train": len(train),
            "selected_dev": len(dev),
        }
        convert = specification["convert"]
        for value in train:
            record, _ = convert(builder, value)
            rl_train.append(record)
        for value in dev:
            record, relevant = convert(builder, value)
            rl_dev.append(record)
            queries.append(
                {
                    "id": record["id"],
                    "dataset": dataset,
                    "query": record["question"],
                    "answer": record["answer"],
                    "relevant_doc_ids": relevant,
                }
            )

    documents = sorted(builder.documents.values(), key=lambda document: document.doc_id)
    corpus_path = output / "in_domain_corpus.jsonl"
    corpus_sha = write_jsonl(corpus_path, documents)
    write_manifest(output / "in_domain_corpus.manifest.json", corpus_path, documents, corpus_sha)
    digests = {
        "rl_train": write_records(output / "rl_train.jsonl", rl_train),
        "rl_dev": write_records(output / "rl_dev.jsonl", rl_dev),
        "retriever_dev_queries": write_records(output / "retriever_dev_queries.jsonl", queries),
    }
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "selection": "smallest sha256(seed:dataset:split:id)",
        "corpus_scope": "deduplicated union of all train and dev contexts",
        "chunking": {"size": 384, "overlap": 64},
        "counts": counts,
        "documents": len(documents),
        "corpus_sha256": corpus_sha,
        "outputs_sha256": digests,
    }
    (output / "in_domain_splits.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
