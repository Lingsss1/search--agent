import json

import pytest

from scripts.prepare_retriever_eval_queries import prepare_retriever_queries


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_prepare_retriever_queries_is_frozen_and_deduplicates_relevance(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "queries.jsonl"
    _write(
        prompts,
        [
            {
                "dataset": "browsecomp_plus",
                "id": "1",
                "question": "question",
                "evidence_doc_ids": ["b", "a", "b"],
            }
        ],
    )
    manifest = prepare_retriever_queries(prompts, output)
    assert manifest["rows"] == 1
    assert manifest["datasets"] == ["browsecomp_plus"]
    assert json.loads(output.read_text())["relevant_doc_ids"] == ["a", "b"]
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_prepare_retriever_queries_rejects_missing_relevance_and_duplicate_id(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "queries.jsonl"
    _write(
        prompts,
        [{"dataset": "d", "id": "1", "question": "q", "evidence_doc_ids": []}],
    )
    with pytest.raises(ValueError, match="no relevance"):
        prepare_retriever_queries(prompts, output)
    row = {"dataset": "d", "id": "1", "question": "q", "evidence_doc_ids": ["x"]}
    _write(prompts, [row, row])
    with pytest.raises(ValueError, match="duplicate"):
        prepare_retriever_queries(prompts, output)
