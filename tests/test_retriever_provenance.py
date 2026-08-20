from __future__ import annotations

import hashlib

from openstatesearch.retriever.service import build_provenance


def test_retriever_provenance_hashes_corpus_model_and_index(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"weights")
    index = tmp_path / "index.npz"
    index.write_bytes(b"index")
    manifest = tmp_path / "index.npz.manifest.json"
    manifest.write_text("{}")

    value = build_provenance(
        name="R4_lrat_hybrid",
        corpus=corpus,
        dense_model=model,
        dense_index=index,
        dtype="bfloat16",
    )

    assert value["name"] == "R4_lrat_hybrid"
    assert value["corpus"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert value["dense_index"]["sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    assert len(value["provenance_sha256"]) == 64
