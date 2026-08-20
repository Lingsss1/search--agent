from __future__ import annotations

import hashlib
import json

from openstatesearch.eval.model_provenance import write_model_provenance


def test_model_provenance_hashes_all_files_and_excludes_itself(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "weights.bin").write_bytes(b"weights")
    output = model / "model_provenance.json"
    manifest = write_model_provenance(model, output)
    assert [item["path"] for item in manifest["files"]] == [
        "config.json",
        "weights.bin",
    ]
    weights = next(item for item in manifest["files"] if item["path"] == "weights.bin")
    assert weights["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert json.loads(output.read_text()) == manifest
