from __future__ import annotations

import json

import pytest

from scripts.merge_adapter import validate_adapter_base, write_merge_manifest


def _write_model(root, name: str, payload: bytes = b"value") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)


def test_merge_manifest_hashes_base_adapter_and_output(tmp_path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "output"
    _write_model(base, "config.json")
    _write_model(adapter, "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    _write_model(output, "model.safetensors", b"merged")
    validate_adapter_base(base, adapter)
    destination = output / "merge_manifest.json"
    manifest = write_merge_manifest(
        base=base,
        adapter=adapter,
        output=output,
        revision="revision",
        destination=destination,
    )
    assert destination.is_file()
    assert manifest["output"]["files"][0]["sha256"]
    assert all(item["path"] != "merge_manifest.json" for item in manifest["output"]["files"])


def test_adapter_base_mismatch_is_rejected(tmp_path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    _write_model(base, "config.json")
    _write_model(adapter, "adapter_model.safetensors")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "different-base"})
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_adapter_base(base, adapter)
