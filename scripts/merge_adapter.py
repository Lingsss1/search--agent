#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> list[dict[str, object]]:
    if not root.is_dir():
        raise ValueError(f"model artifact directory does not exist: {root}")
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "merge_manifest.json":
            continue
        values.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not values:
        raise ValueError(f"model artifact directory is empty: {root}")
    return values


def validate_adapter_base(base: Path, adapter: Path) -> dict[str, object]:
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("adapter requires adapter_config.json and adapter_model.safetensors")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared = str(config.get("base_model_name_or_path", ""))
    if not declared:
        raise ValueError("adapter_config.json has no base_model_name_or_path")
    declared_path = Path(declared)
    matches = declared == str(base)
    if declared_path.exists() and base.exists():
        matches = matches or declared_path.resolve() == base.resolve()
    if not matches:
        raise ValueError(f"adapter base {declared!r} does not match requested base {str(base)!r}")
    return config


def write_merge_manifest(
    *,
    base: Path,
    adapter: Path,
    output: Path,
    revision: str | None,
    destination: Path,
) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base": {
            "path": str(base),
            "revision": revision,
            "files": _tree_manifest(base),
        },
        "adapter": {"path": str(adapter), "files": _tree_manifest(adapter)},
        "output": {"path": str(output), "files": _tree_manifest(output)},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a Policy LoRA adapter for AReaL initialization"
    )
    parser.add_argument("--base", default="Qwen/Qwen3.6-27B")
    parser.add_argument(
        "--revision",
        help="optional provenance label for the local base; file SHA values remain authoritative",
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    base_path = Path(args.base)
    adapter_path = Path(args.adapter)
    output_path = Path(args.output)
    if not base_path.is_dir():
        raise SystemExit("audited adapter merging requires a local base model directory")
    validate_adapter_base(base_path, adapter_path)
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("install openstatesearch[training] before merging adapters") from exc
    revision_args = {"local_files_only": True}
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True, **revision_args)
    base = AutoModelForMultimodalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, **revision_args
    )
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_path / "merge_manifest.json"
    manifest = write_merge_manifest(
        base=base_path,
        adapter=adapter_path,
        output=output_path,
        revision=args.revision,
        destination=manifest_path,
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "output_files": len(manifest["output"]["files"])}
        )
    )


if __name__ == "__main__":
    main()
