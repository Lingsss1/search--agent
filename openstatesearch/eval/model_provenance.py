from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_provenance(model_root: str | Path, output_path: str | Path) -> dict[str, Any]:
    root = Path(model_root)
    output = Path(output_path)
    if not root.is_dir():
        raise ValueError(f"model root is not a directory: {root}")
    output_resolved = output.resolve()
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.resolve() == output_resolved:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    if not files:
        raise ValueError("model artifact has no files")
    manifest = {
        "schema_version": 1,
        "model_root": str(root),
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
