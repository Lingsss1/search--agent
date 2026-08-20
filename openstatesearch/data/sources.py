from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def huggingface_dataset_id(url: str) -> str | None:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc == "huggingface.co" and len(parts) >= 3 and parts[0] == "datasets":
        return "/".join(parts[1:3])
    return None


def load_sources(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("source manifest requires a sources list")
    for source in sources:
        revision = source.get("revision", "")
        if not isinstance(revision, str) or len(revision) != 40:
            raise ValueError(f"source {source.get('name')} is not pinned to a 40-character commit")
    return sources


def fetch_source(source: dict[str, Any], root: str | Path) -> Path:
    """Fetch exactly one pinned source and never overwrite an existing directory."""
    destination = Path(root).resolve() / str(source["name"])
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = huggingface_dataset_id(str(source["url"]))
    if dataset_id:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "install openstatesearch[data] to fetch Hugging Face datasets"
            ) from exc
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=str(source["revision"]),
            local_dir=destination,
        )
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                str(source["url"]),
                str(destination),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--depth=1",
                "origin",
                str(source["revision"]),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"],
            check=True,
        )
    provenance = {
        "name": source["name"],
        "url": source["url"],
        "revision": source["revision"],
        "license": source["license"],
        "role": source["role"],
    }
    (destination / "OSS36_PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination
