from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts import train_grpo


def _populate_source_tree(root: Path) -> None:
    for index, relative in enumerate(train_grpo._ROLLOUT_SOURCE_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")


def _remote_output(root: Path, *, corrupt: str | None = None) -> str:
    lines = []
    for relative in train_grpo._ROLLOUT_SOURCE_FILES:
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if relative == corrupt:
            digest = "0" * 64
        lines.append(f"{digest}  /remote/workspace/{relative}")
    return "\n".join(lines) + "\n"


def test_remote_source_parity_records_matching_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate_source_tree(tmp_path)
    output = _remote_output(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train_grpo.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )

    evidence = train_grpo._verify_remote_source_parity("h800", "/remote/workspace")

    assert evidence["source"] == "h800"
    assert len(evidence["files"]) == len(train_grpo._ROLLOUT_SOURCE_FILES)


def test_remote_source_parity_rejects_stale_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _populate_source_tree(tmp_path)
    stale = train_grpo._ROLLOUT_SOURCE_FILES[0]
    output = _remote_output(tmp_path, corrupt=stale)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        train_grpo.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""),
    )

    with pytest.raises(RuntimeError, match="mismatched"):
        train_grpo._verify_remote_source_parity("h800", "/remote/workspace")
