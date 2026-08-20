#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openstatesearch.demo import demo_documents, run_demo
from openstatesearch.eval.replay import replay_trajectory
from openstatesearch.retriever import HybridRetriever
from openstatesearch.retriever.service import load_corpus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_replay_artifact(
    artifact_dir: Path,
    trajectory: dict,
    documents: list,
    report: dict,
) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = artifact_dir / "trajectory.json"
    corpus_path = artifact_dir / "corpus.jsonl"
    report_path = artifact_dir / "replay_report.json"
    trajectory_path.write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    corpus_path.write_text(
        "".join(
            json.dumps(asdict(document), ensure_ascii=False, sort_keys=True) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "replay_command": (
            f"python scripts/replay_demo.py --trajectory {trajectory_path} --corpus {corpus_path}"
        ),
        "exact": bool(report.get("exact")),
        "artifacts": {
            "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
            "corpus": {"path": str(corpus_path), "sha256": _sha256(corpus_path)},
            "report": {"path": str(report_path), "sha256": _sha256(report_path)},
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a frozen SearchState trajectory")
    parser.add_argument(
        "--trajectory", help="saved trajectory JSON; omitted uses the built-in demo"
    )
    parser.add_argument("--corpus", help="frozen corpus JSONL for external trajectories")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="write a self-contained trajectory, corpus, replay report, and SHA manifest",
    )
    args = parser.parse_args()
    if args.trajectory:
        if not args.corpus:
            parser.error("--corpus is required with --trajectory")
        trajectory = json.loads(Path(args.trajectory).read_text(encoding="utf-8"))
        documents = load_corpus(args.corpus)
    else:
        trajectory = run_demo()
        documents = demo_documents()
    report = replay_trajectory(trajectory, HybridRetriever(documents))
    if args.artifact_dir:
        _write_replay_artifact(args.artifact_dir, trajectory, documents, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["exact"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
