import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _record(identity: str) -> dict:
    return {
        "dataset": "hotpotqa",
        "id": identity,
        "actions": [
            {
                "action": {"type": "SEARCH", "query": "q", "target_constraint": "c"},
                "visible_doc_ids": [],
                "opened_doc_ids": [],
            }
        ],
        "trajectory_completed": True,
        "prediction": "answer",
        "answers": ["answer"],
        "evidence": [{"doc_id": "d", "sent_ids": [0]}],
        "legal_evidence": [{"doc_id": "d", "sent_ids": [0]}],
        "citations": [{"doc_id": "d", "sent_ids": [0]}],
        "gold_evidence": [{"doc_id": "d", "sent_ids": [0]}],
        "search_count": 1,
        "open_count": 0,
        "input_tokens": 100,
        "generated_tokens": 10,
        "experiment": "A",
        "state_mode": "transcript",
        "retriever": "base_hybrid",
        "memory_token_budget": 4096,
    }


def test_merge_eval_shards_writes_hash_manifest_and_eval_metrics(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    shard = tmp_path / "part.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(
        prompts,
        [
            {"dataset": "hotpotqa", "id": "one"},
            {"dataset": "hotpotqa", "id": "two"},
        ],
    )
    _write(shard, [_record("one"), _record("two")])
    command = [
        sys.executable,
        str(ROOT / "scripts/merge_sft_gate_shards.py"),
        "--prompts",
        str(prompts),
        "--inputs",
        str(shard),
        "--output",
        str(output),
        "--count",
        "2",
        "--expected-experiment",
        "A",
        "--expected-state-mode",
        "transcript",
        "--expected-retriever",
        "base_hybrid",
        "--expected-memory-token-budget",
        "4096",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    metrics = json.loads(output.with_suffix(".jsonl.eval_metrics.json").read_text())
    assert manifest["output"]["rows"] == 2
    assert len(manifest["output"]["sha256"]) == 64
    assert manifest["expected"]["state_mode"] == "transcript"
    assert manifest["metrics"]["sha256"]
    assert metrics["overall"]["examples"] == 2
    assert metrics["overall"]["citation_validity_micro"] == 1.0


def test_merge_eval_shards_rejects_configuration_mixing(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    shard = tmp_path / "part.jsonl"
    output = tmp_path / "merged.jsonl"
    _write(prompts, [{"dataset": "hotpotqa", "id": "one"}])
    _write(shard, [_record("one")])
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/merge_sft_gate_shards.py"),
            "--prompts",
            str(prompts),
            "--inputs",
            str(shard),
            "--output",
            str(output),
            "--count",
            "1",
            "--expected-state-mode",
            "external_state",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "expected 'external_state'" in result.stderr
