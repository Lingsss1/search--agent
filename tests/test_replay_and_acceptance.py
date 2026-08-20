import json
import unittest

from openstatesearch.demo import demo_documents, run_demo
from openstatesearch.eval.acceptance import audit_acceptance
from openstatesearch.eval.replay import replay_frozen_trajectory, replay_trajectory
from openstatesearch.retriever import HybridRetriever
from openstatesearch.retriever.service import load_corpus
from scripts.replay_demo import _write_replay_artifact
from scripts.replay_formal_demo import build_formal_replay_artifact


class ReplayAndAcceptanceTests(unittest.TestCase):
    def test_demo_replays_exactly(self):
        report = replay_trajectory(run_demo(), HybridRetriever(demo_documents()))
        self.assertTrue(report["exact"], report["divergences"])

    def test_replay_artifact_is_self_contained(self):
        import tempfile
        from pathlib import Path

        trajectory = run_demo()
        documents = demo_documents()
        report = replay_trajectory(trajectory, HybridRetriever(documents))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _write_replay_artifact(root, trajectory, documents, report)
            saved = json.loads((root / "trajectory.json").read_text())
            replayed = replay_trajectory(
                saved,
                HybridRetriever(load_corpus(root / "corpus.jsonl")),
            )
            self.assertTrue(replayed["exact"])
            self.assertTrue(manifest["exact"])

    def test_formal_prediction_builds_exact_frozen_replay(self):
        import tempfile
        from pathlib import Path

        source = run_demo()
        record = {
            "dataset": "demo",
            "id": "formal-1",
            "question": source["question"],
            "actions": [
                {"action": event["action"], "tool_result": event["result"]}
                for event in source["events"]
            ],
            "trajectory_completed": True,
            "search_count": sum(
                event["action"]["type"] in {"SEARCH", "VERIFY"} for event in source["events"]
            ),
            "open_count": sum(event["action"]["type"] == "OPEN" for event in source["events"]),
            "generated_tokens": 100,
            "generation_token_budget": 8192,
            "model_provenance_sha256": "a" * 64,
            "retriever_provenance_sha256": "b" * 64,
            "run_config_sha256": "c" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps(record) + "\n")
            manifest = build_formal_replay_artifact(predictions, root / "demo")
            saved = json.loads((root / "demo" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["exact"])
            self.assertEqual(manifest["kind"], "formal_f8192_frozen_environment_replay")
            self.assertTrue(replay_frozen_trajectory(saved)["exact"])

    def test_acceptance_missing_evidence_always_fails(self):
        thresholds = {
            "retriever_recall_at_20_absolute_gain": 0.03,
            "external_state_token_reduction_at_same_f1": 0.2,
            "external_state_f1_gain_at_same_tokens": 2,
            "grpo_f1_gain": 3,
            "grpo_search_reduction_if_quality_stable": 0.15,
            "citation_validity": 0.98,
            "citation_precision": 0.85,
            "require_cost_records": ["main"],
            "completion_artifacts": ["A-F"],
        }
        self.assertFalse(audit_acceptance({}, thresholds)["passed"])


if __name__ == "__main__":
    unittest.main()
