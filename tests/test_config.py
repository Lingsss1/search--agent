import unittest
from pathlib import Path

from openstatesearch.training import load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_sft_and_grpo_configs_are_frozen_and_valid(self):
        for filename, kind in [
            ("policy_sft.yaml", "sft"),
            ("grpo_a.yaml", "grpo"),
            ("grpo_b.yaml", "grpo"),
            ("evaluation.yaml", "evaluation"),
        ]:
            config = load_config(ROOT / "configs" / filename)
            self.assertEqual(validate_config(config, kind), [], filename)

    def test_retriever_revisions_are_pinned(self):
        config = load_config(ROOT / "configs" / "retriever_lrat.yaml")
        errors = validate_config(config, "retriever")
        self.assertEqual(errors, [])
        self.assertEqual(len(config["model_revision"]), 40)
        self.assertEqual(len(config["dataset_revision"]), 40)


if __name__ == "__main__":
    unittest.main()
