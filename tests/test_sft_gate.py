import unittest

from openstatesearch.eval.sft_gate import passes_sft_gate, sft_gate_metrics


class SFTGateTests(unittest.TestCase):
    def test_gate_counts_json_references_queries_and_completion(self):
        records = [
            {
                "action": {"type": "SEARCH", "query": "q", "target_constraint": "c"},
                "trajectory_completed": True,
            },
            {
                "action": {"type": "OPEN", "doc_id": "d"},
                "visible_doc_ids": ["d"],
                "trajectory_completed": True,
            },
        ]
        metrics = sft_gate_metrics(records)
        self.assertEqual(metrics["valid_tool_json"], 1.0)
        self.assertEqual(metrics["invalid_doc_reference"], 0.0)
        self.assertTrue(
            passes_sft_gate(
                metrics,
                {
                    "valid_tool_json": 0.98,
                    "invalid_doc_reference_max": 0.02,
                    "nonempty_search_query": 0.99,
                    "completion_rate": 0.9,
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
