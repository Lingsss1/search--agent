import unittest

from openstatesearch.eval.runner import evaluate_by_dataset, evaluate_records


def _record(dataset: str = "hotpotqa") -> dict:
    return {
        "id": "example-1",
        "dataset": dataset,
        "prediction": "Alpha",
        "answers": ["Alpha"],
        "evidence": [
            {"doc_id": "gold", "sent_ids": [0]},
            {"doc_id": "distractor", "sent_ids": [2]},
        ],
        "citations": [
            {"doc_id": "gold", "sent_ids": [0]},
            {"doc_id": "distractor", "sent_ids": [2]},
        ],
        "gold_evidence": [{"doc_id": "gold", "sent_ids": [0]}],
        "search_count": 2,
        "open_count": 2,
        "input_tokens": 100,
        "generated_tokens": 20,
    }


class EvalRunnerTests(unittest.TestCase):
    def test_citation_validity_is_distinct_from_gold_precision(self):
        metrics = evaluate_records([_record()])
        self.assertEqual(metrics["citation_validity_micro"], 1.0)
        self.assertEqual(metrics["citation_precision_micro"], 0.5)
        self.assertEqual(metrics["citation_precision_macro"], 0.5)
        self.assertEqual(metrics["evidence_precision"], 0.5)
        self.assertEqual(metrics["avg_input_tokens"], 100.0)
        self.assertEqual(metrics["avg_total_tokens"], 120.0)

    def test_explicit_legal_evidence_detects_illegal_citation(self):
        record = _record()
        record["legal_evidence"] = [{"doc_id": "gold", "sent_ids": [0]}]
        metrics = evaluate_records([record])
        self.assertEqual(metrics["citation_validity_micro"], 0.5)
        self.assertEqual(metrics["citation_precision_micro"], 0.5)

    def test_by_dataset_requires_labels_and_emits_overall(self):
        records = [_record("hotpotqa"), _record("musique")]
        metrics = evaluate_by_dataset(records)
        self.assertEqual(set(metrics), {"overall", "hotpotqa", "musique"})
        self.assertEqual(metrics["overall"]["examples"], 2)
        with self.assertRaisesRegex(ValueError, "dataset"):
            evaluate_by_dataset([{**_record(), "dataset": ""}])

    def test_negative_cost_is_rejected(self):
        record = _record()
        record["input_tokens"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate_records([record])


if __name__ == "__main__":
    unittest.main()
