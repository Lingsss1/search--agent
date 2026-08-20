import tempfile
import unittest
from pathlib import Path

from openstatesearch.data.contamination import find_contamination
from openstatesearch.data.corpus import build_documents, write_jsonl
from openstatesearch.demo import run_demo


class DataAndDemoTests(unittest.TestCase):
    def test_corpus_deduplicates_and_hashes_output(self):
        records = [
            {"title": "T", "text": "same paragraph", "source": "s"},
            {"title": "T", "text": "same   paragraph", "source": "s"},
        ]
        documents = build_documents(records)
        self.assertEqual(len(documents), 1)
        with tempfile.TemporaryDirectory() as directory:
            digest = write_jsonl(Path(directory) / "corpus.jsonl", documents)
            self.assertEqual(len(digest), 64)

    def test_contamination_detects_normalized_exact_match(self):
        matches = find_contamination(["Who, Was ADA?"], ["who was ada"])
        self.assertEqual(matches[0].reason, "exact")

    def test_demo_finishes_with_citation(self):
        trajectory = run_demo()
        self.assertTrue(trajectory["finished"])
        self.assertEqual(trajectory["events"][-1]["result"]["payload"]["answer"], "Charles Babbage")


if __name__ == "__main__":
    unittest.main()
