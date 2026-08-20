import unittest

from openstatesearch.agent.harness import SearchHarness
from openstatesearch.agent.schemas import parse_action
from openstatesearch.agent.state import SearchState
from openstatesearch.retriever.types import Document, SearchHit


class CountingRetriever:
    def __init__(self):
        self.document = Document.create("Doc", "First sentence. Second sentence.", "test")
        self.calls = 0

    def search(self, query, k=5):
        self.calls += 1
        doc = self.document
        return [SearchHit(doc.doc_id, doc.title, 1.0, doc.text, doc.source)]

    def get_document(self, doc_id):
        return self.document if doc_id == self.document.doc_id else None


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.retriever = CountingRetriever()
        self.state = SearchState("question")
        self.harness = SearchHarness(self.state, self.retriever)

    def search(self, query="A Query"):
        return self.harness.apply(
            parse_action({"type": "SEARCH", "query": query, "target_constraint": "c"})
        )

    def test_gold_cannot_enter_policy_observation(self):
        observation = self.state.observation()
        self.assertNotIn("gold", observation)
        self.assertNotIn("answer", observation)
        self.assertNotIn("supporting_facts", observation)

    def test_duplicate_query_does_not_call_or_consume(self):
        self.assertTrue(self.search().ok)
        budget = self.state.budget.search_left
        duplicate = self.search("  a   query ")
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(self.retriever.calls, 1)
        self.assertEqual(self.state.budget.search_left, budget)

    def test_keep_requires_opened_document_and_legal_sentence(self):
        result = self.search()
        doc_id = result.payload["hits"][0]["doc_id"]
        unopened = self.harness.apply(
            parse_action(
                {
                    "type": "KEEP",
                    "doc_id": doc_id,
                    "sent_ids": [0],
                    "claim": "x",
                    "constraint_id": "c",
                }
            )
        )
        self.assertFalse(unopened.ok)
        self.assertTrue(self.harness.apply(parse_action({"type": "OPEN", "doc_id": doc_id})).ok)
        invalid = self.harness.apply(
            parse_action(
                {
                    "type": "KEEP",
                    "doc_id": doc_id,
                    "sent_ids": [9],
                    "claim": "x",
                    "constraint_id": "c",
                }
            )
        )
        self.assertFalse(invalid.ok)
        valid = self.harness.apply(
            parse_action(
                {
                    "type": "KEEP",
                    "doc_id": doc_id,
                    "sent_ids": [1],
                    "claim": "x",
                    "constraint_id": "c",
                }
            )
        )
        self.assertTrue(valid.ok)

    def test_open_must_come_from_candidate_pool_and_repeat_is_free(self):
        outsider = Document.create("Other", "No result.", "test")
        denied = self.harness.apply(parse_action({"type": "OPEN", "doc_id": outsider.doc_id}))
        self.assertFalse(denied.ok)
        doc_id = self.search().payload["hits"][0]["doc_id"]
        first = self.harness.apply(parse_action({"type": "OPEN", "doc_id": doc_id}))
        budget = self.state.budget.open_left
        second = self.harness.apply(parse_action({"type": "OPEN", "doc_id": doc_id}))
        self.assertTrue(first.ok)
        self.assertTrue(second.duplicate)
        self.assertEqual(self.state.budget.open_left, budget)

    def test_invalid_citation_cannot_finish(self):
        answer = self.harness.apply(
            parse_action(
                {
                    "type": "ANSWER",
                    "answer": "x",
                    "citations": [{"claim": "x", "doc_id": "missing", "sent_ids": [0]}],
                }
            )
        )
        self.assertFalse(answer.ok)
        self.assertFalse(self.harness.finished)

    def test_answer_citations_must_have_been_kept(self):
        doc_id = self.search().payload["hits"][0]["doc_id"]
        self.assertTrue(self.harness.apply(parse_action({"type": "OPEN", "doc_id": doc_id})).ok)
        unkept = self.harness.apply(
            parse_action(
                {
                    "type": "ANSWER",
                    "answer": "x",
                    "citations": [{"claim": "x", "doc_id": doc_id, "sent_ids": [0]}],
                }
            )
        )
        self.assertFalse(unkept.ok)
        self.assertIn("was not kept", unkept.error)
        self.assertTrue(
            self.harness.apply(
                parse_action(
                    {
                        "type": "KEEP",
                        "doc_id": doc_id,
                        "sent_ids": [0],
                        "claim": "x",
                        "constraint_id": "c",
                    }
                )
            ).ok
        )
        kept = self.harness.apply(
            parse_action(
                {
                    "type": "ANSWER",
                    "answer": "x",
                    "citations": [{"claim": "x", "doc_id": doc_id, "sent_ids": [0]}],
                }
            )
        )
        self.assertTrue(kept.ok)
        self.assertTrue(self.harness.finished)


if __name__ == "__main__":
    unittest.main()
