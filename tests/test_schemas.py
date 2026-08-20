import unittest

from openstatesearch.agent.schemas import (
    ActionValidationError,
    AnswerAction,
    KeepAction,
    SearchAction,
    parse_action,
)


class SchemaTests(unittest.TestCase):
    def test_parses_every_payload_shape(self):
        self.assertIsInstance(
            parse_action({"type": "SEARCH", "query": "q", "target_constraint": "c"}),
            SearchAction,
        )
        self.assertIsInstance(
            parse_action(
                {
                    "type": "KEEP",
                    "doc_id": "d",
                    "sent_ids": [0, 2],
                    "claim": "claim",
                    "constraint_id": "c1",
                }
            ),
            KeepAction,
        )
        self.assertIsInstance(
            parse_action(
                {
                    "type": "ANSWER",
                    "answer": "answer",
                    "citations": [{"claim": "c", "doc_id": "d", "sent_ids": [0]}],
                }
            ),
            AnswerAction,
        )

    def test_rejects_unknown_fields_and_bad_sentence_ids(self):
        with self.assertRaises(ActionValidationError):
            parse_action({"type": "OPEN", "doc_id": "d", "gold": "leak"})
        with self.assertRaises(ActionValidationError):
            parse_action(
                {
                    "type": "KEEP",
                    "doc_id": "d",
                    "sent_ids": [-1],
                    "claim": "claim",
                    "constraint_id": "c",
                }
            )

    def test_search_query_cannot_be_empty(self):
        with self.assertRaises(ActionValidationError):
            parse_action({"type": "SEARCH", "query": "  ", "target_constraint": "c"})

    def test_keep_constraint_id_is_optional_bookkeeping(self):
        for value in ({}, {"constraint_id": ""}, {"constraint_id": None}):
            payload = {
                "type": "KEEP",
                "doc_id": "doc",
                "sent_ids": [0],
                "claim": "supported claim",
                **value,
            }
            self.assertEqual(parse_action(payload).constraint_id, "unlabelled")


if __name__ == "__main__":
    unittest.main()
