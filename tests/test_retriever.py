import unittest

from openstatesearch.retriever import Document, HybridRetriever
from openstatesearch.retriever.hybrid import reciprocal_rank_fusion
from openstatesearch.retriever.types import split_sentences, stable_doc_id
from openstatesearch.eval.retrieval import ndcg_at_k, recall_at_k
from openstatesearch.eval.retriever_benchmark import evaluate_retriever


class RetrieverTests(unittest.TestCase):
    def test_doc_id_and_sentence_ids_are_deterministic(self):
        self.assertEqual(stable_doc_id("s", "t", "a  b"), stable_doc_id("s", "t", "a b"))
        self.assertEqual(split_sentences("第一句。第二句！"), ("第一句。", "第二句！"))

    def test_rrf_formula(self):
        scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], 60)
        self.assertAlmostEqual(scores["a"], scores["b"])

    def test_hybrid_top_k_is_reproducible(self):
        documents = [
            Document.create("Apples", "Apples grow on trees.", "test"),
            Document.create("Oranges", "Oranges are citrus fruit.", "test"),
            Document.create("Trees", "Trees are perennial plants.", "test"),
        ]
        retriever = HybridRetriever(documents)
        first = [hit.doc_id for hit in retriever.search("apples trees", 2)]
        second = [hit.doc_id for hit in retriever.search("apples trees", 2)]
        self.assertEqual(first, second)
        self.assertEqual(first[0], documents[0].doc_id)

    def test_hybrid_batch_matches_individual_search(self):
        documents = [
            Document.create("Apples", "Apples grow on trees.", "test"),
            Document.create("Oranges", "Oranges are citrus fruit.", "test"),
            Document.create("Trees", "Trees are perennial plants.", "test"),
        ]
        retriever = HybridRetriever(documents)
        queries = ["apples trees", "citrus"]
        batched = retriever.search_batch(queries, 2)
        individual = [retriever.search(query, 2) for query in queries]
        self.assertEqual(
            [[hit.doc_id for hit in ranking] for ranking in batched],
            [[hit.doc_id for hit in ranking] for ranking in individual],
        )

    def test_retrieval_metrics(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a", "c"], 2), 0.5)
        self.assertAlmostEqual(ndcg_at_k(["a", "b"], {"a": 1, "b": 0}, 2), 1.0)

    def test_benchmark_reports_all_required_metrics(self):
        documents = [Document.create("Alpha", "alpha target", "test")]
        metrics = evaluate_retriever(
            HybridRetriever(documents),
            [{"query": "alpha", "relevant_doc_ids": [documents[0].doc_id]}],
        )
        self.assertEqual(metrics["recall_at_20"], 1.0)
        self.assertEqual(metrics["ndcg_at_10"], 1.0)


if __name__ == "__main__":
    unittest.main()
