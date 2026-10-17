"""
tests/test_retrieval_metrics.py

Unit tests for retrieval evaluation metrics and random baseline in src/retrieval_metrics.py.
"""

import unittest
import pandas as pd
import numpy as np

from src.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    compute_retrieval_metrics,
    compute_random_baseline,
    find_retrieval_failures,
    compare_retrieval_to_baseline,
)


class TestRetrievalMetrics(unittest.TestCase):

    def test_precision_at_k(self):
        # Top 3 have [1, 0, 1] -> 2/3
        self.assertAlmostEqual(precision_at_k([1, 0, 1, 0, 0], 3), 2 / 3)
        self.assertAlmostEqual(precision_at_k([1, 0, 1, 0, 0], 1), 1.0)
        self.assertAlmostEqual(precision_at_k([0, 0, 1, 0, 0], 1), 0.0)
        self.assertAlmostEqual(precision_at_k([0, 0, 1, 0, 0], 5), 1 / 5)
        self.assertEqual(precision_at_k([], 3), 0.0)

    def test_recall_at_k(self):
        # 4 total relevant in pool
        self.assertAlmostEqual(recall_at_k([1, 0, 1, 0, 0], total_relevant_in_pool=4, k=3), 2 / 4)
        self.assertAlmostEqual(recall_at_k([1, 1, 1, 1, 0], total_relevant_in_pool=4, k=4), 1.0)
        self.assertAlmostEqual(recall_at_k([0, 0, 0], total_relevant_in_pool=4, k=3), 0.0)
        # Pool with 0 relevant items returns 0.0
        self.assertAlmostEqual(recall_at_k([0, 0], total_relevant_in_pool=0, k=2), 0.0)

    def test_reciprocal_rank(self):
        # First relevant at rank 1 -> 1.0
        self.assertAlmostEqual(reciprocal_rank([1, 0, 0]), 1.0)
        # First relevant at rank 2 -> 0.5
        self.assertAlmostEqual(reciprocal_rank([0, 1, 0]), 0.5)
        # First relevant at rank 3 -> 1/3
        self.assertAlmostEqual(reciprocal_rank([0, 0, 1]), 1 / 3)
        # No relevant -> 0.0
        self.assertAlmostEqual(reciprocal_rank([0, 0, 0]), 0.0)

    def test_compute_retrieval_metrics(self):
        # Mock labeled dataframe for 2 questions
        labeled_df = pd.DataFrame([
            # Q1: 3 candidates, c1 and c2 relevant
            {"question_id": 1, "chunk_id": "c1", "relevant": 1},
            {"question_id": 1, "chunk_id": "c2", "relevant": 1},
            {"question_id": 1, "chunk_id": "c3", "relevant": 0},
            # Q2: 3 candidates, only c3 relevant
            {"question_id": 2, "chunk_id": "c1", "relevant": 0},
            {"question_id": 2, "chunk_id": "c2", "relevant": 0},
            {"question_id": 2, "chunk_id": "c3", "relevant": 1},
        ])

        # Real ranking:
        # Q1: c1, c3, c2 -> labels: [1, 0, 1]
        # Q2: c1, c2, c3 -> labels: [0, 0, 1]
        real_ranking = {
            1: ["c1", "c3", "c2"],
            2: ["c1", "c2", "c3"],
        }

        metrics = compute_retrieval_metrics(labeled_df, real_ranking=real_ranking)

        # Q1 P@1=1.0, Q2 P@1=0.0 -> mean 0.5
        self.assertAlmostEqual(metrics["Precision@1"], 0.5)
        # Q1 P@3=2/3, Q2 P@3=1/3 -> mean 0.5
        self.assertAlmostEqual(metrics["Precision@3"], 0.5)
        # Q1 R@3=2/2=1.0, Q2 R@3=1/1=1.0 -> mean 1.0
        self.assertAlmostEqual(metrics["Recall@3"], 1.0)
        # Q1 MRR=1.0, Q2 MRR=1/3 -> mean (1 + 1/3)/2 = 2/3
        self.assertAlmostEqual(metrics["MRR"], 2 / 3)

    def test_empty_labels_raises(self):
        labeled_df = pd.DataFrame([
            {"question_id": 1, "chunk_id": "c1", "relevant": ""},
            {"question_id": 1, "chunk_id": "c2", "relevant": None},
        ])
        with self.assertRaises(ValueError):
            compute_retrieval_metrics(labeled_df)

    def test_compute_random_baseline(self):
        labeled_df = pd.DataFrame([
            {"question_id": 1, "chunk_id": "c1", "relevant": 1},
            {"question_id": 1, "chunk_id": "c2", "relevant": 0},
            {"question_id": 1, "chunk_id": "c3", "relevant": 0},
        ])
        baseline = compute_random_baseline(
            labeled_df=labeled_df,
            n_repetitions=50,
            seed=42,
            kb_chunk_ids=["c1", "c2", "c3"],
        )
        self.assertIn("Precision@1", baseline)
        self.assertIn("MRR", baseline)
        self.assertGreater(baseline["Precision@1"], 0.0)

    def test_find_retrieval_failures(self):
        labeled_df = pd.DataFrame([
            {"question_id": 1, "chunk_id": "c1", "relevant": 0},
            {"question_id": 1, "chunk_id": "c2", "relevant": 0},
            {"question_id": 1, "chunk_id": "c3", "relevant": 1},
        ])
        # c3 is relevant, but real ranking places it at rank 4 (not in top 3)
        real_ranking = {1: ["c1", "c2", "c4", "c3"]}
        failures = find_retrieval_failures(labeled_df, real_ranking=real_ranking, k=3)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures.iloc[0]["question_id"], 1)


if __name__ == "__main__":
    unittest.main()
