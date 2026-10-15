"""
tests/test_retrieval.py

Unit tests for knowledge base retrieval module src/retrieval.py.
"""

import unittest
from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
import faiss

from src.retrieval import build_index, retrieve, format_query


class TestRetrieval(unittest.TestCase):

    def test_format_query_question_mode(self):
        q = "What is a stack?"
        ref = "A LIFO structure"
        formatted = format_query(q, ref, mode="question")
        self.assertEqual(formatted, "What is a stack?")

    def test_format_query_question_plus_reference(self):
        q = "What is a stack?"
        ref = "A LIFO structure"
        formatted = format_query(q, ref, mode="question+reference")
        self.assertIn("What is a stack?", formatted)
        self.assertIn("Reference Answer: A LIFO structure", formatted)

    def test_format_query_invalid_mode(self):
        with self.assertRaises(ValueError):
            format_query("question", "ref", mode="invalid_mode")

    def test_retrieve_mock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks_csv = tmp_path / "chunks.csv"
            index_faiss = tmp_path / "index.faiss"

            # Create dummy chunks
            df = pd.DataFrame([
                {"chunk_id": "c1", "text": "A stack is a last in first out data structure."},
                {"chunk_id": "c2", "text": "A queue operates on first in first out principles."},
                {"chunk_id": "c3", "text": "Binary search trees maintain ordered nodes."},
            ])
            df.to_csv(chunks_csv, index=False)

            # Build index
            index = build_index(chunks_path=chunks_csv, index_path=index_faiss)
            self.assertTrue(index_faiss.exists())
            self.assertEqual(index.ntotal, 3)

            # Retrieve top 2
            chunk_ids, texts, scores = retrieve(
                "How does a stack push and pop elements?",
                k=2,
                index_path=index_faiss,
                chunks_path=chunks_csv,
            )
            self.assertEqual(len(chunk_ids), 2)
            self.assertEqual(len(texts), 2)
            self.assertEqual(len(scores), 2)
            self.assertEqual(chunk_ids[0], "c1")
            self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
