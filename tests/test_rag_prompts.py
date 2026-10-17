"""
tests/test_rag_prompts.py

Tests for RAG prompt templates, context formatting, and retrieval integration.
"""

import unittest
from pathlib import Path

from src.prompts import load_prompt, render
from src.rag import build_rag_context, truncate_chunk, retrieve_rag_context


class TestRAGPrompts(unittest.TestCase):

    def test_rag_v1_template_exists_and_renders(self):
        rendered, ver = render(
            "rag_v1",
            question="What is a stack?",
            reference_answer="A LIFO structure.",
            student_answer="Last in first out.",
            retrieved_context="Stacks adhere strictly to LIFO.",
        )
        self.assertEqual(ver, "rag_v1")
        self.assertIn("Course material (supporting context", rendered)
        self.assertIn("The reference answer remains the primary standard", rendered)
        self.assertIn("Stacks adhere strictly to LIFO.", rendered)

    def test_rag_rubric_v1_template_exists_and_renders(self):
        rendered, ver = render(
            "rag_rubric_v1",
            question="What is a queue?",
            reference_answer="A FIFO structure.",
            student_answer="First in first out.",
            retrieved_context="Queues adhere to FIFO.",
        )
        self.assertEqual(ver, "rag_rubric_v1")
        self.assertIn("Course material (supporting context", rendered)
        self.assertIn("criterion_scores", rendered)
        self.assertIn("Overall scoring anchors:", rendered)

    def test_truncate_chunk(self):
        words = ["word"] * 200
        text = " ".join(words)
        truncated = truncate_chunk(text, max_words=150)
        self.assertEqual(len(truncated.split()), 150)
        self.assertTrue(truncated.endswith("..."))


    def test_build_rag_context_limits(self):
        chunks = ["chunk1 " * 50, "chunk2 " * 50, "chunk3 " * 50, "chunk4 " * 50]
        ctx = build_rag_context(chunks, max_chunks=3, max_words_per_chunk=150)
        # Should only contain top 3 chunks
        self.assertIn("chunk1", ctx)
        self.assertIn("chunk2", ctx)
        self.assertIn("chunk3", ctx)
        self.assertNotIn("chunk4", ctx)

    def test_retrieve_rag_context_uses_question_only(self):
        chunk_ids, ctx = retrieve_rag_context("What is a binary tree?", k=3)
        self.assertEqual(len(chunk_ids), 3)
        self.assertGreater(len(ctx), 0)


if __name__ == "__main__":
    unittest.main()
