"""
tests/test_kb.py

Unit tests for knowledge base parsing and chunking functionality in src/kb.py.
"""

import unittest
from pathlib import Path
import tempfile
import pandas as pd

from src.kb import parse_header, chunk_markdown, process_markdown_file, build_kb


class TestKB(unittest.TestCase):

    def test_parse_header_standard(self):
        line = "Source: https://en.wikipedia.org/wiki/Stack | License: CC BY-SA 4.0"
        url, lic = parse_header(line)
        self.assertEqual(url, "https://en.wikipedia.org/wiki/Stack")
        self.assertEqual(lic, "CC BY-SA 4.0")

    def test_parse_header_comment_format(self):
        line = "<!-- URL: https://example.com/algo | License: MIT -->"
        url, lic = parse_header(line)
        self.assertEqual(url, "https://example.com/algo")
        self.assertEqual(lic, "MIT")

    def test_parse_header_empty(self):
        url, lic = parse_header("")
        self.assertEqual(url, "")
        self.assertEqual(lic, "")

    def test_chunk_markdown_bounds_and_overlap(self):
        # Create text with several paragraphs of ~50 words each
        paragraphs = [" ".join([f"w{p}_{i}" for i in range(50)]) for p in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_markdown(text, target_min_words=100, target_max_words=180, overlap_words=15)
        self.assertGreater(len(chunks), 1)

        # Check overlap between first and second chunk
        words_c0 = chunks[0].split()
        words_c1 = chunks[1].split()
        overlap_tail = words_c0[-15:]
        overlap_head = words_c1[:15]
        self.assertEqual(overlap_tail, overlap_head)

    def test_build_kb_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            raw_dir.mkdir()
            md_path = raw_dir / "test_topic.md"
            md_content = (
                "Source: https://example.com/test | License: Apache-2.0\n\n"
                "# Heading\n\n"
                + "\n\n".join([" ".join([f"word_{i}" for i in range(60)]) for _ in range(4)])
            )
            md_path.write_text(md_content, encoding="utf-8")

            out_csv = Path(tmpdir) / "chunks.csv"
            df = build_kb(raw_dir, out_csv, target_min_words=100, target_max_words=200, overlap_words=15)

            self.assertTrue(out_csv.exists())
            self.assertGreater(len(df), 0)
            self.assertEqual(df.iloc[0]["source_file"], "test_topic.md")
            self.assertEqual(df.iloc[0]["topic"], "test_topic")
            self.assertEqual(df.iloc[0]["source_url"], "https://example.com/test")
            self.assertEqual(df.iloc[0]["license"], "Apache-2.0")
            self.assertIn("word_count", df.columns)
            self.assertIn("text", df.columns)


if __name__ == "__main__":
    unittest.main()
