import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prompts import load_prompt, render
from src.llm_eval import validate_direct_output, validate_rubric_output, validate_response, is_valid_score


class TestPrompts(unittest.TestCase):
    def test_load_and_render_direct(self):
        template = load_prompt("direct_v1")
        self.assertIn("{question}", template)
        self.assertIn("{reference_answer}", template)
        self.assertIn("{student_answer}", template)
        
        rendered, ver = render("direct_v1", question="Q1", reference_answer="A1", student_answer="S1")
        self.assertEqual(ver, "direct_v1")
        self.assertIn("Q1", rendered)
        self.assertIn("A1", rendered)
        self.assertIn("S1", rendered)
        self.assertNotIn("{question}", rendered)

    def test_load_and_render_rubric(self):
        rendered, ver = render("rubric_v1.txt", question="Q2", reference_answer="A2", student_answer="S2")
        self.assertEqual(ver, "rubric_v1")
        self.assertIn("correctness", rendered)
        self.assertIn("completeness", rendered)
        self.assertIn("relevance", rendered)
        self.assertIn("understanding", rendered)
        self.assertIn("clarity", rendered)
        self.assertIn("criterion_scores", rendered)


class TestValidators(unittest.TestCase):
    def test_is_valid_score(self):
        self.assertTrue(is_valid_score(0))
        self.assertTrue(is_valid_score(5))
        self.assertTrue(is_valid_score(3.5))
        self.assertTrue(is_valid_score("4.0"))
        
        self.assertFalse(is_valid_score(-0.1))
        self.assertFalse(is_valid_score(5.1))
        self.assertFalse(is_valid_score("abc"))
        self.assertFalse(is_valid_score(None))
        self.assertFalse(is_valid_score(True))
        self.assertFalse(is_valid_score(False))
        self.assertFalse(is_valid_score(np.nan))

    def test_direct_valid(self):
        data = {"reasoning": "Good answer.", "score": 4.5}
        res = validate_direct_output(data)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["score"], 4.5)
        self.assertIsNone(res["error"])

    def test_direct_invalid(self):
        bad_score = {"reasoning": "Good answer.", "score": 6.0}
        res = validate_direct_output(bad_score)
        self.assertFalse(res["is_valid"])
        self.assertTrue(np.isnan(res["score"]))
        self.assertIsNotNone(res["error"])

        bad_type = {"error": "invalid_json"}
        res2 = validate_direct_output(bad_type)
        self.assertFalse(res2["is_valid"])
        self.assertTrue(np.isnan(res2["score"]))

    def test_rubric_valid(self):
        data = {
            "criterion_scores": {
                "correctness": 5,
                "completeness": 4,
                "relevance": 5,
                "understanding": 4.5,
                "clarity": 5
            },
            "strengths": ["Clear terminology"],
            "missing_concepts": ["Edge cases"],
            "errors": [],
            "feedback": "Great job, consider edge cases next time.",
            "reasoning": "Mostly complete and accurate.",
            "score": 4.5
        }
        res = validate_rubric_output(data)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["score"], 4.5)
        self.assertEqual(res["criterion_scores"]["completeness"], 4.0)

    def test_rubric_invalid_criterion(self):
        data = {
            "criterion_scores": {
                "correctness": 5,
                "completeness": 6,  # Out of bounds!
                "relevance": 5,
                "understanding": 4,
                "clarity": 5
            },
            "strengths": [],
            "missing_concepts": [],
            "errors": [],
            "feedback": "Feedback",
            "reasoning": "Reason",
            "score": 4.0
        }
        res = validate_rubric_output(data)
        self.assertFalse(res["is_valid"])
        self.assertTrue(np.isnan(res["score"]))
        self.assertIn("completeness", res["error"])

    def test_rubric_missing_criterion(self):
        data = {
            "criterion_scores": {
                "correctness": 5,
                "completeness": 4,
                # missing relevance, understanding, clarity
            },
            "score": 4.0
        }
        res = validate_rubric_output(data)
        self.assertFalse(res["is_valid"])
        self.assertTrue(np.isnan(res["score"]))


if __name__ == "__main__":
    unittest.main()
