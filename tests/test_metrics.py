import unittest
import numpy as np
import os
import sys

# Ensure src is in path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import evaluate, paired_bootstrap_diff, bootstrap_ci

class TestMetrics(unittest.TestCase):
    def test_perfect_predictions(self):
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        results = evaluate(human, pred)
        
        self.assertAlmostEqual(results['mae'], 0.0)
        self.assertAlmostEqual(results['rmse'], 0.0)
        self.assertAlmostEqual(results['qwk'], 1.0)
        self.assertAlmostEqual(results['pearson'], 1.0)
        self.assertAlmostEqual(results['spearman'], 1.0)
        self.assertAlmostEqual(results['within_1'], 1.0)

    def test_constant_prediction(self):
        human = [1.0, 2.0, 3.0, 4.0, 5.0]
        pred = [3.0, 3.0, 3.0, 3.0, 3.0]
        
        results = evaluate(human, pred)
        
        self.assertTrue(np.isnan(results['pearson']))
        self.assertTrue(np.isnan(results['spearman']))
        self.assertAlmostEqual(results['within_1'], 0.6) # 2.0, 3.0, 4.0 are within 1 of 3.0

    def test_paired_bootstrap_diff(self):
        human = [1.0, 2.0, 3.0, 4.0, 5.0] * 5
        pred_a = [1.0, 2.0, 3.0, 4.0, 5.0] * 5  # Perfect (mae=0)
        pred_b = [2.0, 3.0, 4.0, 5.0, 5.0] * 5  # Off by ~1 (mae ~ 0.8)
        
        res = paired_bootstrap_diff(human, pred_a, pred_b, metric='mae', n=200, seed=42)
        diff, ci = res
        self.assertLess(diff, 0.0)  # pred_a has lower MAE than pred_b, so diff < 0
        self.assertLess(ci[0], ci[1])
        self.assertEqual(res.mean_diff, diff)
        self.assertEqual(res['mean_diff'], diff)

    def test_bootstrap_ci(self):
        human = [1.0, 2.0, 3.0, 4.0, 5.0] * 5
        pred = [1.2, 1.9, 3.1, 3.8, 5.0] * 5
        ci_lo, ci_hi = bootstrap_ci(human, pred, metric='mae', n=200, seed=42)
        self.assertGreater(ci_hi, ci_lo)
        self.assertGreater(ci_lo, 0.0)

if __name__ == '__main__':
    unittest.main()
