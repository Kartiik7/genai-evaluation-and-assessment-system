import unittest
import numpy as np
import os
import sys

# Ensure src is in path so we can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.metrics import evaluate

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

if __name__ == '__main__':
    unittest.main()
