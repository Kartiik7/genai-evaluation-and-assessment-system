import sys
sys.path.insert(0, ".")
import pandas as pd
from src.retrieval_metrics import compute_retrieval_metrics, compute_random_baseline, compare_retrieval_to_baseline

labels_df = pd.read_csv('results/retrieval_labeling.csv')
retriever_metrics = compute_retrieval_metrics(labels_df)

print("=== Retriever Metrics (Initial Labels) ===")
comparison_df = compare_retrieval_to_baseline(labeled_df=labels_df)
print("\n=== Comparison to Random Baseline ===")
print(comparison_df.to_string(index=False))

