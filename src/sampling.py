import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_subsets():
    df = pd.read_csv("data/processed/mohler_clean.csv")
    
    # We want to stratify by question_id where possible. 
    # Since dev is 50 and test is 150 (total 200), and there are 81 questions,
    # we can't perfectly stratify by question_id for small sets.
    # We will stratify by a binned score first, and pick randomly to ensure question diversity.
    
    # Create bins for human_score to maintain the distribution
    df['score_bin'] = pd.qcut(df['human_score'], q=4, duplicates='drop')
    
    # Since some questions have very few samples, stratifying perfectly by question_id is hard.
    # Instead, we will sample 200 items stratified by the score bin.
    # This guarantees the score distribution matches the full dataset.
    pool, _ = train_test_split(df, train_size=200, stratify=df['score_bin'], random_state=42)
    
    # Now split the 200 into 50 dev and 150 test, again stratifying by score_bin
    dev, test = train_test_split(pool, train_size=50, test_size=150, stratify=pool['score_bin'], random_state=42)
    
    # Drop the temporary score_bin column and add unique row_id
    dev = dev.drop(columns=['score_bin']).reset_index(drop=True)
    test = test.drop(columns=['score_bin']).reset_index(drop=True)
    dev.insert(0, 'row_id', [f'dev_{i:03d}' for i in range(len(dev))])
    test.insert(0, 'row_id', [f'test_{i:03d}' for i in range(len(test))])
    
    # Save to disk
    os.makedirs("data/processed", exist_ok=True)
    dev.to_csv("data/processed/dev.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    # Print results
    print("=== Full Dataset Score Distribution ===")
    print(df['human_score'].describe())
    print()
    
    print("=== Dev Set (50 responses) ===")
    print(f"Number of unique questions covered: {dev['question_id'].nunique()}")
    print("Score Distribution:")
    print(dev['human_score'].describe())
    print()
    
    print("=== Test Set (150 responses) ===")
    print(f"Number of unique questions covered: {test['question_id'].nunique()}")
    print("Score Distribution:")
    print(test['human_score'].describe())

if __name__ == "__main__":
    create_subsets()
