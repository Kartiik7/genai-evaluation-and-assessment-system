import pandas as pd

test_df = pd.read_csv('data/processed/test.csv')
label_df = pd.read_csv('results/retrieval_labeling.csv')
qids = label_df['question_id'].unique()

for i, qid in enumerate(qids, 1):
    sub = test_df[test_df['question_id'] == qid]
    q_text = sub.iloc[0]['question'] if len(sub) > 0 else "Unknown"
    ref = sub.iloc[0]['reference_answer'] if len(sub) > 0 else "Unknown"
    print(f"[{i}/30] QID: {qid}")

    print(f"  Question: {q_text}")
    print(f"  Reference Answer: {ref}")
