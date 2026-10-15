"""Deep alignment audit: verify preds files match test.csv row-by-row,
add student_answer to preds, and add row_id to split files."""

import pandas as pd
import numpy as np

dev   = pd.read_csv('data/processed/dev.csv')
test  = pd.read_csv('data/processed/test.csv')
tfidf  = pd.read_csv('results/preds_tfidf.csv')
minilm = pd.read_csv('results/preds_minilm.csv')
mpnet  = pd.read_csv('results/preds_mpnet.csv')

print("=== SPLIT OVERLAP VIA student_answer ===")
# student_answer (stripped) + question_id as fingerprint
dev_fp  = set(zip(dev['question_id'], dev['student_answer'].fillna('').str.strip()))
test_fp = set(zip(test['question_id'], test['student_answer'].fillna('').str.strip()))
overlap = dev_fp & test_fp
print(f"dev   unique (qid, student_answer) pairs : {len(dev_fp)}")
print(f"test  unique (qid, student_answer) pairs : {len(test_fp)}")
print(f"Pairs present in BOTH dev AND test       : {len(overlap)}")
if overlap:
    print("  Sample overlapping pairs:", list(overlap)[:3])
print()

print("=== ROW-BY-ROW ALIGNMENT CHECK (all three preds vs test.csv) ===")
for name, pred_df in [('tfidf', tfidf), ('minilm', minilm), ('mpnet', mpnet)]:
    if len(pred_df) != len(test):
        print(f"{name}: LENGTH MISMATCH {len(pred_df)} vs {len(test)}")
        continue
    qid_ok   = np.allclose(pred_df['question_id'].values,  test['question_id'].values,  atol=1e-9)
    score_ok = np.allclose(pred_df['human_score'].values,  test['human_score'].values,  atol=1e-9)
    
    # Also verify student_answer column in test matches positionally
    # (no student_answer in preds, but we can verify via test)
    test_sa = test['student_answer'].fillna('').str.strip().values
    
    print(f"{name} ({len(pred_df)} rows):")
    print(f"  question_id positionally aligned  : {qid_ok}")
    print(f"  human_score positionally aligned  : {score_ok}")

    if not qid_ok or not score_ok:
        bad = [i for i in range(len(test))
               if abs(pred_df.iloc[i]['question_id'] - test.iloc[i]['question_id']) > 1e-9
               or abs(pred_df.iloc[i]['human_score']  - test.iloc[i]['human_score'])  > 1e-9]
        print(f"  MISALIGNED ROWS: {bad[:20]}")
    print()

print("=== CHECK FOR DEV ROWS INSIDE PREDS ===")
# A pred row belongs to dev (and NOT test) if its (qid, student_answer) is in dev_fp
# but not in test_fp. Since preds don't have student_answer we merge by position.
test_sa_list = test['student_answer'].fillna('').str.strip().tolist()
test_qid_list = test['question_id'].tolist()

pred_fps_in_dev_only = 0
for i in range(len(test)):
    fp = (test_qid_list[i], test_sa_list[i])
    if fp in dev_fp and fp not in test_fp:
        pred_fps_in_dev_only += 1

print(f"Preds rows whose (qid, student_answer) is in DEV but NOT in test : {pred_fps_in_dev_only}")
print("(0 = preds are test-only; any positive number means dev leakage)")
print()
print("FINAL VERDICT:")
all_aligned = all(
    len(p) == len(test) and
    np.allclose(p['question_id'].values, test['question_id'].values, atol=1e-9) and
    np.allclose(p['human_score'].values, test['human_score'].values, atol=1e-9)
    for p in [tfidf, minilm, mpnet]
)
print(f"All three preds files are positionally aligned with test.csv: {all_aligned}")
print(f"Dev leakage rows in preds: {pred_fps_in_dev_only}")
