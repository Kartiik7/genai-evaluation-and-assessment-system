"""
scripts/add_row_id.py

1. Adds a unique row_id to data/processed/dev.csv and test.csv.
   Format: dev_000 .. dev_049, test_000 .. test_149
2. Adds student_answer and row_id to results/preds_*.csv by positional join with test.csv.
3. Saves everything in-place.
"""

import pandas as pd
from pathlib import Path

DEV_PATH   = Path('data/processed/dev.csv')
TEST_PATH  = Path('data/processed/test.csv')
PREDS      = [
    Path('results/preds_tfidf.csv'),
    Path('results/preds_minilm.csv'),
    Path('results/preds_mpnet.csv'),
]

# ── 1. Add row_id to dev.csv ───────────────────────────────────────────────────
dev = pd.read_csv(DEV_PATH)
if 'row_id' in dev.columns:
    print(f'dev.csv already has row_id — skipping.')
else:
    dev.insert(0, 'row_id', [f'dev_{i:03d}' for i in range(len(dev))])
    dev.to_csv(DEV_PATH, index=False)
    print(f'Added row_id to dev.csv  ({len(dev)} rows: dev_000..dev_{len(dev)-1:03d})')

# ── 2. Add row_id to test.csv ──────────────────────────────────────────────────
test = pd.read_csv(TEST_PATH)
if 'row_id' in test.columns:
    print(f'test.csv already has row_id — skipping.')
else:
    test.insert(0, 'row_id', [f'test_{i:03d}' for i in range(len(test))])
    test.to_csv(TEST_PATH, index=False)
    print(f'Added row_id to test.csv ({len(test)} rows: test_000..test_{len(test)-1:03d})')

# Reload to get fresh row_id column
dev  = pd.read_csv(DEV_PATH)
test = pd.read_csv(TEST_PATH)

# ── 3. Add row_id + student_answer to preds files ─────────────────────────────
for pred_path in PREDS:
    pred = pd.read_csv(pred_path)
    changed = False

    if 'row_id' not in pred.columns:
        pred.insert(0, 'row_id', test['row_id'].values)
        changed = True
        print(f'Added row_id to {pred_path.name}')

    if 'student_answer' not in pred.columns:
        pred.insert(pred.columns.get_loc('human_score') + 1,
                    'student_answer', test['student_answer'].values)
        changed = True
        print(f'Added student_answer to {pred_path.name}')

    if changed:
        pred.to_csv(pred_path, index=False)
    else:
        print(f'{pred_path.name}: already has row_id and student_answer — no changes.')

# ── 4. Verification ────────────────────────────────────────────────────────────
import numpy as np
print()
print('=== POST-UPDATE VERIFICATION ===')
test = pd.read_csv(TEST_PATH)
for pred_path in PREDS:
    pred = pd.read_csv(pred_path)
    rid_ok   = (pred['row_id'].values == test['row_id'].values).all()
    score_ok = np.allclose(pred['human_score'].values, test['human_score'].values, atol=1e-9)
    sa_ok    = (pred['student_answer'].fillna('').values ==
                test['student_answer'].fillna('').values).all()
    print(f'{pred_path.name}: row_id={rid_ok}, human_score={score_ok}, student_answer={sa_ok}')

print()
print('=== FINAL COLUMN ORDER ===')
for pred_path in PREDS:
    pred = pd.read_csv(pred_path)
    print(f'{pred_path.name}: {pred.columns.tolist()}')
