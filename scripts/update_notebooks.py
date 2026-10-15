import json

def update_02():
    path = 'notebooks/02_tfidf.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    updated = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = cell.get('source', [])
            for i, line in enumerate(src):
                old_str = "df_test[['question_id', 'human_score', 'pred_scaled', 'pred_calibrated']]"
                new_str = "df_test[['row_id', 'question_id', 'human_score', 'student_answer', 'pred_scaled', 'pred_calibrated']]"
                if old_str in line:
                    src[i] = line.replace(old_str, new_str)
                    updated = True
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Updated 02_tfidf.ipynb")
    else:
        print("02_tfidf.ipynb already up to date")

def update_03():
    path = 'notebooks/03_semantic_baselines.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    updated = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = cell.get('source', [])
            if isinstance(src, str):
                old_str = "out = df_test[['question_id', 'human_score']].copy()"
                new_str = "out = df_test[['row_id', 'question_id', 'human_score', 'student_answer']].copy()"
                if old_str in src:
                    cell['source'] = src.replace(old_str, new_str)
                    updated = True
            elif isinstance(src, list):
                for i, line in enumerate(src):
                    old_str = "out = df_test[['question_id', 'human_score']].copy()"
                    new_str = "out = df_test[['row_id', 'question_id', 'human_score', 'student_answer']].copy()"
                    if old_str in line:
                        src[i] = line.replace(old_str, new_str)
                        updated = True
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Updated 03_semantic_baselines.ipynb")
    else:
        print("03_semantic_baselines.ipynb already up to date")

def update_04():
    path = 'notebooks/04_baseline_comparison.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    updated = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = cell.get('source', [])
            if isinstance(src, list):
                for i, line in enumerate(src):
                    old_str = "'TF-IDF (Calibrated)': preds_tfidf['pred_calibrated'].values"
                    new_str = "'TF-IDF (Calibrated)': preds_tfidf.set_index('row_id').loc[df_test['row_id']]['pred_calibrated'].values"
                    if old_str in line:
                        src[i] = line.replace(old_str, new_str)
                        updated = True
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Updated 04_baseline_comparison.ipynb to join preds_tfidf on row_id")
    else:
        print("04_baseline_comparison.ipynb already up to date")

def update_06():
    path = 'notebooks/06_llm_vs_baselines.ipynb'
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    updated = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            src = cell.get('source', [])
            if isinstance(src, list):
                for i, line in enumerate(src):
                    old_str = "pred_tfidf  = pd.read_csv(PREDS_TFIDF)['pred_calibrated'].values"
                    new_str = "pred_tfidf  = pd.read_csv(PREDS_TFIDF).set_index('row_id').loc[test_df['row_id']]['pred_calibrated'].values"
                    if old_str in line:
                        src[i] = line.replace(old_str, new_str)
                        updated = True
                    old_str = "pred_minilm = pd.read_csv(PREDS_MINILM)['pred_calibrated'].values"
                    new_str = "pred_minilm = pd.read_csv(PREDS_MINILM).set_index('row_id').loc[test_df['row_id']]['pred_calibrated'].values"
                    if old_str in line:
                        src[i] = line.replace(old_str, new_str)
                        updated = True
                    old_str = "pred_mpnet  = pd.read_csv(PREDS_MPNET)['pred_calibrated'].values"
                    new_str = "pred_mpnet  = pd.read_csv(PREDS_MPNET).set_index('row_id').loc[test_df['row_id']]['pred_calibrated'].values"
                    if old_str in line:
                        src[i] = line.replace(old_str, new_str)
                        updated = True
    if updated:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Updated 06_llm_vs_baselines.ipynb to join preds on row_id")
    else:
        print("06_llm_vs_baselines.ipynb already up to date")

if __name__ == '__main__':
    update_02()
    update_03()
    update_04()
    update_06()
