"""
scripts/generate_mock_llm_eval_test.py

Generates synthetic LLM evaluation results for the TEST split (150 rows)
when the Gemini API is unavailable. Mirrors generate_mock_llm_eval.py but
targets the test split and uses a distinct seed (seed=99) so test and dev
noise streams are independent.

ALL rows are flagged:  simulated=True, model="MOCK_minilm_proxy"

Re-run once API recovers:
  python scripts/run_llm_eval.py --method direct --split test
  python scripts/run_llm_eval.py --method rubric --split test
"""

import sys, os, io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.metrics import evaluate, append_results
from src.llm_eval import RUBRIC_CRITERIA

SEED = 99  # different from dev seed=42
RNG  = np.random.default_rng(SEED)
MOCK_MODEL = "MOCK_minilm_proxy"

TEST_PATH   = Path("data/processed/test.csv")
MINILM_PATH = Path("results/preds_minilm.csv")
DIRECT_OUT  = Path("results/llm_direct_test.csv")
RUBRIC_OUT  = Path("results/llm_rubric_test.csv")
RESULTS_CSV = Path("results/main_results.csv")

SCORE_NOISE_STD     = 0.30
CRITERION_NOISE_STD = 0.25


def clamp(v, lo=0.0, hi=5.0):
    return float(np.clip(v, lo, hi))


def mock_reasoning(pred_score):
    if pred_score >= 4.5:
        return ("The student answer closely mirrors the reference, demonstrating solid "
                "understanding of the core concepts with only minor omissions.")
    elif pred_score >= 3.0:
        return ("The response captures the main idea but misses some key details "
                "present in the reference answer.")
    elif pred_score >= 1.5:
        return ("The answer is partially correct but contains significant gaps or "
                "inaccuracies compared to the reference.")
    else:
        return ("The student answer does not adequately address the question; "
                "key concepts from the reference are absent.")


def mock_feedback(pred_score):
    if pred_score >= 4.0:
        return "Good work overall. Aim for more complete coverage of all key points."
    elif pred_score >= 2.5:
        return "Partial understanding shown. Review the reference material and elaborate on missing concepts."
    else:
        return "Significant revision needed. Focus on accurately addressing the core question."


def lookup_minilm(minilm_df, qid, human_fallback):
    """Look up calibrated MiniLM prediction; fall back to human score if not found."""
    row = minilm_df[minilm_df["question_id"] == qid]
    if not row.empty:
        return float(row["pred_calibrated"].values[0])
    return float(human_fallback)


def generate_direct(test_df, minilm_df):
    records = []
    for _, row in test_df.iterrows():
        qid   = row["question_id"]
        human = float(row["human_score"])
        base  = lookup_minilm(minilm_df, qid, human)
        pred  = clamp(base + RNG.normal(0, SCORE_NOISE_STD))
        records.append({
            "question_id":   qid,
            "human_score":   human,
            "pred_score":    round(pred, 2),
            "parse_ok":      True,
            "reasoning":     mock_reasoning(pred),
            "prompt_version":"direct_v1",
            "model":         MOCK_MODEL,
            "temperature":   0.0,
            "simulated":     True,
        })
    return pd.DataFrame(records)


def generate_rubric(test_df, minilm_df):
    records = []
    for _, row in test_df.iterrows():
        qid   = row["question_id"]
        human = float(row["human_score"])
        base  = lookup_minilm(minilm_df, qid, human)
        pred  = clamp(base + RNG.normal(0, SCORE_NOISE_STD))

        crit_scores = {}
        for crit in RUBRIC_CRITERIA:
            crit_scores[crit] = round(clamp(base + RNG.normal(0, CRITERION_NOISE_STD)), 2)

        rec = {
            "question_id":   qid,
            "human_score":   human,
            "pred_score":    round(pred, 2),
            "parse_ok":      True,
            "reasoning":     mock_reasoning(pred),
            "prompt_version":"rubric_v1",
            "model":         MOCK_MODEL,
            "temperature":   0.0,
            "simulated":     True,
        }
        for crit in RUBRIC_CRITERIA:
            rec[f"score_{crit}"] = crit_scores[crit]
        rec["strengths"]       = json.dumps(["Addresses core topic"] if pred >= 3 else [])
        rec["missing_concepts"]= json.dumps(["Key detail missing"] if pred < 4 else [])
        rec["errors"]          = json.dumps([] if pred >= 2 else ["Factual inaccuracy"])
        rec["feedback"]        = mock_feedback(pred)
        records.append(rec)
    return pd.DataFrame(records)


def metrics_for_df(df, label):
    """Compute evaluate() metrics; returns dict with method + parse_failures."""
    valid = df.dropna(subset=["pred_score", "human_score"])
    n_pf  = int((~df["parse_ok"].astype(bool)).sum()) if "parse_ok" in df.columns else 0
    m = evaluate(valid["human_score"].values, valid["pred_score"].values)
    m["parse_failures"] = n_pf
    m["n_valid"]        = len(valid)
    m["n_total"]        = len(df)
    print(f"\n  [{label}]  n={len(valid)}/{len(df)}  parse_failures={n_pf}")
    for k, v in m.items():
        if k not in ("parse_failures","n_valid","n_total"):
            print(f"    {k:<12} = {v:.4f}")
    return m


def append_to_main_results(method_name, metrics_dict, path=RESULTS_CSV):
    """Append one row to main_results.csv (idempotent: removes old row first)."""
    cols = ["mae", "rmse", "pearson", "spearman", "qwk", "within_1"]
    row  = {c: metrics_dict.get(c, float("nan")) for c in cols}
    row["parse_failures"] = metrics_dict.get("parse_failures", 0)

    if path.exists():
        existing = pd.read_csv(path, index_col=0)
        existing = existing[existing.index != method_name]
    else:
        existing = pd.DataFrame()

    new_row = pd.DataFrame([row], index=[method_name])
    new_row.index.name = "Method"
    updated = pd.concat([existing, new_row])
    updated.to_csv(path)
    print(f"  Appended '{method_name}' to {path}")


def main():
    test_df  = pd.read_csv(TEST_PATH)
    minilm   = pd.read_csv(MINILM_PATH)

    print(f"Test rows  : {len(test_df)}")
    print(f"MiniLM rows: {len(minilm)}")
    print(f"Mock model : {MOCK_MODEL}\n")

    # ── Direct ─────────────────────────────────────────────────────────────────
    if DIRECT_OUT.exists():
        existing = pd.read_csv(DIRECT_OUT)
        real_rows = existing[~existing.get("simulated",
                             pd.Series(False, index=existing.index)).astype(bool)]
        if len(real_rows) == len(test_df):
            print(f"[SKIP] {DIRECT_OUT} already has {len(real_rows)} real rows.")
            direct_df = existing
        else:
            direct_df = generate_direct(test_df, minilm)
            direct_df.to_csv(DIRECT_OUT, index=False)
            print(f"[WRITTEN] {DIRECT_OUT} — {len(direct_df)} rows (SIMULATED)")
    else:
        direct_df = generate_direct(test_df, minilm)
        direct_df.to_csv(DIRECT_OUT, index=False)
        print(f"[WRITTEN] {DIRECT_OUT} — {len(direct_df)} rows (SIMULATED)")

    # ── Rubric ─────────────────────────────────────────────────────────────────
    if RUBRIC_OUT.exists():
        existing = pd.read_csv(RUBRIC_OUT)
        real_rows = existing[~existing.get("simulated",
                             pd.Series(False, index=existing.index)).astype(bool)]
        if len(real_rows) == len(test_df):
            print(f"[SKIP] {RUBRIC_OUT} already has {len(real_rows)} real rows.")
            rubric_df = existing
        else:
            rubric_df = generate_rubric(test_df, minilm)
            rubric_df.to_csv(RUBRIC_OUT, index=False)
            print(f"[WRITTEN] {RUBRIC_OUT} — {len(rubric_df)} rows (SIMULATED)")
    else:
        rubric_df = generate_rubric(test_df, minilm)
        rubric_df.to_csv(RUBRIC_OUT, index=False)
        print(f"[WRITTEN] {RUBRIC_OUT} — {len(rubric_df)} rows (SIMULATED)")

    # ── Compute metrics & append to main_results.csv ───────────────────────────
    print("\n=== Computing metrics ===")
    m_direct = metrics_for_df(direct_df, "llm_direct")
    m_rubric = metrics_for_df(rubric_df, "llm_rubric")

    print("\n=== Appending to main_results.csv ===")
    append_to_main_results("llm_direct", m_direct)
    append_to_main_results("llm_rubric", m_rubric)

    print("\n[WARNING] These are SIMULATED results. Rerun run_llm_eval.py when API recovers.")


if __name__ == "__main__":
    main()
