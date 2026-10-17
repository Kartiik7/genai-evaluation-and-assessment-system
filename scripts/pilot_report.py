"""
scripts/pilot_report.py

Generates evaluation report for the 30-row dev pilot across LLM methods
(direct, rubric, rag, rag_rubric) and the constant dev-mean baseline.
Saves to results/pilot_dev_summary.csv with note="DEV PILOT n=30, not final".
Does NOT append to results/main_results.csv.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.metrics import evaluate

DEV_FILE = Path("data/processed/dev.csv")
DEV_PILOT_SUMMARY = Path("results/pilot_dev_summary.csv")


def compute_metrics(human: np.ndarray, pred: np.ndarray):
    res = evaluate(human, pred)
    mse_signed = float(np.mean(pred - human))
    return {
        "MAE": round(res["mae"], 4),
        "RMSE": round(res["rmse"], 4),
        "Pearson": round(res["pearson"], 4) if not np.isnan(res["pearson"]) else np.nan,
        "Spearman": round(res["spearman"], 4) if not np.isnan(res["spearman"]) else np.nan,
        "QWK": round(res["qwk"], 4) if not np.isnan(res["qwk"]) else np.nan,
        "Mean Signed Error": round(mse_signed, 4),
    }


def main():
    if not DEV_FILE.exists():
        raise FileNotFoundError(f"Missing {DEV_FILE}")

    dev_df = pd.read_csv(DEV_FILE).sort_values("row_id").reset_index(drop=True)
    dev_30 = dev_df.head(30)
    human_scores = dev_30["human_score"].values

    dev_mean = float(dev_df["human_score"].mean())
    const_pred = np.full(len(dev_30), dev_mean)

    rows = []

    # Constant Dev-Mean Baseline
    base_m = compute_metrics(human_scores, const_pred)
    base_m["Method"] = "constant_dev_mean"
    base_m["Parse Failures"] = 0
    base_m["Note"] = "DEV PILOT n=30, not final"
    rows.append(base_m)

    methods = ["direct", "rubric", "rag", "rag_rubric"]
    rubric_df = None

    for m in methods:
        pred_file = Path(f"results/llm_{m}_dev.csv")
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found. Skipping {m}.")
            continue
        pdf = pd.read_csv(pred_file).sort_values("row_id").reset_index(drop=True)
        # Take first 30 rows
        pdf_30 = pdf.head(30)
        if m == "rubric":
            rubric_df = pdf_30.copy()

        pred_scores = pdf_30["llm_score"].values
        parse_fails = int(pdf_30["parse_failure"].sum()) if "parse_failure" in pdf_30.columns else 0

        m_dict = compute_metrics(human_scores, pred_scores)
        m_dict["Method"] = m
        m_dict["Parse Failures"] = parse_fails
        m_dict["Note"] = "DEV PILOT n=30, not final"
        rows.append(m_dict)

    summary_df = pd.DataFrame(rows)[
        ["Method", "MAE", "RMSE", "Pearson", "Spearman", "QWK", "Mean Signed Error", "Parse Failures", "Note"]
    ]
    DEV_PILOT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(DEV_PILOT_SUMMARY, index=False)

    print("=" * 80)
    print("DEV PILOT EVALUATION SUMMARY (n=30 rows)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Saved to {DEV_PILOT_SUMMARY}")

    # Rubric top 3 largest disagreements
    if rubric_df is not None:
        rubric_df["error_abs"] = (rubric_df["human_score"] - rubric_df["llm_score"]).abs()
        top3_disagree = rubric_df.sort_values("error_abs", ascending=False).head(3)
        print("\nTOP 3 RUBRIC DISAGREEMENTS (|Human - LLM|):")
        for i, (_, row) in enumerate(top3_disagree.iterrows(), 1):
            q = str(row.get("question", ""))[:200]
            ans = str(row.get("student_answer", ""))[:200]
            reason = str(row.get("reasoning", ""))[:200]
            print(f"\n[{i}] Question: {q}")
            print(f"    Student : {ans}")
            print(f"    Human   : {row.get('human_score')}, LLM: {row.get('llm_score')} (Diff: {row['error_abs']:.2f})")
            print(f"    Reason  : {reason}")


if __name__ == "__main__":
    main()
