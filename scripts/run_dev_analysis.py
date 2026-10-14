"""
scripts/run_dev_analysis.py

Runs all 6 analysis sections from notebooks/05_llm_dev_analysis.ipynb as a
standalone script, printing results and saving figures to results/figures/.

Run this if Jupyter/nbconvert is not available.
"""

import sys
import os
import re
import io
import warnings

# Force UTF-8 stdout on Windows to avoid cp1252 encoding errors with special chars
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
from sklearn.metrics import confusion_matrix

from src.metrics import evaluate
from src.llm_eval import RUBRIC_CRITERIA

# Paths
DIRECT_PATH = Path("results/llm_direct_dev.csv")
RUBRIC_PATH = Path("results/llm_rubric_dev.csv")
DEV_PATH    = Path("data/processed/dev.csv")
FIG_DIR     = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"direct": "#4C9BE8", "rubric": "#E87D4C"}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(path):
    df = pd.read_csv(path)
    df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce")
    return df


def compute_metrics(df, label):
    valid = df.dropna(subset=["pred_score", "human_score"])
    n_total = len(df)
    n_parse_fail = int((~df["parse_ok"].astype(bool)).sum()) if "parse_ok" in df.columns else 0
    if len(valid) == 0:
        print(f"  {label}: No valid predictions!")
        return None
    m = evaluate(valid["human_score"].values, valid["pred_score"].values)
    m.update({"n_valid": len(valid), "n_total": n_total,
               "parse_failures": n_parse_fail, "method": label})
    return m


# ============================================================
# SECTION 1
# ============================================================
def section_1(direct, rubric):
    print("\n" + "=" * 65)
    print("SECTION 1: METRICS (MAE, RMSE, Pearson, Spearman, QWK, Within-1)")
    print("=" * 65)

    m_d = compute_metrics(direct, "direct_v1")
    m_r = compute_metrics(rubric, "rubric_v1")

    print(f"{'Metric':<20} {'Direct':>10} {'Rubric':>10}")
    print("-" * 42)
    rows = [
        ("mae",      "MAE (lower=better)"),
        ("rmse",     "RMSE (lower=better)"),
        ("pearson",  "Pearson (higher=better)"),
        ("spearman", "Spearman (higher=better)"),
        ("qwk",      "QWK (higher=better)"),
        ("within_1", "Within-1 (higher=better)"),
    ]
    for col, lbl in rows:
        d = m_d[col] if m_d else float("nan")
        r = m_r[col] if m_r else float("nan")
        print(f"  {lbl:<30} {d:>8.4f} {r:>8.4f}")
    print("-" * 42)
    d_pf = m_d["parse_failures"] if m_d else "N/A"
    r_pf = m_r["parse_failures"] if m_r else "N/A"
    print(f"  {'Parse failures':<30} {str(d_pf):>8} {str(r_pf):>8}")

    if direct.get("simulated", pd.Series(False)).any():
        print("  [NOTE: direct results are SIMULATED (API down)]")
    if rubric.get("simulated", pd.Series(False)).any():
        print("  [NOTE: rubric results are SIMULATED (API down)]")

    # Bar chart
    metric_names  = ["mae", "rmse", "pearson", "spearman", "qwk", "within_1"]
    metric_labels = ["MAE", "RMSE", "Pearson", "Spearman", "QWK", "Within-1"]
    colors = [PALETTE["direct"], PALETTE["rubric"]]
    vals_all = [(m_d[c] if m_d else 0, m_r[c] if m_r else 0) for c in metric_names]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(14, 3.5))
    for ax, lbl, (vd, vr) in zip(axes, metric_labels, vals_all):
        bars = ax.bar(["Direct", "Rubric"], [vd, vr], color=colors, edgecolor="white")
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        for bar, v in zip(bars, [vd, vr]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("LLM Grading -- DEV Split Metrics (SIMULATED)", fontsize=11,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_dev_metrics_bar.png", bbox_inches="tight")
    plt.close()
    print("  -> Saved: results/figures/llm_dev_metrics_bar.png")

    return m_d, m_r


# ============================================================
# SECTION 2
# ============================================================
def section_2(direct, rubric):
    print("\n" + "=" * 65)
    print("SECTION 2: MEAN SIGNED ERROR (Bias Analysis)")
    print("=" * 65)

    results = {}
    for label, df in [("direct_v1", direct), ("rubric_v1", rubric)]:
        valid = df.dropna(subset=["pred_score", "human_score"]).copy()
        valid["signed_error"] = valid["pred_score"] - valid["human_score"]
        mse = valid["signed_error"].mean()
        std = valid["signed_error"].std()
        direction = "LENIENT" if mse > 0.05 else ("STRICT" if mse < -0.05 else "WELL-CALIBRATED")
        print(f"  {label}: mean signed error = {mse:+.4f} +/- {std:.4f}  -> {direction}")
        results[label] = valid

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, (label, df_v, color) in zip(axes, [
        ("Direct (direct_v1)", results["direct_v1"], PALETTE["direct"]),
        ("Rubric (rubric_v1)", results["rubric_v1"], PALETTE["rubric"]),
    ]):
        ax.hist(df_v["signed_error"], bins=15, color=color, edgecolor="white", alpha=0.85)
        mse = df_v["signed_error"].mean()
        ax.axvline(mse, color="black", linestyle="--", linewidth=1.5, label=f"Mean = {mse:+.3f}")
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("Predicted - Human Score", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
    fig.suptitle("Signed Error Distribution -- DEV Split (SIMULATED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_dev_signed_error.png", bbox_inches="tight")
    plt.close()
    print("  -> Saved: results/figures/llm_dev_signed_error.png")
    return results


# ============================================================
# SECTION 3
# ============================================================
def section_3(direct, rubric):
    print("\n" + "=" * 65)
    print("SECTION 3: CONFUSION MATRIX (Rounded Human vs Rounded Predicted)")
    print("=" * 65)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = list(range(0, 6))

    for ax, (df, label) in zip(axes, [
        (direct, "Direct (direct_v1)"),
        (rubric, "Rubric (rubric_v1)"),
    ]):
        valid = df.dropna(subset=["pred_score", "human_score"])
        h_int = valid["human_score"].round().clip(0, 5).astype(int)
        p_int = valid["pred_score"].round().clip(0, 5).astype(int)
        cm = confusion_matrix(h_int, p_int, labels=labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(cm.sum(axis=1, keepdims=True) > 0,
                               cm / cm.sum(axis=1, keepdims=True), 0)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    linewidths=0.4, xticklabels=labels, yticklabels=labels,
                    ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted (rounded)", fontsize=9)
        ax.set_ylabel("Human (rounded)", fontsize=9)

        exact_pct = 100 * np.diag(cm).sum() / cm.sum()
        within1 = sum(cm[i, j] for i in range(len(labels))
                      for j in range(len(labels)) if abs(labels[i] - labels[j]) <= 1)
        print(f"  {label}: exact match = {exact_pct:.1f}%,  "
              f"within-1 = {100*within1/cm.sum():.1f}%")

    fig.suptitle("Confusion Matrix -- DEV Split (SIMULATED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_dev_confusion.png", bbox_inches="tight")
    plt.close()
    print("  -> Saved: results/figures/llm_dev_confusion.png")


# ============================================================
# SECTION 4
# ============================================================
def section_4(direct, rubric, dev_data):
    print("\n" + "=" * 65)
    print("SECTION 4: TOP-5 LARGEST DISAGREEMENTS")
    print("=" * 65)

    for label, df in [("Direct", direct), ("Rubric", rubric)]:
        valid = df.dropna(subset=["pred_score", "human_score"]).copy()
        valid["abs_error"] = (valid["pred_score"] - valid["human_score"]).abs()
        top = valid.nlargest(5, "abs_error").reset_index(drop=True)

        print(f"\n  --- {label}: Top-5 Disagreements ---")
        for _, row in top.iterrows():
            qid = row["question_id"]
            dev_row = dev_data[dev_data["question_id"] == qid]
            q   = dev_row["question"].values[0][:80] if not dev_row.empty else "N/A"
            ref = dev_row["reference_answer"].values[0][:80] if not dev_row.empty else "N/A"
            stu = dev_row["student_answer"].values[0][:80] if not dev_row.empty else "N/A"
            delta = row["pred_score"] - row["human_score"]
            print(f"    QID={qid}  Human={row['human_score']:.1f}  LLM={row['pred_score']:.1f}  "
                  f"Delta={delta:+.1f}")
            print(f"    Q:        {q}")
            print(f"    Ref:      {ref}")
            print(f"    Student:  {stu}")
            print(f"    Reasoning:{str(row.get('reasoning', ''))[:150]}")
            print()


# ============================================================
# SECTION 5
# ============================================================
def section_5(rubric):
    print("\n" + "=" * 65)
    print("SECTION 5: RUBRIC CRITERION ANALYSIS")
    print("=" * 65)

    crit_cols = [f"score_{c}" for c in RUBRIC_CRITERIA]
    available = [c for c in crit_cols if c in rubric.columns]
    valid = rubric.dropna(subset=["human_score", "pred_score"])

    crit_stats = []
    for crit, col in zip(RUBRIC_CRITERIA, available):
        vals = pd.to_numeric(valid[col], errors="coerce").values
        h = valid["human_score"].values
        mask = ~np.isnan(vals)
        if mask.sum() > 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = pearsonr(h[mask], vals[mask])
        else:
            corr = float("nan")
        row = {
            "criterion": crit,
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "pearson_with_human": corr,
            "n_valid": int(mask.sum()),
        }
        crit_stats.append(row)
        print(f"  {crit:<15} mean={row['mean']:.3f}  std={row['std']:.3f}  "
              f"r_with_human={corr:.3f}  n={row['n_valid']}")

    print(f"\n  Overall rubric pred mean : {valid['pred_score'].mean():.3f}")
    print(f"  Human score mean         : {valid['human_score'].mean():.3f}")

    crit_df = pd.DataFrame(crit_stats).set_index("criterion")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    y = range(len(RUBRIC_CRITERIA))
    bars = ax.barh(y, crit_df["mean"], xerr=crit_df["std"], color=PALETTE["rubric"],
                   alpha=0.8, edgecolor="white", capsize=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels([c.capitalize() for c in RUBRIC_CRITERIA], fontsize=10)
    ax.set_xlabel("Mean Score (0-5)", fontsize=10)
    ax.set_title("Mean Criterion Scores +/- SD", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 5.5)
    for bar, v in zip(bars, crit_df["mean"]):
        ax.text(v + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=9)

    ax2 = axes[1]
    corrs = crit_df["pearson_with_human"].values
    colors_c = [PALETTE["rubric"] if c >= 0 else "#E84C4C" for c in corrs]
    bars2 = ax2.barh(y, corrs, color=colors_c, alpha=0.85, edgecolor="white")
    ax2.set_yticks(list(y))
    ax2.set_yticklabels([c.capitalize() for c in RUBRIC_CRITERIA], fontsize=10)
    ax2.set_xlabel("Pearson r with Human Score", fontsize=10)
    ax2.set_title("Criterion-Human Correlation", fontsize=11, fontweight="bold")
    ax2.axvline(0, color="grey", linestyle=":", linewidth=1)
    for bar, v in zip(bars2, corrs):
        offset = 0.02 if v >= 0 else -0.08
        ax2.text(v + offset, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Rubric Criterion Analysis -- DEV Split (SIMULATED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "llm_dev_rubric_criteria.png", bbox_inches="tight")
    plt.close()
    print("  -> Saved: results/figures/llm_dev_rubric_criteria.png")


# ============================================================
# SECTION 6
# ============================================================
def section_6(dev_data, direct, rubric):
    print("\n" + "=" * 65)
    print("SECTION 6: NO ANSWER / EMPTY RESPONSE COUNTS")
    print("=" * 65)

    NO_ANSWER_PATTERNS = [
        r"^\s*$",
        r"^\s*(no\s+answer|n/a|not\s+answered?|no\s+response|blank)\s*$",
    ]

    def is_no_answer(text):
        if not isinstance(text, str):
            return True
        for pat in NO_ANSWER_PATTERNS:
            if re.match(pat, text.strip(), re.IGNORECASE):
                return True
        return False

    dev_data = dev_data.copy()
    dev_data["is_no_answer"] = dev_data["student_answer"].apply(is_no_answer)
    n_no_answer = int(dev_data["is_no_answer"].sum())

    print(f"  Total dev responses  : {len(dev_data)}")
    print(f"  NO ANSWER / empty    : {n_no_answer} ({100*n_no_answer/len(dev_data):.1f}%)")

    if n_no_answer > 0:
        for _, r in dev_data[dev_data["is_no_answer"]].iterrows():
            print(f"    QID={r['question_id']}  human_score={r['human_score']}  "
                  f"student_answer={repr(str(r['student_answer'])[:60])}")
    else:
        print("  No empty or NO ANSWER responses detected.")

    print("\n  Zero-score items (human_score == 0):")
    for label, df in [("direct", direct), ("rubric", rubric)]:
        zero = df[df["human_score"] == 0].copy()
        print(f"    {label}: {len(zero)} items with human_score=0")
        if len(zero) > 0:
            vz = zero.dropna(subset=["pred_score"])
            print(f"      LLM pred mean on 0-score items : {vz['pred_score'].mean():.3f}")
            also_zero = int((vz["pred_score"].round() == 0).sum())
            print(f"      LLM also scored 0              : {also_zero} / {len(vz)}")

    return n_no_answer


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading data...")
    direct   = load(DIRECT_PATH)
    rubric   = load(RUBRIC_PATH)
    dev_data = pd.read_csv(DEV_PATH)

    is_simulated = bool(direct.get("simulated", pd.Series(False)).any())
    if is_simulated:
        print("\n[WARNING] Using SIMULATED results (Gemini API was unavailable).")
        print("   Model field: MOCK_minilm_proxy (MiniLM calibrated predictions + Gaussian noise)")
        print("   Re-run scripts/run_llm_eval.py when API recovers to get real LLM results.\n")

    m_d, m_r    = section_1(direct, rubric)
    signed      = section_2(direct, rubric)
    section_3(direct, rubric)
    section_4(direct, rubric, dev_data)
    section_5(rubric)
    n_no_answer = section_6(dev_data, direct, rubric)

    # Final summary
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<26} {'Direct':>10} {'Rubric':>10}")
    print("-" * 48)
    for col, lbl in [
        ("mae",      "MAE (lower=better)"),
        ("rmse",     "RMSE (lower=better)"),
        ("pearson",  "Pearson (higher=better)"),
        ("spearman", "Spearman (higher=better)"),
        ("qwk",      "QWK (higher=better)"),
        ("within_1", "Within-1 (higher=better)"),
    ]:
        d = m_d[col] if m_d else float("nan")
        r = m_r[col] if m_r else float("nan")
        print(f"  {lbl:<24} {d:>10.4f} {r:>10.4f}")
    print("-" * 48)
    pf_d = m_d["parse_failures"] if m_d else "N/A"
    pf_r = m_r["parse_failures"] if m_r else "N/A"
    print(f"  {'Parse failures':<24} {str(pf_d):>10} {str(pf_r):>10}")
    print("-" * 48)
    for lbl, df_v in [("Direct bias", signed["direct_v1"]), ("Rubric bias", signed["rubric_v1"])]:
        mse = df_v["signed_error"].mean()
        direction = "lenient" if mse > 0.05 else ("strict" if mse < -0.05 else "well-calibrated")
        print(f"  {lbl:<24} {mse:>+10.4f}  ({direction})")
    print(f"\n  NO ANSWER in dev : {n_no_answer} / {len(dev_data)}")
    if is_simulated:
        print("\n  [WARNING] All results above are SIMULATED. Rerun with real API when available.")
    print("=" * 65)


if __name__ == "__main__":
    main()
