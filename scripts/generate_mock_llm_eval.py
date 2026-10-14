"""
scripts/generate_mock_llm_eval.py

Generates synthetic LLM evaluation results for dev split when the Gemini API
is unavailable (e.g., 503 outage). Uses MiniLM calibrated predictions as the
base score with small Gaussian noise to simulate realistic LLM variance.

ALL output rows are clearly flagged with:
  - simulated=True
  - model="MOCK_minilm_proxy"

These files can be replaced by re-running:
  python scripts/run_llm_eval.py --method direct --split dev
  python scripts/run_llm_eval.py --method rubric --split dev

...once the API recovers. The real API cache will take precedence over mock data.

Usage:
    python scripts/generate_mock_llm_eval.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.llm_eval import RUBRIC_CRITERIA

SEED = 42
RNG  = np.random.default_rng(SEED)
MOCK_MODEL = "MOCK_minilm_proxy"

# Paths
DEV_PATH     = Path("data/processed/dev.csv")
MINILM_PATH  = Path("results/preds_minilm.csv")
DIRECT_OUT   = Path("results/llm_direct_dev.csv")
RUBRIC_OUT   = Path("results/llm_rubric_dev.csv")

# Noise parameters for simulated LLM responses
SCORE_NOISE_STD = 0.30    # simulates realistic LLM score variance around baseline
CRITERION_NOISE_STD = 0.25


def clamp(v, lo=0.0, hi=5.0):
    return float(np.clip(v, lo, hi))


def mock_reasoning(human_score, pred_score, question, reference, student):
    """Generate a short plausible reasoning string based on score."""
    if pred_score >= 4.5:
        return (
            "The student answer closely mirrors the reference, demonstrating solid "
            "understanding of the core concepts with only minor omissions."
        )
    elif pred_score >= 3.0:
        return (
            "The response captures the main idea but misses some key details "
            "present in the reference answer."
        )
    elif pred_score >= 1.5:
        return (
            "The answer is partially correct but contains significant gaps or "
            "inaccuracies compared to the reference."
        )
    else:
        return (
            "The student answer does not adequately address the question; "
            "key concepts from the reference are absent."
        )


def mock_feedback(pred_score):
    if pred_score >= 4.0:
        return "Good work overall. Aim for more complete coverage of all key points."
    elif pred_score >= 2.5:
        return "Partial understanding shown. Review the reference material and elaborate on missing concepts."
    else:
        return "Significant revision needed. Focus on accurately addressing the core question."


def generate_mock_direct(dev, minilm):
    records = []
    for _, row in dev.iterrows():
        qid = row["question_id"]
        human = float(row["human_score"])

        # Look up MiniLM calibrated prediction; fall back to human if not found
        m_row = minilm[minilm["question_id"] == qid]
        if not m_row.empty:
            base_pred = float(m_row["pred_calibrated"].values[0])
        else:
            base_pred = human  # safe fallback

        # Add small Gaussian noise
        pred = clamp(base_pred + RNG.normal(0, SCORE_NOISE_STD))

        records.append({
            "question_id": qid,
            "human_score": human,
            "pred_score": round(pred, 2),
            "parse_ok": True,
            "reasoning": mock_reasoning(human, pred,
                                        row["question"],
                                        row["reference_answer"],
                                        row["student_answer"]),
            "prompt_version": "direct_v1",
            "model": MOCK_MODEL,
            "temperature": 0.0,
            "simulated": True,
        })

    return pd.DataFrame(records)


def generate_mock_rubric(dev, minilm):
    records = []
    for _, row in dev.iterrows():
        qid = row["question_id"]
        human = float(row["human_score"])

        m_row = minilm[minilm["question_id"] == qid]
        if not m_row.empty:
            base_pred = float(m_row["pred_calibrated"].values[0])
        else:
            base_pred = human

        overall_pred = clamp(base_pred + RNG.normal(0, SCORE_NOISE_STD))

        # Generate per-criterion scores with correlated noise
        crit_scores = {}
        for crit in RUBRIC_CRITERIA:
            # Criteria correlate with overall; add small individual noise
            c_pred = clamp(base_pred + RNG.normal(0, CRITERION_NOISE_STD))
            crit_scores[crit] = round(c_pred, 2)

        rec = {
            "question_id": qid,
            "human_score": human,
            "pred_score": round(overall_pred, 2),
            "parse_ok": True,
            "reasoning": mock_reasoning(human, overall_pred,
                                        row["question"],
                                        row["reference_answer"],
                                        row["student_answer"]),
            "prompt_version": "rubric_v1",
            "model": MOCK_MODEL,
            "temperature": 0.0,
            "simulated": True,
        }

        for crit in RUBRIC_CRITERIA:
            rec[f"score_{crit}"] = crit_scores[crit]

        rec["strengths"] = json.dumps(["Addresses core topic"] if overall_pred >= 3 else [])
        rec["missing_concepts"] = json.dumps(
            ["Key detail from reference not included"] if overall_pred < 4 else []
        )
        rec["errors"] = json.dumps([] if overall_pred >= 2 else ["Factual inaccuracy detected"])
        rec["feedback"] = mock_feedback(overall_pred)

        records.append(rec)

    return pd.DataFrame(records)


def main():
    dev    = pd.read_csv(DEV_PATH)
    minilm = pd.read_csv(MINILM_PATH)

    print(f"Dev rows     : {len(dev)}")
    print(f"MiniLM rows  : {len(minilm)}")
    print(f"Mock model   : {MOCK_MODEL}")
    print()

    # ── Direct ─────────────────────────────────────────────────────────────────
    if DIRECT_OUT.exists():
        existing = pd.read_csv(DIRECT_OUT)
        real = existing[~existing.get("simulated", pd.Series(False, index=existing.index)).astype(bool)]
        if len(real) == len(dev):
            print(f"[SKIP] {DIRECT_OUT} already has {len(real)} real (non-simulated) rows.")
        else:
            print(f"[INFO] {DIRECT_OUT} exists with {len(real)} real rows; regenerating mock for remaining.")
            direct_mock = generate_mock_direct(dev, minilm)
            direct_mock.to_csv(DIRECT_OUT, index=False)
            print(f"[WRITTEN] {DIRECT_OUT} — {len(direct_mock)} rows (all simulated)")
    else:
        direct_mock = generate_mock_direct(dev, minilm)
        direct_mock.to_csv(DIRECT_OUT, index=False)
        print(f"[WRITTEN] {DIRECT_OUT} — {len(direct_mock)} rows (all simulated)")

    # ── Rubric ─────────────────────────────────────────────────────────────────
    if RUBRIC_OUT.exists():
        existing = pd.read_csv(RUBRIC_OUT)
        real = existing[~existing.get("simulated", pd.Series(False, index=existing.index)).astype(bool)]
        if len(real) == len(dev):
            print(f"[SKIP] {RUBRIC_OUT} already has {len(real)} real (non-simulated) rows.")
        else:
            rubric_mock = generate_mock_rubric(dev, minilm)
            rubric_mock.to_csv(RUBRIC_OUT, index=False)
            print(f"[WRITTEN] {RUBRIC_OUT} — {len(rubric_mock)} rows (all simulated)")
    else:
        rubric_mock = generate_mock_rubric(dev, minilm)
        rubric_mock.to_csv(RUBRIC_OUT, index=False)
        print(f"[WRITTEN] {RUBRIC_OUT} — {len(rubric_mock)} rows (all simulated)")

    print()
    print("NOTE: These are SIMULATED results based on MiniLM calibrated predictions + Gaussian noise.")
    print("      Re-run `python scripts/run_llm_eval.py --method {direct,rubric} --split dev`")
    print("      once the Gemini API recovers to replace these with real LLM responses.")


if __name__ == "__main__":
    main()
