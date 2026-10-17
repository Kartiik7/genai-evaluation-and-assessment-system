# System Status & Project Overview

**Date**: October 17, 2026  
**Project**: GenAI Evaluation & Assessment System for Short-Answer Grading  
**Dataset**: Mohler Short-Answer Grading Dataset (C++ Data Structures & CS Fundamentals)  

---

## 1. End-to-End Pipeline Architecture

```text
[ Raw Mohler Data (2,273 rows) ]
               │
               ▼
   [ Stratified Splitting ] ─────────► [ Dev Split: 50 rows ] (Tuning, LR Calibration)
               │                     ► [ Test Split: 150 rows ] (Final Frozen Evaluation)
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Deterministic Scoring Baselines                       │
│  - Constant Dev-Mean (Dummy baseline)                                       │
│  - TF-IDF Cosine Similarity (Lexical n-gram overlap)                        │
│  - Sentence-BERT Cosine: all-MiniLM-L6-v2 & all-mpnet-base-v2               │
│  - Linear Regression Calibration (Dev-fitted mapping onto [0, 5] scale)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GenAI LLM Grading Pipeline                         │
│  Model: gemini-3.8-flash (temp=0.0, JSON mode, SHA-256 disk cache)          │
│                                                                             │
│  Method 1: Direct Prompting (direct_v2.txt)                                 │
│  Method 2: Multi-Criteria Rubric (rubric_v2.txt)                            │
│  Method 3: Retrieval-Augmented Grading (rag_v2.txt)                         │
│  Method 4: RAG + Rubric Joint Assessment (rag_rubric_v2.txt)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Retrieval Subsystem (Dense RAG)                       │
│  - Knowledge Base: 12 Markdown topic files, 21 semantic chunks (~191 words) │
│  - Embedding Index: all-MiniLM-L6-v2 + FAISS IndexFlatIP (Inner Product)    │
│  - Query Formulation: Question text only (Strict top-3 chunk cutoff)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Evaluation & Audit Metrics                           │
│  - Regression & Ranking: MAE, RMSE, Pearson r, Spearman rho, QWK, Within+-1 │
│  - Statistical Rigor: 1,000-replicate empirical percentile bootstrap CIs    │
│  - Retrieval Performance: Precision@k, Recall@k, MRR vs Random Baseline     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Experimental Results Summary

### 2.1 Baselines on Test Set ($N = 150$, with 95% Bootstrap CIs)
*Exact values reproduced from `results/baseline_table.csv` ($B = 1,000$ resamples, seed 42):*

| Method | Scaling / Variant | MAE [95% CI] | RMSE [95% CI] | Pearson $r$ [95% CI] | Spearman $\rho$ | QWK $\kappa$ [95% CI] | Within $\pm 1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **constant_mean** | Dev Mean Baseline | 0.926 [0.810, 1.041] | 1.155 [0.969, 1.334] | — | — | 0.000 [0.000, 0.000] | 78.7% |
| **TF-IDF** | Scaled ($5 \times \cos$) | 2.367 | 2.729 | 0.168 | 0.229 | 0.085 | 20.7% |
| **TF-IDF** | Calibrated (OLS) | 0.870 [0.742, 1.003] | 1.180 [0.960, 1.392] | 0.178 [0.004, 0.361] | 0.231 | 0.163 [0.006, 0.352] | 62.7% |
| **MiniLM-L6-v2** | Scaled ($5 \times \cos$) | 1.464 | 1.756 | 0.521 | 0.502 | 0.317 | 36.7% |
| **MiniLM-L6-v2** | Calibrated (OLS) | **0.781** [0.688, 0.879] | **0.989** [0.877, 1.095] | **0.524** [0.387, 0.641] | **0.502** | **0.400** [0.296, 0.493] | 64.7% |
| **MPNet-base-v2** | Scaled ($5 \times \cos$) | 1.366 | 1.650 | 0.521 | 0.479 | 0.344 | 39.3% |
| **MPNet-base-v2** | Calibrated (OLS) | 0.800 [0.708, 0.898] | 0.988 [0.874, 1.100] | 0.522 [0.364, 0.642] | 0.480 | 0.369 [0.253, 0.462] | **69.3%** |

### 2.2 LLM Methods Evaluation
*Status: Full 30-row dev pilot and 150-row test evaluations pending execution.*

| Method | Split | MAE | RMSE | Pearson $r$ | Spearman $\rho$ | QWK $\kappa$ | Parse Failures | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct (`direct_v2`)** | Dev pilot ($n=30$) | — | — | — | — | — | — | *not yet run* |
| **Rubric (`rubric_v2`)** | Dev pilot ($n=30$) | — | — | — | — | — | — | *not yet run* |
| **RAG (`rag_v2`)** | Dev pilot ($n=30$) | — | — | — | — | — | — | *not yet run* |
| **RAG Rubric (`rag_rubric_v2`)** | Dev pilot ($n=30$) | — | — | — | — | — | — | *not yet run* |

---

## 3. Knowledge Base & Retrieval Performance

*Source: `results/retrieval_results.csv` ($N = 30$ blinded test questions, 300 judged pairs)*

| Method | Precision@1 | Precision@3 | Precision@5 | Recall@3 | Recall@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retriever (`all-MiniLM-L6-v2`)** | **0.8333** | **0.3667** | **0.2267** | **0.9833** | **1.0000** | **0.9111** |
| **Random Baseline (1,000 runs, seed 42)** | 0.0537 | 0.1433 | 0.0542 | 0.1433 | 0.2394 | 0.1541 |

*Note on Knowledge Base Scale*: The current knowledge base comprises **21 chunks** across 12 core computer science markdown documents in `data/kb/raw/`. While the index achieved 100% top-3 coverage across the 30 evaluated questions (0 failures), expanding the KB to cover additional peripheral curriculum topics is planned.

---

## 4. Known Limitations

1. **Small Knowledge Base**: Contains 21 chunks (~4,000 words total). While sufficient for the sampled 30 test questions, broader CS topic questions may encounter vocabulary or concept gaps.
2. **Single Holistic Human Score**: Mohler ground truth provides only a single overall score (0.0–5.0). Sub-criteria scores (correctness, completeness, relevance, understanding, clarity) from rubric grading cannot be directly cross-validated against human sub-criterion ratings.
3. **Single LLM Evaluator**: Evaluations rely solely on `gemini-3.8-flash`. Cross-model validation (e.g. Claude 3.5 Sonnet, GPT-4o) has not yet been performed.
4. **Noisy Human Annotations**: Mohler dataset labels exhibit inherent annotator variance across questions, capping theoretical upper-bound correlation at inter-annotator agreement levels ($\rho \approx 0.60\text{--}0.70$).

---

## 5. Remaining Work Checklist

- [ ] **Freeze Prompts**: Formally lock `v2` prompt family across `direct_v2`, `rubric_v2`, `rag_v2`, and `rag_rubric_v2`.
- [ ] **Dev Pilot Execution**: Complete the $n=30$ dev pilot run with `scripts/run_llm_eval.py` and compile `results/pilot_dev_summary.csv`.
- [ ] **Run Test Split Once**: Execute single test-split evaluation ($N = 150$) across all 4 LLM methods with locked prompts.
- [ ] **Consistency Runs**: Measure prompt stochasticity / temperature stability across repeated seed runs.
- [ ] **Feedback Rating**: Perform qualitative assessment of generated student feedback and missing-concept diagnoses.
- [ ] **Failure Analysis**: Systematically audit the top residual error cases (|Human - LLM| > 1.5).
- [ ] **Expand Knowledge Base**: Grow raw topic markdown documents from 21 chunks to 60+ chunks to provide comprehensive curriculum coverage.
- [ ] **Sensitivity Analysis**: Measure robustness to chunk window sizes, top-$k$ retrieval cutoffs, and prompt variations.
