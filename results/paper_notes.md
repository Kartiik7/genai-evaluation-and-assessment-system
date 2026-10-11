# Paper & Experimental Notes: Phase II Baseline Evaluation

**Date Tested**: October 11, 2026  
**Primary Frameworks & Libraries**:
- `google-genai==2.20.0` (Official Google GenAI SDK for Gemini API)
- `sentence-transformers==3.4.1` (with CPU-only `torch` installed as an inference dependency, not a training framework)
- `scikit-learn==1.6.1`, `scipy==1.15.2`, `pandas==2.2.3`, `numpy==2.2.3`

---

## 1. Exact Model Identifiers & Hugging Face Hub IDs

| Model Role | Canonical Name | Exact Hub / API Identifier | Parameters / Arch |
|---|---|---|---|
| **Semantic Embedding 1** | MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M params, 384 dim, 6 layers |
| **Semantic Embedding 2** | MPNet-base-v2 | `sentence-transformers/all-mpnet-base-v2` | 109M params, 768 dim, 12 layers |
| **LLM Evaluator** | Gemini Flash | `gemini-3.8-flash` | Gemini 3.8 Flash model via Google GenAI SDK |

*Note on PyTorch*: PyTorch is installed strictly as an inference dependency for `sentence-transformers`, not utilized as an active model-training or fine-tuning framework in this phase.

---

## 2. Dataset Split & Setup
- **Dataset**: Mohler Short Answer Grading Dataset (2,273 student responses across 81 distinct questions, graded on 0–5 scale).
- **Stratified Dev Set**: 50 instances sampled across question IDs (`random_state=42`).
- **Stratified Test Set**: 150 instances disjoint from dev (`random_state=42`).
- **Data Leakage Safeguard**: TF-IDF vocabulary, Linear Regression parameters, and scaling thresholds are fitted exclusively on the DEV set and never fit on the TEST set.

---

## 3. Calibration & Scoring Methodology
1. **Raw / Scaled Baseline**:
   - Cosine similarity clipped to $[0.0, 1.0]$.
   - Linear scale: $\text{pred} = 5.0 \times \cos(\text{reference}, \text{student})$.
2. **Calibrated Baseline**:
   - Fitted ordinary least squares linear regression (`LinearRegression`) on DEV split:
     $$\hat{y} = \beta_0 + \beta_1 \cdot \cos(\text{reference}, \text{student})$$
   - Applied to TEST set similarities, clipped strictly to $[0.0, 5.0]$.
   - Prevents systematic over-penalization inherent in raw uncalibrated cosine scores.

---

## 4. Bootstrap Confidence Interval Setup
- **Number of Resamples**: $B = 1,000$ iterations.
- **Random Seed**: `np.random.default_rng(42)`.
- **Method**: Empirical percentile bootstrap interval (2.5th and 97.5th percentiles) computed over the 150 test instances.
- **Metrics Evaluated**: MAE, RMSE, Pearson ($r$), Spearman ($\rho$), and QWK ($\kappa$).
- **Paired Hypothesis Testing**: `paired_bootstrap_diff(human, pred_a, pred_b, metric, n=1000, seed=42)` evaluating differences on identical resampled indices.

---

## 5. Recorded Baseline Evaluation Results (Test Set, N = 150)

| Method | Scaling Type | MAE [95% CI] | RMSE [95% CI] | Pearson [95% CI] | Spearman [95% CI] | QWK [95% CI] | Within $\pm 1$ |
|---|---|---|---|---|---|---|---|
| **constant_mean** | Dev Mean (4.13) | 0.926 [0.810, 1.041] | 1.155 [0.969, 1.334] | — | — | 0.000 [0.000, 0.000] | 78.7% |
| **TF-IDF** | Scaled ($5 \times \cos$) | 2.367 | 2.729 | 0.168 | 0.229 | 0.085 | 20.7% |
| **TF-IDF** | Calibrated (OLS) | 0.870 [0.742, 1.003] | 1.180 [0.960, 1.392] | 0.178 [0.004, 0.361] | 0.231 [0.065, 0.381] | 0.163 [0.006, 0.352] | 62.7% |
| **MiniLM-L6-v2** | Scaled ($5 \times \cos$) | 1.464 | 1.756 | 0.521 | 0.502 | 0.317 | 36.7% |
| **MiniLM-L6-v2** | Calibrated (OLS) | **0.781** [0.688, 0.879] | 0.989 [0.877, 1.095] | **0.524** [0.387, 0.641] | **0.502** [0.367, 0.621] | **0.400** [0.296, 0.493] | 64.7% |
| **MPNet-base-v2** | Scaled ($5 \times \cos$) | 1.366 | 1.650 | 0.521 | 0.479 | 0.344 | 39.3% |
| **MPNet-base-v2** | Calibrated (OLS) | 0.800 [0.708, 0.898] | **0.988** [0.874, 1.100] | 0.522 [0.364, 0.642] | 0.480 [0.322, 0.606] | 0.369 [0.253, 0.462] | **69.3%** |

### Paired Bootstrap Difference Results (n=1000, seed=42):
- **Calibrated Baselines vs `constant_mean`**:
  - `TF-IDF vs constant_mean`: $\Delta$ MAE = -0.0546 `[-0.1367, +0.0259]` (not significant), $\Delta$ RMSE = +0.0240 `[-0.0631, +0.1031]` (not significant), $\Delta$ QWK = +0.1644 `[+0.0062, +0.3524]` ($p < 0.05$).
  - `MiniLM vs constant_mean`: $\Delta$ MAE = **-0.1438** `[-0.2452, -0.0444]` ($p < 0.05$), $\Delta$ RMSE = **-0.1628** `[-0.2863, -0.0377]` ($p < 0.05$), $\Delta$ QWK = **+0.3967** `[+0.2958, +0.4933]` ($p < 0.05$).
  - `MPNet vs constant_mean`: $\Delta$ MAE = **-0.1254** `[-0.2136, -0.0362]` ($p < 0.05$), $\Delta$ RMSE = **-0.1647** `[-0.2713, -0.0509]` ($p < 0.05$), $\Delta$ QWK = **+0.3661** `[+0.2535, +0.4624]` ($p < 0.05$).
- **MiniLM (Calibrated) vs TF-IDF (Calibrated)**:
  - $\Delta$ MAE: -0.0892 `[-0.1880, +0.0127]`
  - $\Delta$ RMSE: **-0.1868** `[-0.3425, -0.0322]` ($p < 0.05$)
  - $\Delta$ Pearson: **+0.3428** `[+0.1161, +0.5738]` ($p < 0.05$)
  - $\Delta$ Spearman: **+0.2716** `[+0.1052, +0.4447]` ($p < 0.05$)
  - $\Delta$ QWK: **+0.2323** `[+0.0106, +0.4157]` ($p < 0.05$)

---

## 6. Gemini Client & Smoke Test Verification

- **API Model ID**: `gemini-3.8-flash`
- **Output Format**: Structured JSON schema `{"score": float, "reasoning": str}` with temperature `0.0`.
- **Disk Cache**: SHA-256 keyed cache stored in `results/llm_cache/`.
- **Smoke Test Verification Results**:
  1. **Run 1 (API calls)**: Latency ~5.8s per call (with rate-limit backoff), 3/3 valid JSON payloads parsed.
     - Sample 1 (No answer): Human `0.0` → Gemini `0` (`"The student did not provide an answer to the question."`)
     - Sample 2 (Accurate simulation definition): Human `5.0` → Gemini `5.0` (`"The student's response correctly and completely captures the role of a prototype..."`)
     - Sample 3 (Pointer constness modes): Human `5.0` → Gemini `5` (`"The student correctly identified all four combinations of pointer and data constness..."`)
  2. **Run 2 (Cached)**: Execution time < 0.2s. 3 cache hits logged (`cached=True` in `results/llm_log.csv`). 0 API quota consumed.
