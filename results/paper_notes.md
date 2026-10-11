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
- **Metrics Evaluated**: Mean Absolute Error (MAE) and Pearson Correlation ($r$).

---

## 5. Recorded Baseline Evaluation Results (Test Set, N = 150)

| Method | Scaling Type | MAE | RMSE | Pearson ($r$) | Spearman ($\rho$) | QWK ($\kappa$) | Within $\pm 1$ | 95% CI (MAE) | 95% CI (Pearson) |
|---|---|---|---|---|---|---|---|---|---|
| **TF-IDF** | Scaled ($5 \times \cos$) | 2.367 | 2.729 | 0.168 | 0.229 | 0.085 | 20.7% | — | — |
| **TF-IDF** | Calibrated (OLS) | 0.870 | 1.180 | 0.178 | 0.231 | 0.163 | 62.7% | [0.742, 1.003] | [0.004, 0.361] |
| **MiniLM-L6-v2** | Scaled ($5 \times \cos$) | 1.464 | 1.756 | 0.521 | 0.502 | 0.317 | 36.7% | — | — |
| **MiniLM-L6-v2** | Calibrated (OLS) | **0.781** | **0.989** | **0.524** | **0.502** | **0.400** | 64.7% | [0.685, 0.884] | [0.391, 0.634] |
| **MPNet-base-v2** | Scaled ($5 \times \cos$) | 1.366 | 1.650 | 0.521 | 0.479 | 0.344 | 39.3% | — | — |
| **MPNet-base-v2** | Calibrated (OLS) | 0.800 | 0.988 | 0.522 | 0.480 | 0.369 | **69.3%** | [0.710, 0.896] | [0.375, 0.636] |

### Key Observations for Discussion:
- **Embedding Superiority over Lexical (TF-IDF)**: MiniLM and MPNet achieve a correlation ($r \approx 0.524$) roughly 3x higher than TF-IDF ($r \approx 0.178$). The bootstrap 95% CIs for Pearson ($[0.391, 0.634]$ vs $[0.004, 0.361]$) confirm non-overlapping intervals, proving the gap is statistically meaningful.
- **MiniLM vs MPNet**: MiniLM performs comparably or marginally better than MPNet in correlation and QWK (0.400 vs 0.369) despite having 5x fewer parameters, demonstrating that lightweight sentence transformers provide strong sentence-level grading baselines.
- **Impact of Calibration**: Calibrating via dev-fitted linear regression decreases MAE dramatically (from 2.367 to 0.870 for TF-IDF; from 1.464 to 0.781 for MiniLM) and shifts predictions into the natural human distribution.

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
