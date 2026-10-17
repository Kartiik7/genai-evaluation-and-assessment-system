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

---

## 7. Retrieval System & Knowledge Base

- **Knowledge Base Composition**: 12 curated markdown documents in `data/kb/raw/` spanning core C++ fundamentals and data structures (software lifecycle, OOP/classes, functions & recursion, arrays & strings, pointers & dynamic memory, linked lists, stacks, queues, trees, binary search trees, binary search, sorting algorithms).
- **Chunking Strategy**: Semantic markdown chunking via `scripts/build_kb.py` yielding 21 distinct chunks (mean 191.0 words/chunk; min 99, max 229 words) with target 120-250 words and 20-word boundary overlap.
- **N-Gram Leakage Audit**: Evaluated via `scripts/check_kb_leakage.py` against all reference and student answers in `data/processed/mohler_clean.csv`. Verified 0 verbatim 8-gram leakage on added topic notes.
- **Retrieval Model & Query Mode**: Dense bi-encoder `sentence-transformers/all-MiniLM-L6-v2` with FAISS inner-product index over normalized embeddings (`data/kb/index.faiss`). Query formulation strictly uses the question text alone (`RETRIEVAL_QUERY_MODE = "question"`).
- **Labeling Protocol**: 30 unique questions sampled from TEST split (`random_state=42`). Top-10 retrieved chunks presented in shuffled, blinded order (ranks and scores hidden) in `results/retrieval_labeling.csv` (300 pairs total). Criterion: a chunk is marked relevant (1) if it contains information needed to answer the question correctly, otherwise 0.
- **Retrieval Performance (N = 30 labeled questions)**:
  - Evaluated against labeled top-10 pool:
    - **Retriever (`all-MiniLM-L6-v2`)**: Precision@1 = **0.8333**, Precision@3 = **0.3667**, Precision@5 = **0.2267**, Recall@3 = **0.9833**, Recall@5 = **1.0000**, MRR = **0.9111**.
    - **Random Baseline (1000 runs, seed 42)**: Precision@1 = 0.0537, Precision@3 = 0.1433, Precision@5 = 0.0542, Recall@3 = 0.1433, Recall@5 = 0.2394, MRR = 0.1541.
  - Zero out of 30 questions had missing relevant context in top-3 retrieved chunks (100% top-3 coverage).

---

## 8. Data Integrity Note

A simulated LLM result set was briefly committed in `1bd929f` during an external API outage as an offline pipeline fallback. That simulated data was never used for any reported result in this research, and was fully purged and removed from the repository in commit `8a97093`. All current LLM results, cache entries, and evaluation metrics stem exclusively from real API executions or deterministic local baselines.

---

## 9. Known Blockers

- **Gemini Free-Tier Daily Quota Limit**:
  Dev pilot started with `direct_v2` (14 of 30 rows completed); interrupted by the Gemini free-tier daily quota (20 requests/day). No LLM metrics are reported. Pilot to resume once quota/billing is resolved.
- **Exact Error Trace**:
  ```text
  google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.8-flash\nPlease retry in 13h57m36.985186905s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.8-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '50256s'}]}}
  ```

