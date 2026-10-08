# Methodology Notes for Paper

## Dataset & Sampling
- **Total Dataset Size:** 2,273 short-answer grading responses.
- **Sample Sizes:** 
  - DEV Set: 50 responses (isolated for prompt engineering and calibration).
  - TEST Set: 150 responses (strictly held-out for final reporting).
- **Sampling Strategy:** Responses were sampled randomly ensuring maximum coverage of unique questions (DEV spans 40 unique questions, TEST spans 69 unique questions) whilst strictly maintaining the aggregate `human_score` distribution of the full population.
- **Random Seed:** `random_state=42` was used for reproducible stratification and shuffling.

## Baselines
- **Algorithm:** TF-IDF vectorization with Cosine Similarity between the student's answer and the gold-standard reference answer.
- **Text Processing:** Lowercase normalization and English stop-word removal.
- **Data Isolation (Zero Leakage):** The TF-IDF vocabulary was fitted *exclusively* on the DEV set (both reference and student answers). The vectorizer was then applied to transform the unseen TEST set.
- **Calibration Methods:** 
  - (a) **Scaled:** Direct scaling of the cosine similarity (`score = 5 * cosine`).
  - (b) **Calibrated:** A Linear Regression model was fitted on the DEV set to map cosine similarity to the `human_score`, then applied to the TEST set (predictions clipped bounded between 0 and 5).
