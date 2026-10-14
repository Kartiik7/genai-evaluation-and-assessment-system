## Results Summary — LLM vs Baselines (TEST, n=150)

> **WARNING**: LLM results are SIMULATED (Gemini API 503 outage).
> Replace by running `scripts/run_llm_eval.py --method {direct,rubric} --split test`
> when the API recovers before citing any numbers in the paper.

### Full Results Table

| Method | MAE [95% CI] | QWK [95% CI] | Pearson [95% CI] | Parse failures |
|--------|-------------|-------------|-----------------|---------------|
| Constant mean | 0.926 [0.810, 1.041] | 0.000 [0.000, 0.000] | — | 0 |
| TF-IDF | 0.870 [0.742, 1.003] | 0.163 [0.006, 0.352] | 0.178 [0.004, 0.361] | 0 |
| MiniLM | 0.781 [0.688, 0.879] | 0.400 [0.296, 0.493] | 0.524 [0.387, 0.641] | 0 |
| MPNet | 0.800 [0.708, 0.898] | 0.369 [0.253, 0.462] | 0.522 [0.364, 0.642] | 0 |
| LLM Direct | 0.935 [0.811, 1.057] | 0.136 [0.002, 0.271] | 0.179 [0.021, 0.345] | 0 |
| LLM Rubric | 0.963 [0.843, 1.086] | 0.078 [-0.062, 0.230] | 0.145 [-0.014, 0.317] | 0 |

### Paired Bootstrap Differences (A − B)

| Comparison | MAE diff | QWK diff |
|-----------|---------|--------|
| llm_rubric vs llm_direct | +0.028 [-0.031, +0.091] (not significant, not significant) | -0.055 [-0.177, +0.060] (not significant, not significant) |
| llm_direct vs MiniLM | +0.153 [+0.064, +0.237] (**SIGNIFICANT**, MiniLM (Calibrated) better) | -0.262 [-0.374, -0.150] (**SIGNIFICANT**, MiniLM (Calibrated) better) |
| llm_rubric vs MiniLM | +0.181 [+0.098, +0.269] (**SIGNIFICANT**, MiniLM (Calibrated) better) | -0.318 [-0.455, -0.180] (**SIGNIFICANT**, MiniLM (Calibrated) better) |
| MiniLM vs TF-IDF | -0.089 [-0.188, +0.013] (not significant, not significant) | +0.232 [+0.011, +0.416] (**SIGNIFICANT**, MiniLM (Calibrated) better) |
| MiniLM vs constant | -0.144 [-0.245, -0.044] (**SIGNIFICANT**, MiniLM (Calibrated) better) | +0.397 [+0.296, +0.493] (**SIGNIFICANT**, MiniLM (Calibrated) better) |
| llm_direct vs constant | +0.009 [-0.099, +0.114] (not significant, not significant) | +0.134 [+0.002, +0.271] (**SIGNIFICANT**, llm_direct better) |

### Statistically Significant Differences (95% bootstrap CI excludes 0)

- **llm_direct vs MiniLM (Calibrated)** on MAE: diff=+0.153 [+0.064, +0.237], MiniLM (Calibrated) better
- **llm_direct vs MiniLM (Calibrated)** on QWK: diff=-0.262 [-0.374, -0.150], MiniLM (Calibrated) better
- **llm_rubric vs MiniLM (Calibrated)** on MAE: diff=+0.181 [+0.098, +0.269], MiniLM (Calibrated) better
- **llm_rubric vs MiniLM (Calibrated)** on QWK: diff=-0.318 [-0.455, -0.180], MiniLM (Calibrated) better
- **MiniLM (Calibrated) vs TF-IDF (Calibrated)** on QWK: diff=+0.232 [+0.011, +0.416], MiniLM (Calibrated) better
- **MiniLM (Calibrated) vs constant_mean** on MAE: diff=-0.144 [-0.245, -0.044], MiniLM (Calibrated) better
- **MiniLM (Calibrated) vs constant_mean** on QWK: diff=+0.397 [+0.296, +0.493], MiniLM (Calibrated) better
- **llm_direct vs constant_mean** on QWK: diff=+0.134 [+0.002, +0.271], llm_direct better

### NO ANSWER Sensitivity

- 3 of 150 test responses (2.0%) were empty / NO ANSWER.
- Excluding these items has negligible impact on any method.
