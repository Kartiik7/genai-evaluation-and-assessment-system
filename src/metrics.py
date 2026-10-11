import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
import warnings

def evaluate(human, pred):
    """
    Evaluates short-answer grading predictions against human scores.
    Returns a dictionary of metrics.
    """
    human = np.array(human)
    pred = np.array(pred)
    
    mae = mean_absolute_error(human, pred)
    rmse = np.sqrt(mean_squared_error(human, pred))
    
    # Handle Pearson and Spearman where predictions might be constant
    if np.std(pred) == 0 or np.std(human) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pearson = pearsonr(human, pred)[0]
            spearman = spearmanr(human, pred)[0]
    
    # QWK
    # Clip predictions to [0, 5] and round both to integers
    pred_int = np.clip(np.round(pred), 0, 5).astype(int)
    human_int = np.round(human).astype(int)
    
    # Check if we have more than 1 class in the rounded data, else QWK can be ill-defined
    qwk = cohen_kappa_score(human_int, pred_int, weights='quadratic', labels=[0, 1, 2, 3, 4, 5])
    
    within_1 = np.mean(np.abs(human - pred) <= 1.0)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "qwk": qwk,
        "within_1": within_1
    }

def results_table(dict_of_method_to_metrics):
    """
    Takes a dictionary mapping method names to metrics dictionaries and returns a tidy DataFrame.
    """
    df = pd.DataFrame.from_dict(dict_of_method_to_metrics, orient='index')
    df.index.name = 'Method'
    return df

def append_results(method_name, metrics_dict, file_path="results/main_results.csv"):
    """
    Appends a method's results to a CSV file. Creates the file and headers if it doesn't exist.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df_new = pd.DataFrame([metrics_dict], index=[method_name])
    df_new.index.name = 'Method'
    
    header = not os.path.exists(file_path)
    df_new.to_csv(file_path, mode='a', header=header)


def _compute_single_metric(h, p, metric):
    if callable(metric):
        return metric(h, p)
    metric_name = str(metric).lower()
    if metric_name == 'mae':
        return mean_absolute_error(h, p)
    elif metric_name == 'rmse':
        return np.sqrt(mean_squared_error(h, p))
    elif metric_name == 'pearson':
        if np.std(p) == 0 or np.std(h) == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pearsonr(h, p)[0]
    elif metric_name == 'spearman':
        if np.std(p) == 0 or np.std(h) == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return spearmanr(h, p)[0]
    elif metric_name == 'qwk':
        p_int = np.clip(np.round(p), 0, 5).astype(int)
        h_int = np.round(h).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cohen_kappa_score(h_int, p_int, weights='quadratic', labels=[0, 1, 2, 3, 4, 5])
    elif metric_name == 'within_1':
        return np.mean(np.abs(h - p) <= 1.0)
    else:
        raise ValueError(f"Unknown metric: {metric}")


class BootstrapDiffResult(tuple):
    """
    Tuple containing (mean_diff, (ci_lo, ci_hi)).
    Supports unpacking as (diff, ci), access via properties (.mean_diff, .ci_lo, .ci_hi),
    or dict-style indexing.
    """
    def __new__(cls, mean_diff, ci):
        return super().__new__(cls, (mean_diff, ci))

    @property
    def mean_diff(self):
        return self[0]

    @property
    def ci(self):
        return self[1]

    @property
    def ci_lo(self):
        return self[1][0]

    @property
    def ci_hi(self):
        return self[1][1]

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in ('mean_diff', 'diff'):
                return self[0]
            elif item == 'ci':
                return self[1]
            elif item in ('ci_lo', 'ci_lower', 'ci_min'):
                return self[1][0]
            elif item in ('ci_hi', 'ci_upper', 'ci_max'):
                return self[1][1]
            raise KeyError(item)
        return super().__getitem__(item)

    def __repr__(self):
        return f"BootstrapDiffResult(mean_diff={self[0]:.4f}, ci=({self[1][0]:.4f}, {self[1][1]:.4f}))"


def paired_bootstrap_diff(human, pred_a, pred_b, metric, n=1000, seed=42):
    """
    Computes the paired bootstrap difference (metric_a - metric_b) and its 95% CI
    using the SAME resampled indices for both methods.

    Returns: BootstrapDiffResult with (mean_diff, (ci_lo, ci_hi)).
    """
    human = np.array(human)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    N = len(human)

    rng = np.random.default_rng(seed)
    diffs = []

    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        h = human[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]

        val_a = _compute_single_metric(h, pa, metric)
        val_b = _compute_single_metric(h, pb, metric)

        if np.isnan(val_a) or np.isnan(val_b):
            diffs.append(np.nan)
        else:
            diffs.append(val_a - val_b)

    diffs = np.array(diffs)
    valid_diffs = diffs[~np.isnan(diffs)]

    if len(valid_diffs) == 0:
        return BootstrapDiffResult(np.nan, (np.nan, np.nan))

    mean_diff = float(np.mean(valid_diffs))
    ci_lo = float(np.percentile(valid_diffs, 2.5))
    ci_hi = float(np.percentile(valid_diffs, 97.5))

    return BootstrapDiffResult(mean_diff, (ci_lo, ci_hi))


def bootstrap_ci(human, pred, metric, n=1000, seed=42):
    """
    Computes a 95% bootstrap confidence interval for a single metric.
    Returns: (ci_lo, ci_hi).
    """
    human = np.array(human)
    pred = np.array(pred)
    N = len(human)

    rng = np.random.default_rng(seed)
    vals = []

    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        val = _compute_single_metric(human[idx], pred[idx], metric)
        vals.append(val)

    vals = np.array(vals)
    valid_vals = vals[~np.isnan(vals)]

    if len(valid_vals) == 0:
        return (np.nan, np.nan)

    return (float(np.percentile(valid_vals, 2.5)), float(np.percentile(valid_vals, 97.5)))

