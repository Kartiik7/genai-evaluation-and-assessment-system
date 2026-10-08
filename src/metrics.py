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
