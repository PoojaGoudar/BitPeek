"""
Statistical Significance Testing Module for Financial Time-Series Forecasting
=============================================================================
Provides formal econometric and statistical tests to evaluate whether 
forecasting improvements of the proposed architecture are statistically significant:

1. Diebold-Mariano (DM) Test with Harvey-Leybourne-Newbold (HLN) small-sample 
   correction and Newey-West HAC autocorrelation standard errors.
2. McNemar's Test for Directional Accuracy (paired chi-square with continuity correction).
3. Paired Wilcoxon Signed-Rank Test for non-parametric loss comparison.
"""
import numpy as np
from scipy import stats

def diebold_mariano_test(y_true: np.ndarray, 
                         y_pred_benchmark: np.ndarray, 
                         y_pred_proposed: np.ndarray, 
                         h: int = 1, 
                         criterion: str = "MSE") -> dict:
    """
    Computes the Diebold-Mariano test statistic comparing two forecast error series.
    
    H0: Both models have equal forecast accuracy.
    H1: Proposed model has superior forecast accuracy (one-sided) or different accuracy (two-sided).
    
    Parameters:
        y_true: Ground truth target values
        y_pred_benchmark: Forecasts from baseline model
        y_pred_proposed: Forecasts from proposed model (e.g. M2-WaveAttnNet)
        h: Forecast horizon (default 1)
        criterion: "MSE" (squared error) or "MAE" (absolute error)
        
    Returns:
        dict containing DM statistic, p-value (two-sided and one-sided), and significance flag.
    """
    e1 = y_true - y_pred_benchmark
    e2 = y_true - y_pred_proposed
    
    if criterion.upper() == "MSE":
        d = (e1 ** 2) - (e2 ** 2)
    elif criterion.upper() == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
        
    N = len(d)
    mean_d = np.mean(d)
    
    # Auto-covariance estimation (HAC / Newey-West weighting)
    def autocovariance(x, k):
        n = len(x)
        x_centered = x - np.mean(x)
        if k == 0:
            return np.var(x)
        return np.sum(x_centered[k:] * x_centered[:-k]) / n

    # Spectral density at frequency 0
    gamma_0 = autocovariance(d, 0)
    var_d = gamma_0
    for lag in range(1, h):
        weight = 1.0 - (lag / h)
        var_d += 2.0 * weight * autocovariance(d, lag)
        
    var_d = max(var_d, 1e-12)
    se_d = np.sqrt(var_d / N)
    
    # Standard DM statistic
    dm_stat = mean_d / se_d
    
    # Harvey, Leybourne, and Newbold (HLN, 1997) small-sample modification
    hln_factor = np.sqrt((N + 1 - 2 * h + (h * (h - 1)) / N) / N)
    dm_stat_hln = dm_stat * hln_factor
    
    # Student's t distribution with (N - 1) degrees of freedom
    p_val_two_sided = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_hln), df=N - 1))
    p_val_one_sided = 1.0 - stats.t.cdf(dm_stat_hln, df=N - 1)
    
    return {
        'dm_stat': float(dm_stat_hln),
        'p_value_two_sided': float(p_val_two_sided),
        'p_value_one_sided': float(p_val_one_sided),
        'significant_5pct': bool(p_val_one_sided < 0.05 and dm_stat_hln > 0),
        'significant_1pct': bool(p_val_one_sided < 0.01 and dm_stat_hln > 0)
    }

def mcnemar_test(y_true_dir: np.ndarray, 
                 y_pred_benchmark: np.ndarray, 
                 y_pred_proposed: np.ndarray) -> dict:
    """
    McNemar's test for paired binary directional classification.
    
    Evaluates whether the difference in directional accuracy is statistically significant.
    Contingency table:
      - n00: both incorrect
      - n01: benchmark correct, proposed incorrect
      - n10: benchmark incorrect, proposed correct
      - n11: both correct
    """
    b_correct = (y_pred_benchmark == y_true_dir)
    p_correct = (y_pred_proposed == y_true_dir)
    
    n01 = np.sum(b_correct & (~p_correct)) # Benchmark won
    n10 = np.sum((~b_correct) & p_correct) # Proposed won
    
    discordant = n01 + n10
    if discordant == 0:
        return {'chi2_stat': 0.0, 'p_value': 1.0, 'n10': int(n10), 'n01': int(n01)}
        
    # Edwards continuity correction
    chi2_stat = ((np.abs(n10 - n01) - 1.0) ** 2) / float(discordant)
    p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
    
    return {
        'chi2_stat': float(chi2_stat),
        'p_value': float(p_value),
        'n_proposed_correct_only': int(n10),
        'n_baseline_correct_only': int(n01),
        'significant_5pct': bool(p_value < 0.05 and n10 > n01),
        'significant_1pct': bool(p_value < 0.01 and n10 > n01)
    }

def wilcoxon_signed_rank_test(y_true: np.ndarray, 
                              y_pred_benchmark: np.ndarray, 
                              y_pred_proposed: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test on absolute forecast errors."""
    e1 = np.abs(y_true - y_pred_benchmark)
    e2 = np.abs(y_true - y_pred_proposed)
    diff = e1 - e2
    diff = diff[diff != 0] # non-zero differences
    
    if len(diff) < 10:
        return {'stat': 0.0, 'p_value': 1.0}
        
    stat, p_val = stats.wilcoxon(diff, alternative='greater')
    return {
        'stat': float(stat),
        'p_value': float(p_val),
        'significant_5pct': bool(p_val < 0.05)
    }

