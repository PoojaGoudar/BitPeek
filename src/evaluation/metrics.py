import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def compute_all_metrics(true_close, pred_close, true_returns, pred_returns, true_dir, pred_dir_prob):
    """
    Computes rigorous academic benchmark metrics for both price level, percentage returns, and directional accuracy.
    """
    # 1. Price-Level Regression Metrics
    rmse = np.sqrt(mean_squared_error(true_close, pred_close))
    mae = mean_absolute_error(true_close, pred_close)
    mape = np.mean(np.abs((true_close - pred_close) / (true_close + 1e-8))) * 100.0
    r2 = r2_score(true_close, pred_close)

    # 2. Return-Level Metrics
    rmse_ret = np.sqrt(mean_squared_error(true_returns, pred_returns))
    mae_ret = mean_absolute_error(true_returns, pred_returns)

    # 3. Direction Classification Metrics
    pred_binary = (pred_dir_prob >= 0.5).astype(int)
    true_binary = true_dir.astype(int)
    
    da = accuracy_score(true_binary, pred_binary) * 100.0
    prec = precision_score(true_binary, pred_binary, zero_division=0) * 100.0
    rec = recall_score(true_binary, pred_binary, zero_division=0) * 100.0
    f1 = f1_score(true_binary, pred_binary, zero_division=0) * 100.0

    # 4. Sign Match Rate (Consistency between return forecast and direction forecast)
    pred_ret_sign = (pred_returns > 0).astype(int)
    sign_da = accuracy_score(true_binary, pred_ret_sign) * 100.0

    return {
        'RMSE ($)': float(rmse),
        'MAE ($)': float(mae),
        'MAPE (%)': float(mape),
        'R2 Score': float(r2),
        'Return RMSE': float(rmse_ret),
        'Return MAE': float(mae_ret),
        'Directional Accuracy (%)': float(da),
        'Sign DA (%)': float(sign_da),
        'Precision (%)': float(prec),
        'Recall (%)': float(rec),
        'F1-Score (%)': float(f1)
    }
