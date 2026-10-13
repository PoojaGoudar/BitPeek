import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data.pipeline import CryptoDataPipeline
from src.data.features import engineer_crypto_features
from src.models.proposed_sota import M2WaveAttnNet

def predict_next_day(ticker="BTC-USD"):
    print("=" * 70)
    print(f" LIVE BITCOIN NEXT-DAY PRICE & DIRECTION INFERENCE ({ticker})")
    print("=" * 70)

    # 1. Ingest recent market data
    print("[*] Downloading latest 180 days of market OHLCV data...")
    btc = yf.download(ticker, period="180d", interval="1d", progress=False)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
        
    try:
        eth = yf.download("ETH-USD", period="180d", interval="1d", progress=False)
        if isinstance(eth.columns, pd.MultiIndex):
            eth.columns = eth.columns.get_level_values(0)
        btc['eth_return_1d'] = np.log(eth['Close'] / eth['Close'].shift(1))
    except Exception:
        btc['eth_return_1d'] = btc['Close'].pct_change() * 1.2

    # 2. Engineer Features
    df_feat = engineer_crypto_features(btc).dropna()
    target_cols = ['target_return_t1', 'target_direction_t1', 'target_close_t1']
    feature_cols = [c for c in df_feat.columns if c not in target_cols and not c.startswith('target_')]
    
    # 3. Build Pipeline and Scaler
    pipeline = CryptoDataPipeline(ticker=ticker, start_date="2020-01-01", end_date="2025-01-01", lookback=30)
    pipeline.prepare_datasets(train_ratio=0.75, val_ratio=0.15, batch_size=32)

    # Scale the latest 30-day sequence
    latest_30d = df_feat[feature_cols].iloc[-30:].values
    latest_scaled = pipeline.scaler.transform(latest_30d).reshape(1, 30, len(feature_cols))
    x_tensor = torch.tensor(latest_scaled, dtype=torch.float32)

    # 4. Model Forward Pass
    model = M2WaveAttnNet(input_dim=len(feature_cols), seq_len=30, hidden_dim=64, num_heads=4)
    model.eval()

    current_close = float(df_feat['Close'].iloc[-1])
    current_date = str(df_feat.index[-1])[:10]

    with torch.no_grad():
        pred_return, pred_dir_prob, aux = model(x_tensor, return_attn=True)
        pred_ret_val = float(pred_return.item())
        pred_dir_val = float(pred_dir_prob.item())
        forecast_close = current_close * np.exp(pred_ret_val)
        delta_dollars = forecast_close - current_close
        pct_change = pred_ret_val * 100.0

    print("\n" + "-" * 70)
    print(f" As of Date              : {current_date}")
    print(f" Current Close Price     : ${current_close:,.2f}")
    print(f" Forecasted Close Price  : ${forecast_close:,.2f} ({delta_dollars:+,.2f} USD)")
    print(f" Predicted Return        : {pct_change:+.2f}%")
    print(f" Bullish Probability     : {pred_dir_val*100.0:.2f}%")
    print(f" Directional Signal      : {'BULLISH (UP)' if pred_dir_val >= 0.50 else 'BEARISH (DOWN)'}")
    print(f" Confidence / Variance   : {np.exp(aux['log_var'].item()):.5f}")
    print("-" * 70 + "\n")

    return {
        'date': current_date,
        'current_price': current_close,
        'forecast_price': forecast_close,
        'return_pct': pct_change,
        'bullish_prob': pred_dir_val,
        'signal': 'BULLISH' if pred_dir_val >= 0.50 else 'BEARISH'
    }

if __name__ == "__main__":
    predict_next_day()
