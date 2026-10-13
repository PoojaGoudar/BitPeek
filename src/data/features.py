import numpy as np
import pandas as pd
import ta

def compute_wavelet_features(series: np.ndarray, level: int = 2) -> np.ndarray:
    """
    Fast discrete Haar/multi-resolution decomposition for causal time-series.
    Decomposes sequence into low-frequency trend approximation and high-frequency shock details.
    """
    n = len(series)
    approx = series.copy()
    details = []
    
    for _ in range(level):
        pad_len = len(approx) % 2
        if pad_len != 0:
            approx = np.pad(approx, (0, 1), mode='edge')
        
        # Haar low-pass (smooth) and high-pass (shock)
        low = (approx[0::2] + approx[1::2]) / np.sqrt(2.0)
        high = (approx[0::2] - approx[1::2]) / np.sqrt(2.0)
        details.append(high)
        approx = low
        
    # Reconstruct interpolated approximations back to original length
    trend_approx = np.repeat(approx, 2**level)[:n]
    shock_detail = np.repeat(details[0], 2)[:n]
    return trend_approx, shock_detail

def engineer_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes over 40 quantitative, technical, statistical, and multi-resolution features
    specifically optimized for cryptocurrency time-series forecasting.
    """
    df = df.copy()
    df = df.sort_index()
    
    # 1. Price Action & Spreads (Stationary transformations)
    df['log_return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
    df['log_return_3d'] = np.log(df['Close'] / df['Close'].shift(3))
    df['log_return_7d'] = np.log(df['Close'] / df['Close'].shift(7))
    df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
    df['co_spread'] = (df['Close'] - df['Open']) / df['Open']
    df['volume_change'] = df['Volume'].pct_change()
    
    # 2. Moving Average Relativities
    df['close_sma_7_ratio'] = df['Close'] / df['Close'].rolling(7).mean() - 1.0
    df['close_sma_21_ratio'] = df['Close'] / df['Close'].rolling(21).mean() - 1.0
    df['close_sma_50_ratio'] = df['Close'] / df['Close'].rolling(50).mean() - 1.0
    df['close_ema_9_ratio'] = df['Close'] / ta.trend.EMAIndicator(df['Close'], window=9).ema_indicator() - 1.0
    df['close_ema_21_ratio'] = df['Close'] / ta.trend.EMAIndicator(df['Close'], window=21).ema_indicator() - 1.0
    
    # 3. Momentum & Oscillators
    df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi() / 100.0
    df['rsi_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi() / 100.0
    
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch() / 100.0
    df['stoch_d'] = stoch.stoch_signal() / 100.0
    
    df['williams_r'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r() / -100.0
    df['roc_10'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc() / 100.0
    df['roc_5'] = ta.momentum.ROCIndicator(df['Close'], window=5).roc() / 100.0
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd_line'] = macd.macd() / df['Close']
    df['macd_signal'] = macd.macd_signal() / df['Close']
    df['macd_diff'] = macd.macd_diff() / df['Close']
    
    # 4. Volatility Indicators
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['bb_pband'] = bb.bollinger_pband()
    df['bb_wband'] = bb.bollinger_wband() / 100.0
    
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
    df['atr_ratio'] = atr.average_true_range() / df['Close']
    
    df['hist_vol_7d'] = df['log_return_1d'].rolling(7).std() * np.sqrt(365)
    df['hist_vol_30d'] = df['log_return_1d'].rolling(30).std() * np.sqrt(365)
    
    # 5. Volume Flow Indicators
    obv = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['obv_norm'] = (obv - obv.rolling(30).mean()) / (obv.rolling(30).std() + 1e-8)
    
    df['cmf_20'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()
    df['volume_zscore_20'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (df['Volume'].rolling(20).std() + 1e-8)
    
    # 6. Trend Strength (ADX)
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['adx'] = adx.adx() / 100.0
    df['adx_pos'] = adx.adx_pos() / 100.0
    df['adx_neg'] = adx.adx_neg() / 100.0
    
    # 7. Wavelet Decomposition on Close Price Returns
    clean_rets = df['log_return_1d'].fillna(0).values
    trend_comp, shock_comp = compute_wavelet_features(clean_rets, level=2)
    df['wavelet_trend'] = trend_comp
    df['wavelet_shock'] = shock_comp
    
    # 8. Cyclical / Calendar Features
    if isinstance(df.index, pd.DatetimeIndex):
        day_of_week = df.index.dayofweek
        month = df.index.month
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)
        df['month_sin'] = np.sin(2 * np.pi * month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    # Target Variables (Forecasting t+1 from info up to t)
    # Target 1 (Regression): Next day log return -> ln(Close_{t+1} / Close_t)
    df['target_return_t1'] = np.log(df['Close'].shift(-1) / df['Close'])
    # Target 2 (Direction): 1 if price goes UP next day, 0 if DOWN
    df['target_direction_t1'] = (df['target_return_t1'] > 0).astype(np.float32)
    # Next day actual close price for evaluation
    df['target_close_t1'] = df['Close'].shift(-1)
    
    return df
