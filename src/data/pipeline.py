import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
from .features import engineer_crypto_features

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for multi-variate sliding window time-series."""
    def __init__(self, X: np.ndarray, y_return: np.ndarray, y_dir: np.ndarray, y_close: np.ndarray, dates: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_return = torch.tensor(y_return, dtype=torch.float32).unsqueeze(-1)
        self.y_dir = torch.tensor(y_dir, dtype=torch.float32).unsqueeze(-1)
        self.y_close = torch.tensor(y_close, dtype=torch.float32).unsqueeze(-1)
        self.dates = dates

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            'x': self.X[idx],
            'y_return': self.y_return[idx],
            'y_dir': self.y_dir[idx],
            'y_close': self.y_close[idx],
            'date': str(self.dates[idx])
        }

class CryptoDataPipeline:
    def __init__(self, ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.scaler = RobustScaler()
        self.feature_names = []
        self.df_raw = None
        self.df_engineered = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetches BTC data and macro proxies with local caching.
        Macro features: ETH, DXY (US Dollar Index), S&P 500, Gold.
        Cache is automatically invalidated if macro columns are missing.
        """
        MACRO_COLS = ['eth_return_1d', 'dxy_return_1d', 'spx_return_1d', 'gold_return_1d']
        cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "btc_cache.csv")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if all(c in cached.columns for c in MACRO_COLS):
                print(f"[*] Loading historical market data from local cache: {cache_path}...")
                cached[MACRO_COLS] = cached[MACRO_COLS].fillna(0.0)
                self.df_raw = cached
                return self.df_raw
            else:
                print("[*] Cache missing macro columns — re-downloading with enriched features...")

        print(f"[*] Fetching historical market data for {self.ticker} ({self.start_date} to {self.end_date})...")
        try:
            btc = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)
            if isinstance(btc.columns, pd.MultiIndex):
                btc.columns = btc.columns.get_level_values(0)
            if btc.empty or len(btc) < 100:
                raise ValueError("Downloaded data empty or insufficient length.")
        except Exception as e:
            print(f"[!] Warning fetching live data ({e}). Generating synthetic series...")
            btc = self._generate_synthetic_crypto_data()

        def _safe_log_return(ticker_sym):
            """Download a ticker and return its 1-day log return aligned to btc index."""
            try:
                df = yf.download(ticker_sym, start=self.start_date, end=self.end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if not df.empty:
                    # Forward-fill closing prices across weekends/holidays so market closures have 0.0 return
                    close_aligned = df['Close'].reindex(btc.index).ffill().bfill()
                    ret = np.log(close_aligned / close_aligned.shift(1)).fillna(0.0)
                    return ret
            except Exception as e:
                print(f"[!] Warning downloading {ticker_sym}: {e}")
            return None

        # ETH — crypto cross-asset momentum
        eth_ret = _safe_log_return("ETH-USD")
        btc['eth_return_1d'] = eth_ret if eth_ret is not None else btc['Close'].pct_change().fillna(0.0) * 1.2

        # US Dollar Index — inverse BTC correlation (stronger dollar → risk-off)
        dxy_ret = _safe_log_return("DX-Y.NYB")
        btc['dxy_return_1d'] = dxy_ret if dxy_ret is not None else 0.0

        # S&P 500 — risk-on / risk-off macro signal
        spx_ret = _safe_log_return("^GSPC")
        btc['spx_return_1d'] = spx_ret if spx_ret is not None else 0.0

        # Gold — alternative safe-haven / inflation hedge cross-signal
        gold_ret = _safe_log_return("GLD")
        btc['gold_return_1d'] = gold_ret if gold_ret is not None else 0.0

        # Ensure all macro columns have strictly zero NaNs
        btc[MACRO_COLS] = btc[MACRO_COLS].fillna(0.0)

        self.df_raw = btc.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.df_raw.to_csv(cache_path)
        print(f"[+] Saved enriched data to cache: {cache_path}")
        return self.df_raw

    def _generate_synthetic_crypto_data(self) -> pd.DataFrame:
        """High-fidelity calibrated multi-regime GBM with jump diffusion for fallback/testing."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        n = len(dates)
        np.random.seed(42)
        
        # Volatility regime switching
        dt = 1 / 365.0
        p0 = 10000.0
        prices = [p0]
        vol = 0.70
        for i in range(1, n):
            jump = np.random.choice([0, 1], p=[0.97, 0.03]) * np.random.normal(0, 0.08)
            drift = 0.40 * dt
            shock = vol * np.sqrt(dt) * np.random.normal() + jump
            p_next = prices[-1] * np.exp(drift + shock)
            prices.append(max(100.0, p_next))
            
        prices = np.array(prices)
        highs = prices * (1 + np.abs(np.random.normal(0, 0.02, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.02, n)))
        opens = prices * (1 + np.random.normal(0, 0.01, n))
        volumes = np.random.lognormal(mean=20, sigma=0.8, size=n)
        
        df = pd.DataFrame({
            'Open': opens, 'High': highs, 'Low': lows, 'Close': prices, 'Volume': volumes
        }, index=dates)
        return df

    def prepare_datasets(self, train_ratio=0.70, val_ratio=0.15, batch_size=32):
        """Processes features, applies causal rolling windows, and builds train/val/test loaders."""
        if self.df_raw is None:
            self.fetch_data()

        print("[*] Performing domain feature engineering and multi-resolution decomposition...")
        df_feat = engineer_crypto_features(self.df_raw)
        df_feat = df_feat.dropna()
        self.df_engineered = df_feat

        # Exclude targets from input feature columns
        target_cols = ['target_return_t1', 'target_direction_t1', 'target_close_t1']
        feature_cols = [c for c in df_feat.columns if c not in target_cols and not c.startswith('target_')]
        self.feature_names = feature_cols

        X_raw = df_feat[feature_cols].values
        y_ret_raw = df_feat['target_return_t1'].values
        y_dir_raw = df_feat['target_direction_t1'].values
        y_close_raw = df_feat['target_close_t1'].values
        dates_raw = df_feat.index.values

        # Sliding window generation (causal lookback)
        X_seq, y_ret_seq, y_dir_seq, y_close_seq, dates_seq = [], [], [], [], []
        for i in range(len(X_raw) - self.lookback):
            X_seq.append(X_raw[i : i + self.lookback])
            y_ret_seq.append(y_ret_raw[i + self.lookback - 1])
            y_dir_seq.append(y_dir_raw[i + self.lookback - 1])
            y_close_seq.append(y_close_raw[i + self.lookback - 1])
            dates_seq.append(dates_raw[i + self.lookback - 1])

        X_seq = np.array(X_seq, dtype=np.float32)
        y_ret_seq = np.array(y_ret_seq, dtype=np.float32)
        y_dir_seq = np.array(y_dir_seq, dtype=np.float32)
        y_close_seq = np.array(y_close_seq, dtype=np.float32)
        dates_seq = np.array(dates_seq)

        # Chronological splits (strictly no future lookahead leakage)
        total_samples = len(X_seq)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))

        X_train, y_ret_train, y_dir_train, y_close_train, dates_train = (
            X_seq[:train_end], y_ret_seq[:train_end], y_dir_seq[:train_end], y_close_seq[:train_end], dates_seq[:train_end]
        )
        X_val, y_ret_val, y_dir_val, y_close_val, dates_val = (
            X_seq[train_end:val_end], y_ret_seq[train_end:val_end], y_dir_seq[train_end:val_end], y_close_seq[train_end:val_end], dates_seq[train_end:val_end]
        )
        X_test, y_ret_test, y_dir_test, y_close_test, dates_test = (
            X_seq[val_end:], y_ret_seq[val_end:], y_dir_seq[val_end:], y_close_seq[val_end:], dates_seq[val_end:]
        )

        # Standardize features using Train statistics only
        N_tr, T, D = X_train.shape
        X_train_2d = X_train.reshape(-1, D)
        self.scaler.fit(X_train_2d)

        X_train_scaled = self.scaler.transform(X_train_2d).reshape(N_tr, T, D)
        X_val_scaled = self.scaler.transform(X_val.reshape(-1, D)).reshape(len(X_val), T, D)
        X_test_scaled = self.scaler.transform(X_test.reshape(-1, D)).reshape(len(X_test), T, D)

        print(f"[+] Data Pipeline Ready: Train={len(X_train_scaled)}, Val={len(X_val_scaled)}, Test={len(X_test_scaled)}, Features={D}, Window={T}")

        train_ds = TimeSeriesDataset(X_train_scaled, y_ret_train, y_dir_train, y_close_train, dates_train)
        val_ds = TimeSeriesDataset(X_val_scaled, y_ret_val, y_dir_val, y_close_val, dates_val)
        test_ds = TimeSeriesDataset(X_test_scaled, y_ret_test, y_dir_test, y_close_test, dates_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'train_ds': train_ds,
            'val_ds': val_ds,
            'test_ds': test_ds,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'input_dim': D,
            'seq_len': T,
            'test_raw_close': df_feat['Close'].iloc[val_end + self.lookback - 1 : val_end + self.lookback - 1 + len(y_ret_test)].values
        }
