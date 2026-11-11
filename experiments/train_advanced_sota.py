import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.data.pipeline import CryptoDataPipeline
from src.models.proposed_sota import M2WaveAttnNet
from src.training.trainer import ModelTrainer
from src.training.losses import DirectionCalibratedLoss
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.backtest import simulate_trading_strategy

def hyperparameter_tune_sota(data, max_trials=6):
    """
    Advanced Hyperparameter Search & Optimization for M2-WaveAttnNet.
    Searches across learning rates, hidden dimensions, dropout rates,
    attention heads, and loss balancing weights.
    """
    print("=" * 80)
    print(" ADVANCED HYPERPARAMETER SEARCH FOR PROPOSED M2-WAVEATTNNET (HYBRID SOTA)")
    print("=" * 80)

    input_dim = data['input_dim']
    seq_len = data['seq_len']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    test_raw_close = data['test_raw_close']

    # Search candidates
    configs = [
        {'hidden_dim': 64, 'num_heads': 4, 'lr': 5e-4, 'dropout': 0.15, 'alpha': 50.0, 'gamma': 15.0, 'epochs': 35},
        {'hidden_dim': 64, 'num_heads': 4, 'lr': 7e-4, 'dropout': 0.10, 'alpha': 60.0, 'gamma': 20.0, 'epochs': 35},
        {'hidden_dim': 96, 'num_heads': 4, 'lr': 5e-4, 'dropout': 0.15, 'alpha': 50.0, 'gamma': 15.0, 'epochs': 35},
        {'hidden_dim': 64, 'num_heads': 8, 'lr': 6e-4, 'dropout': 0.15, 'alpha': 40.0, 'gamma': 15.0, 'epochs': 35},
        {'hidden_dim': 96, 'num_heads': 8, 'lr': 4e-4, 'dropout': 0.20, 'alpha': 50.0, 'gamma': 20.0, 'epochs': 35},
        {'hidden_dim': 128, 'num_heads': 8, 'lr': 3e-4, 'dropout': 0.20, 'alpha': 60.0, 'gamma': 25.0, 'epochs': 35},
    ][:max_trials]

    best_val_score = -float('inf')
    best_config = None
    best_model = None
    best_preds = None
    trial_records = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n--- [Trial {i}/{len(configs)}] Config: hidden={cfg['hidden_dim']}, heads={cfg['num_heads']}, lr={cfg['lr']}, drop={cfg['dropout']}, alpha={cfg['alpha']}, gamma={cfg['gamma']} ---")
        
        # Instantiate tuned model
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        model = M2WaveAttnNet(
            input_dim=input_dim,
            seq_len=seq_len,
            hidden_dim=cfg['hidden_dim'],
            num_heads=cfg['num_heads'],
            dropout=cfg['dropout']
        )

        criterion = DirectionCalibratedLoss(alpha=cfg['alpha'], beta=1.0, gamma=cfg['gamma'])
        trainer = ModelTrainer(model, lr=cfg['lr'], max_epochs=cfg['epochs'], patience=10)
        trainer.criterion = criterion

        history = trainer.train(train_loader, val_loader, verbose=False)
        
        # Validation score
        val_loss, val_m = trainer.evaluate(val_loader)
        val_da = val_m['DA'] * 100.0
        val_mae = val_m['MAE']
        
        # Composite validation metric (prioritizing high directional accuracy and low error)
        val_composite = val_da - 100.0 * val_mae
        print(f"    Validation Results -> Loss: {val_loss:.4f} | Directional Accuracy: {val_da:.2f}% | MAE: {val_mae:.5f}")

        # Test evaluation
        preds = trainer.predict(test_loader)
        pred_returns = preds['pred_returns']
        pred_directions = preds['pred_directions']
        true_returns = preds['true_returns']
        true_directions = preds['true_directions']
        true_closes = preds['true_closes']
        pred_closes = test_raw_close * np.exp(pred_returns)

        metrics = compute_all_metrics(true_closes, pred_closes, true_returns, pred_returns, true_directions, pred_directions)
        
        trial_records.append({
            'Trial': f"Trial {i}",
            'Hidden': cfg['hidden_dim'],
            'Heads': cfg['num_heads'],
            'LR': cfg['lr'],
            'Val DA (%)': round(val_da, 2),
            'Test DA (%)': round(metrics['Directional Accuracy (%)'], 2),
            'Test RMSE ($)': round(metrics['RMSE ($)'], 2),
            'Test F1 (%)': round(metrics['F1-Score (%)'], 2)
        })

        if val_composite > best_val_score:
            best_val_score = val_composite
            best_config = cfg
            best_model = copy.deepcopy(model)
            best_preds = preds
            best_metrics = metrics

    print("\n" + "=" * 80)
    print(" HYPERPARAMETER TUNING SEARCH SUMMARY")
    print("=" * 80)
    df_trials = pd.DataFrame(trial_records)
    print(tabulate(df_trials, headers='keys', tablefmt='grid'))

    print(f"\n[+] Selected Optimal Configuration: {best_config}")
    print(f"[+] Optimal Model Test Directional Accuracy: {best_metrics['Directional Accuracy (%)']:.2f}%")
    print(f"[+] Optimal Model Test RMSE: ${best_metrics['RMSE ($)']:.2f}")

    os.makedirs("results", exist_ok=True)
    torch.save(best_model.state_dict(), "results/m2waveattnnet_best_tuned.pt")
    print("[+] Saved optimal model checkpoint to results/m2waveattnnet_best_tuned.pt")

    return best_model, best_config, best_metrics

if __name__ == "__main__":
    pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30)
    data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=32)
    hyperparameter_tune_sota(data)
