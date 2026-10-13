import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pipeline import CryptoDataPipeline
from src.models.baselines import (
    VanillaLSTM, BiLSTMAttention, GRUModel, TemporalConvNet,
    TimeSeriesTransformer, PatchTST, NBeatsModel, CNNBiLSTMHybrid
)
from src.models.proposed_sota import M2WaveAttnNet
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.backtest import simulate_trading_strategy
from src.evaluation.visualizer import (
    plot_price_forecast_overlay,
    plot_benchmark_comparison_chart,
    plot_cumulative_pnl_curve,
    plot_directional_confusion_matrix
)

def run_benchmark_experiment():
    print("=" * 80)
    print(" BITCOIN FORECASTING BENCHMARK STUDY: 8 TOP DL MODELS + PROPOSED SOTA")
    print("=" * 80)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Initialize Pipeline & Ingest Data
    pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30)
    data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=32)

    input_dim = data['input_dim']
    seq_len = data['seq_len']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    test_raw_close = data['test_raw_close']

    # 2. Instantiate All 8 Baseline Models + 1 Proposed SOTA Model
    models = [
        VanillaLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2),
        BiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2),
        GRUModel(input_dim=input_dim, hidden_dim=64, num_layers=2),
        TemporalConvNet(input_dim=input_dim, num_channels=[32, 64, 64]),
        TimeSeriesTransformer(input_dim=input_dim, d_model=64, nhead=4, num_layers=2),
        PatchTST(input_dim=input_dim, seq_len=seq_len, patch_len=6, stride=3, d_model=64),
        NBeatsModel(input_dim=input_dim, seq_len=seq_len, hidden_dim=128, num_blocks=3),
        CNNBiLSTMHybrid(input_dim=input_dim, hidden_dim=64),
        M2WaveAttnNet(input_dim=input_dim, seq_len=seq_len, hidden_dim=64, num_heads=4)
    ]

    results = []
    trained_models = {}
    best_sota_preds = None

    print(f"\n[*] Commencing Systematic Training & Evaluation on {len(models)} Architectures...\n")

    for idx, model in enumerate(models, 1):
        print(f"[{idx}/{len(models)}] Training: {model.name}...")
        trainer = ModelTrainer(model, lr=1e-3, max_epochs=25, patience=8)
        trainer.train(train_loader, val_loader, verbose=False)

        # Generate Test Predictions
        preds = trainer.predict(test_loader)
        
        # Convert predicted log returns back to absolute dollar prices: P_{t+1} = P_t * exp(r_{t+1})
        pred_returns = preds['pred_returns']
        pred_directions = preds['pred_directions']
        true_returns = preds['true_returns']
        true_directions = preds['true_directions']
        true_closes = preds['true_closes']
        
        # Test close prices reconstruction
        pred_closes = test_raw_close * np.exp(pred_returns)

        # Compute comprehensive metrics
        metrics = compute_all_metrics(
            true_close=true_closes,
            pred_close=pred_closes,
            true_returns=true_returns,
            pred_returns=pred_returns,
            true_dir=true_directions,
            pred_dir_prob=pred_directions
        )

        # Simulate Trading Strategy
        backtest_res = simulate_trading_strategy(true_returns, pred_returns, pred_directions)

        metrics_entry = {
            'Model': model.name,
            'RMSE ($)': metrics['RMSE ($)'],
            'MAE ($)': metrics['MAE ($)'],
            'MAPE (%)': metrics['MAPE (%)'],
            'R2 Score': metrics['R2 Score'],
            'Directional Accuracy (%)': metrics['Directional Accuracy (%)'],
            'Precision (%)': metrics['Precision (%)'],
            'Recall (%)': metrics['Recall (%)'],
            'F1-Score (%)': metrics['F1-Score (%)'],
            'Strategy Return (%)': backtest_res['Strategy Total Return (%)'],
            'Sharpe Ratio': backtest_res['Strategy Sharpe Ratio'],
            'Max Drawdown (%)': backtest_res['Strategy Max Drawdown (%)']
        }
        results.append(metrics_entry)
        trained_models[model.name] = (model, trainer, preds, backtest_res)

        if "Proposed" in model.name:
            best_sota_preds = (true_closes, pred_closes, true_directions, pred_directions, backtest_res)

        print(f"    --> DA: {metrics['Directional Accuracy (%)']:.2f}% | RMSE: ${metrics['RMSE ($)']:,.2f} | Sharpe: {backtest_res['Strategy Sharpe Ratio']:.2f}")

    # 3. Create DataFrame and Output Tables
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Directional Accuracy (%)', ascending=False).reset_index(drop=True)
    
    csv_path = "results/tables/benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n[+] Benchmark Results Saved to: {csv_path}")

    # Print Formatted Table
    table_str = tabulate(results_df, headers='keys', tablefmt='fancy_grid', floatfmt=".2f")
    print("\n" + "=" * 80)
    print(" BENCHMARK COMPARATIVE RESULTS TABLE")
    print("=" * 80)
    print(table_str)

    # 4. Generate LaTeX Table Code for Research Paper
    latex_path = "results/tables/table1_benchmark.tex"
    with open(latex_path, "w") as f:
        f.write(results_df.to_latex(index=False, float_format="%.2f", caption="Comparative Benchmark Performance Across 8 Deep Learning Baselines and the Proposed M2-WaveAttnNet Architecture on Bitcoin Forecasting.", label="tab:benchmark_results"))
    print(f"[+] LaTeX Table Exported: {latex_path}")

    # 5. Generate Figures
    print("\n[*] Generating Publication Figures...")
    plot_benchmark_comparison_chart(results_df, save_path="results/figures/fig_benchmark_comparison.png")
    
    if best_sota_preds is not None:
        true_c, pred_c, true_d, pred_d, bt_res = best_sota_preds
        plot_price_forecast_overlay(true_c, pred_c, model_name="Proposed M2-WaveAttnNet", save_path="results/figures/fig_price_forecast.png")
        plot_directional_confusion_matrix(true_d, pred_d, save_path="results/figures/fig_confusion_matrix.png")
        plot_cumulative_pnl_curve(bt_res['cum_strategy'], bt_res['cum_benchmark'], save_path="results/figures/fig_trading_pnl.png")

    return results_df, trained_models, pipeline, data

if __name__ == "__main__":
    run_benchmark_experiment()
