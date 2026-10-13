import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Style setup for academic journals
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_price_forecast_overlay(true_close, pred_close, model_name="M2-WaveAttnNet", save_path="results/figures/fig_price_forecast.png"):
    """Plots true vs forecasted next-day closing price."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6), dpi=300)
    
    steps = np.arange(len(true_close))
    plt.plot(steps, true_close, label='Ground Truth BTC Close Price ($)', color='#111111', linewidth=1.8, alpha=0.9)
    plt.plot(steps, pred_close, label=f'{model_name} Predicted Price ($)', color='#e63946', linewidth=1.5, linestyle='--', alpha=0.85)
    
    plt.title(f"Bitcoin Next-Day Price Forecasting Evaluation ({model_name})", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Test Set Time Index (Trading Days)", fontsize=11)
    plt.ylabel("Bitcoin Price (USD)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Price Forecast Overlay: {save_path}")

def plot_benchmark_comparison_chart(benchmark_df, save_path="results/figures/fig_benchmark_comparison.png"):
    """Generates dual-axis benchmark comparison bar plot for all 9 models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
    models = benchmark_df['Model'].values
    da_values = benchmark_df['Directional Accuracy (%)'].values
    rmse_values = benchmark_df['RMSE ($)'].values
    
    # Highlight proposed SOTA model with special color
    colors = ['#457b9d' if 'Proposed' not in m else '#e63946' for m in models]
    
    # Chart 1: Directional Accuracy (%)
    bars1 = ax1.barh(models, da_values, color=colors, edgecolor='black', alpha=0.85)
    ax1.set_xlabel("Directional Accuracy (%)", fontsize=11, fontweight='bold')
    ax1.set_title("Benchmark Comparison: Directional Accuracy (DA %)", fontsize=13, fontweight='bold')
    ax1.axvline(50.0, color='gray', linestyle=':', label='Random Walk baseline (50%)')
    ax1.legend(loc='lower right')
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=9, fontweight='bold' if bar.get_facecolor()[:3]==(0.9019607843137255, 0.2235294117647059, 0.27450980392156865) else 'normal')
        
    # Chart 2: Root Mean Squared Error (RMSE $)
    bars2 = ax2.barh(models, rmse_values, color=colors, edgecolor='black', alpha=0.85)
    ax2.set_xlabel("RMSE ($) [Lower is Better]", fontsize=11, fontweight='bold')
    ax2.set_title("Benchmark Comparison: Price RMSE ($)", fontsize=13, fontweight='bold')
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + max(rmse_values)*0.01, bar.get_y() + bar.get_height()/2, f"${w:,.1f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Benchmark Comparison Chart: {save_path}")

def plot_cumulative_pnl_curve(cum_strat, cum_bench, save_path="results/figures/fig_trading_pnl.png"):
    """Plots simulated trading strategy cumulative returns vs Buy & Hold benchmark."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(11, 5.5), dpi=300)
    
    t = np.arange(len(cum_strat))
    plt.plot(t, cum_strat * 100.0, label='M2-WaveAttnNet Directional Strategy (Long/Cash)', color='#2a9d8f', linewidth=2.0)
    plt.plot(t, cum_bench * 100.0, label='Passive Buy & Hold Benchmark', color='#264653', linewidth=1.5, linestyle='--')
    
    plt.axhline(0, color='gray', linestyle=':', alpha=0.7)
    plt.title("Simulated Trading Backtest: Cumulative Strategy Returns (%)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Test Period (Trading Days)", fontsize=11)
    plt.ylabel("Cumulative Return (%)", fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Cumulative PnL Plot: {save_path}")

def plot_directional_confusion_matrix(true_dir, pred_dir_prob, save_path="results/figures/fig_confusion_matrix.png"):
    """Plots confusion matrix for directional classification."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pred_binary = (pred_dir_prob >= 0.5).astype(int)
    cm = confusion_matrix(true_dir, pred_binary)
    
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Down (0)', 'Up (1)'],
                yticklabels=['Down (0)', 'Up (1)'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("Directional Prediction Confusion Matrix", fontsize=12, fontweight='bold', pad=10)
    plt.xlabel("Predicted Market Direction", fontsize=10)
    plt.ylabel("Actual Market Direction", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Confusion Matrix Plot: {save_path}")
