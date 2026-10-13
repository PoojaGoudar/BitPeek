import os
import sys
import numpy as np
import pandas as pd
import torch

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pipeline import CryptoDataPipeline
from src.models.proposed_sota import M2WaveAttnNet
from src.training.trainer import ModelTrainer
from src.explainability.xai_suite import ExplainabilitySuite

def run_explainability_study():
    print("=" * 80)
    print(" EXPLAINABLE AI (XAI) FRAMEWORK: SHAP, INTEGRATED GRADIENTS & ATTENTION MAPS")
    print("=" * 80)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # 1. Pipeline & Ingestion
    pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30)
    data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=32)

    # 2. Train SOTA Model
    model = M2WaveAttnNet(input_dim=data['input_dim'], seq_len=data['seq_len'], hidden_dim=64, num_heads=4)
    trainer = ModelTrainer(model, lr=1e-3, max_epochs=25, patience=8)
    print("[*] Training Proposed M2-WaveAttnNet for Explainability Study...")
    trainer.train(data['train_loader'], data['val_loader'], verbose=False)

    # 3. Initialize Explainability Suite
    xai = ExplainabilitySuite(model, feature_names=data['feature_names'])

    # 4. Attention Map Extraction & Visualization
    print("[*] Extracting Cross-Attention Temporal Saliency Matrices...")
    attn_matrix = xai.extract_attention_weights(data['test_loader'])
    if attn_matrix is not None:
        xai.plot_attention_heatmap(attn_matrix, save_path="results/figures/fig_attention_heatmap.png")

    # 5. SHAP Feature Attribution
    X_train_raw = data['train_ds'].X.numpy()
    X_test_raw = data['test_ds'].X.numpy()
    shap_res = xai.compute_shap_analysis(X_train_raw, X_test_raw, max_evals=150)
    
    # Save Top Ranked Features Table
    df_shap = pd.DataFrame(shap_res['ranked_features'], columns=['Feature Name', 'Mean |SHAP Value|'])
    df_shap.to_csv("results/tables/shap_feature_importance.csv", index=False)
    
    with open("results/tables/table_shap_importance.tex", "w") as f:
        f.write(df_shap.head(15).to_latex(index=False, float_format="%.5f", caption="Top 15 Most Influential Predictive Features Ranked by SHAP Attribution.", label="tab:shap_importance"))
    print("[+] Saved SHAP Importance Table: results/tables/shap_feature_importance.csv")

    # Plot SHAP Ranking
    xai.plot_feature_importance_ranking(shap_res['ranked_features'], top_n=15, save_path="results/figures/fig_feature_importance.png")

    # 6. Integrated Gradients Sample Analysis
    print("[*] Computing Integrated Gradients for Market Regime Attribution...")
    test_sample = torch.tensor(X_test_raw[:5], dtype=torch.float32)
    ig_attr = xai.compute_integrated_gradients(test_sample)
    print(f"[+] Integrated Gradients Attribution Shape: {ig_attr.shape} successfully validated.")

    return shap_res, attn_matrix

if __name__ == "__main__":
    run_explainability_study()
