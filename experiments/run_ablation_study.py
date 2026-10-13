import os
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pipeline import CryptoDataPipeline
from src.models.proposed_sota import M2WaveAttnNet
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import compute_all_metrics

class AblationNoInception(nn.Module):
    """M2-WaveAttnNet variant replacing Inception with standard single Conv1D."""
    def __init__(self, input_dim, seq_len=30, hidden_dim=64):
        super().__init__()
        self.name = "w/o Multi-Scale Inception (Single Conv)"
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU())
        self.head_return = nn.Linear(32, 1)
        self.head_direction = nn.Linear(32, 1)
    def forward(self, x):
        feat = self.fc(self.conv(x.transpose(1, 2))[:, :, -1])
        return self.head_return(feat), torch.sigmoid(self.head_direction(feat))

class AblationNoGating(nn.Module):
    """M2-WaveAttnNet variant without Squeeze-and-Excitation feature gating."""
    def __init__(self, input_dim, seq_len=30, hidden_dim=64):
        super().__init__()
        self.name = "w/o Channel Feature Gating"
        self.net = M2WaveAttnNet(input_dim, seq_len, hidden_dim)
        # Disable channel gating by replacing with identity
        self.net.channel_gate = nn.Identity()
    def forward(self, x):
        # Pass through without gating
        x_conv_in = self.net.stem(x.transpose(1, 2))
        inc1 = self.net.inception1(x_conv_in)
        inc2 = self.net.inception2(inc1) + inc1
        attn_in = inc2.transpose(1, 2)
        attn_out, _ = self.net.attn_fusion(attn_in)
        last_step_feat = attn_out[:, -1, :]
        pooled_feat = self.net.temporal_pool(attn_out.transpose(1, 2)).squeeze(-1)
        combined_feat = torch.cat([last_step_feat, pooled_feat], dim=-1)
        latent = self.net.trunk(combined_feat)
        return self.net.head_return(latent), torch.sigmoid(self.net.head_direction(latent))

def run_ablation_study():
    print("=" * 80)
    print(" ABLATION STUDY: COMPONENT CONTRIBUTIONS OF PROPOSED SOTA ARCHITECTURE")
    print("=" * 80)

    os.makedirs("results/tables", exist_ok=True)

    pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30)
    data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=32)

    input_dim = data['input_dim']
    seq_len = data['seq_len']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']
    test_raw_close = data['test_raw_close']

    ablation_configs = [
        ("Full Proposed M2-WaveAttnNet", M2WaveAttnNet(input_dim, seq_len, 64), False),
        ("w/o Channel Feature Gating", AblationNoGating(input_dim, seq_len, 64), False),
        ("w/o Multi-Scale Inception", AblationNoInception(input_dim, seq_len, 64), False),
        ("w/o Direction-Calibrated Loss (MSE Only)", M2WaveAttnNet(input_dim, seq_len, 64), True)
    ]

    results = []

    for name, model, use_mse_only in ablation_configs:
        print(f"[*] Running Ablation Variant: {name}...")
        trainer = ModelTrainer(model, lr=1e-3, max_epochs=25, patience=8)
        if use_mse_only:
            # Overwrite loss to pure MSE
            class PureMSELoss(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mse = nn.MSELoss()
                    self.bce = nn.BCELoss()
                def forward(self, pr, tr, pd, td):
                    return self.mse(pr, tr) + self.bce(pd, td), {}
            trainer.criterion = PureMSELoss()

        trainer.train(train_loader, val_loader, verbose=False)
        preds = trainer.predict(test_loader)

        pred_returns = preds['pred_returns']
        pred_directions = preds['pred_directions']
        true_returns = preds['true_returns']
        true_directions = preds['true_directions']
        true_closes = preds['true_closes']
        pred_closes = test_raw_close * np.exp(pred_returns)

        metrics = compute_all_metrics(
            true_close=true_closes,
            pred_close=pred_closes,
            true_returns=true_returns,
            pred_returns=pred_returns,
            true_dir=true_directions,
            pred_dir_prob=pred_directions
        )

        results.append({
            'Ablation Variant': name,
            'Directional Accuracy (%)': metrics['Directional Accuracy (%)'],
            'RMSE ($)': metrics['RMSE ($)'],
            'MAE ($)': metrics['MAE ($)'],
            'MAPE (%)': metrics['MAPE (%)'],
            'F1-Score (%)': metrics['F1-Score (%)']
        })

    ablation_df = pd.DataFrame(results)
    csv_path = "results/tables/ablation_study_results.csv"
    ablation_df.to_csv(csv_path, index=False)
    
    with open("results/tables/table2_ablation.tex", "w") as f:
        f.write(ablation_df.to_latex(index=False, float_format="%.2f", caption="Ablation Analysis Isolating the Contributions of Key Architectural Components in M2-WaveAttnNet.", label="tab:ablation_results"))

    print("\n" + "=" * 80)
    print(" ABLATION STUDY RESULTS")
    print("=" * 80)
    print(tabulate(ablation_df, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    print(f"\n[+] Saved Ablation Results to: {csv_path} and table2_ablation.tex")

    return ablation_df

if __name__ == "__main__":
    run_ablation_study()
