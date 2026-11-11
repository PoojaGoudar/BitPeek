"""
Constituent Layer Comparison Experiment  (v3 – Macro Features + Pre-training + Focal Loss + Confidence Thresholding)
====================================================================================================================
Architecture under test:
  L1: Channel Feature Gating   -> Adaptive feature selection only
  L2: Multi-Scale Inception    -> Multi-scale temporal convolutions only
  L3: Temporal Cross-Attention -> Self-attention temporal fusion only
  L4: Residual Dense Trunk     -> Deep MLP mapping only
  FULL: M2-WaveAttnNet (Hybrid)-> All 4 fused (pre-trained encoder + focal fine-tune)

Key Upgrades:
  1. Enriched macro features (ETH momentum, DXY dollar index, S&P 500, Gold) with zero-leakage alignment
  2. Fixed validation DA evaluation (flattened 1D array comparison)
  3. Confidence thresholding (selective prediction): evaluates high-conviction trades (τ >= 0.50, 0.55, 0.60, 0.65, 0.70)
"""
import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

from src.data.pipeline import CryptoDataPipeline
from src.models.proposed_sota import M2WaveAttnNet
from src.models.constituent_models import (
    StandaloneChannelGatingNet,
    StandaloneInceptionConvNet,
    StandaloneTemporalAttnNet,
    StandaloneResidualTrunkNet
)
from src.training.trainer import ModelTrainer
from src.training.losses import DirectionCalibratedLoss, FocalDirectionLoss, MaskedReconstructionLoss
from src.evaluation.metrics import compute_all_metrics

torch.manual_seed(42)
np.random.seed(42)

# ── Constituent-model config ────────────────────────────────────────────────
OPTIMAL_CFG = {
    'hidden_dim': 96,
    'num_heads': 4,
    'lr': 5e-4,
    'dropout': 0.15,
    'alpha': 50.0,
    'gamma': 15.0,
    'epochs': 40,
    'patience': 12,
    'weight_decay': 1e-4
}

# ── Hybrid model config ─────────────────────────────────────────────────────
HYBRID_CFG = {
    'hidden_dim': 128,
    'num_heads': 4,
    'lr': 3e-4,
    'dropout': 0.15,
    'alpha': 50.0,
    'gamma': 10.0,
    'epochs': 55,
    'patience': 18,
    'weight_decay': 5e-4,
    # Pre-training settings
    'pretrain_epochs': 15,
    'pretrain_lr': 1e-3,
    'mask_ratio': 0.15,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Self-supervised pre-training helper
# ─────────────────────────────────────────────────────────────────────────────
def pretrain_encoder(model: M2WaveAttnNet, train_loader, cfg: dict) -> M2WaveAttnNet:
    """
    Stage 1: Masked time-step reconstruction pre-training.
    Randomly mask ~15% of time-steps in each window and train the encoder
    + reconstruction head to predict the original feature values.
    Forces the backbone to learn robust temporal representations before
    it sees the direction labels.
    """
    print("\n[*] Stage 1 — Self-supervised pre-training (masked reconstruction)")
    recon_loss_fn = MaskedReconstructionLoss()
    optimizer = AdamW(
        list(model.parameters()),
        lr=cfg['pretrain_lr'],
        weight_decay=cfg['weight_decay']
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

    model.to(DEVICE)
    model.train()
    best_loss = float('inf')
    best_state = None

    for epoch in range(1, cfg['pretrain_epochs'] + 1):
        epoch_losses = []
        for batch in train_loader:
            x = batch['x'].to(DEVICE)        # (B, T, D)
            B, T, D = x.shape

            # Create random binary mask (1 = masked position to predict)
            mask_t = (torch.rand(B, T, device=DEVICE) < cfg['mask_ratio']).float()  # (B, T)
            mask_full = mask_t.unsqueeze(-1).expand(-1, -1, D)                       # (B, T, D)

            # Zero-out masked positions before feeding to encoder
            x_masked = x * (1.0 - mask_full)

            optimizer.zero_grad()
            enc_out = model.encode(x_masked)          # (B, T, H)
            recon   = model.recon_head(enc_out)        # (B, T, D)
            loss    = recon_loss_fn(recon, x, mask_full)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        avg = np.mean(epoch_losses)
        if avg < best_loss:
            best_loss = avg
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Pre-train Epoch [{epoch:02d}/{cfg['pretrain_epochs']:02d}] - Recon Loss: {avg:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[+] Pre-training complete. Best recon loss: {best_loss:.5f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Focal-loss fine-tuning helper
# ─────────────────────────────────────────────────────────────────────────────
def finetune_with_focal(model: M2WaveAttnNet, train_loader, val_loader, cfg: dict) -> dict:
    """
    Stage 2: Supervised fine-tuning with FocalDirectionLoss.
    Early-stopping on validation DA with proper 1D vector broadcasting.
    Uses cosine warm-restart schedule with linear warm-up.
    """
    print("\n[*] Stage 2 — Supervised fine-tuning with FocalDirectionLoss")
    criterion = FocalDirectionLoss(
        alpha=cfg['alpha'],
        beta=2.0,
        gamma=cfg['gamma'],
        focal_gamma=2.0,
        focal_alpha=0.6,
        label_smoothing=0.05
    )
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_da = 0.0
    best_state  = copy.deepcopy(model.state_dict())
    patience_ctr = 0
    history = {'train_loss': [], 'val_loss': [], 'val_da': []}

    # ---- Warm-up: freeze encoder for first 4 epochs, train heads only --------
    WARMUP = 4
    encoder_params = set(id(p) for p in list(model.channel_gate.parameters())
                         + list(model.stem.parameters())
                         + list(model.inception1.parameters())
                         + list(model.inception2.parameters())
                         + list(model.conv_skip.parameters())
                         + list(model.attn_fusion.parameters())
                         + list(model.attn_fusion2.parameters())
                         + list(model.attn_fusion3.parameters()))

    for epoch in range(1, cfg['epochs'] + 1):
        # Warm-up toggle
        for p in model.parameters():
            p.requires_grad_(True)
        if epoch <= WARMUP:
            for p in model.parameters():
                if id(p) in encoder_params:
                    p.requires_grad_(False)

        model.train()
        train_losses = []
        for batch in train_loader:
            x     = batch['x'].to(DEVICE)
            y_ret = batch['y_return'].to(DEVICE)
            y_dir = batch['y_dir'].to(DEVICE)
            optimizer.zero_grad()
            pred_ret, pred_dir = model(x)
            loss, _ = criterion(pred_ret, y_ret, pred_dir, y_dir)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)

        # ---- Validation ----
        model.eval()
        val_losses, pred_dirs_all, true_dirs_all = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x     = batch['x'].to(DEVICE)
                y_ret = batch['y_return'].to(DEVICE)
                y_dir = batch['y_dir'].to(DEVICE)
                pred_ret, pred_dir = model(x)
                v_loss, _ = criterion(pred_ret, y_ret, pred_dir, y_dir)
                val_losses.append(v_loss.item())
                pred_dirs_all.append((pred_dir.cpu().numpy() >= 0.5).astype(int).flatten())
                true_dirs_all.append(y_dir.cpu().numpy().astype(int).flatten())

        avg_val = np.mean(val_losses)
        # Proper 1D vector comparison (fixes NumPy broadcasting bug)
        pred_flat = np.concatenate(pred_dirs_all)
        true_flat = np.concatenate(true_dirs_all)
        val_da = float(np.mean(pred_flat == true_flat))

        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['val_da'].append(val_da)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch [{epoch:02d}/{cfg['epochs']:02d}] - "
                  f"Train Loss: {avg_train:.5f} | Val Loss: {avg_val:.5f} | "
                  f"Val DA: {val_da*100:.2f}%")

        # Early stopping on best val DA
        if val_da > best_val_da:
            best_val_da  = val_da
            patience_ctr = 0
            best_state   = copy.deepcopy(model.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= cfg['patience']:
                print(f"[*] Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    print(f"[+] Fine-tuning complete. Best Val DA: {best_val_da*100:.2f}%")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Thresholding Helper (Selective Prediction)
# ─────────────────────────────────────────────────────────────────────────────
def compute_confidence_threshold_stats(pred_probs, true_dirs, thresholds=[0.50, 0.55, 0.60, 0.65, 0.70]):
    """
    Evaluates selective prediction where the model only trades if:
        p >= tau (UP) or p <= (1 - tau) (DOWN).
    Returns a list of dicts with threshold, active trades, coverage, DA, and F1.
    """
    total = len(pred_probs)
    stats = []
    for tau in thresholds:
        active_mask = (pred_probs >= tau) | (pred_probs <= (1.0 - tau))
        active_count = int(np.sum(active_mask))
        coverage = (active_count / total) * 100.0

        if active_count > 0:
            pred_active = (pred_probs[active_mask] >= 0.5).astype(int)
            true_active = true_dirs[active_mask].astype(int)
            da = float(np.mean(pred_active == true_active)) * 100.0

            tp = np.sum((pred_active == 1) & (true_active == 1))
            fp = np.sum((pred_active == 1) & (true_active == 0))
            fn = np.sum((pred_active == 0) & (true_active == 1))
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * (prec * rec) / (prec + rec + 1e-8) * 100.0
        else:
            da, f1 = 0.0, 0.0

        stats.append({
            'tau': tau,
            'active_count': active_count,
            'total': total,
            'coverage': round(coverage, 2),
            'da': round(da, 2),
            'f1': round(f1, 2)
        })
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────
def run_constituent_comparison():
    print("=" * 80)
    print(" CONSTITUENT LAYER COMPARISON: INDIVIDUAL vs FULL HYBRID M2-WAVEATTNNET")
    print(" (v3: Macro Features + Pre-training + Focal Loss + Confidence Thresholding)")
    print("=" * 80)

    pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01",
                                  end_date="2025-01-01", lookback=30)
    data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=64)

    input_dim      = data['input_dim']
    seq_len        = data['seq_len']
    train_loader   = data['train_loader']
    val_loader     = data['val_loader']
    test_loader    = data['test_loader']
    test_raw_close = data['test_raw_close']

    print(f"\n[+] Input Features: {input_dim} (including ETH, DXY, SPX, Gold macro proxies)")
    print(f"[+] Lookback Window: {seq_len} days")
    print(f"[+] Dataset Split: Train={len(data['train_ds'])}, Val={len(data['val_ds'])}, Test={len(data['test_ds'])}")

    print("\nArchitecture under test:")
    print("  [L1] Channel Feature Gating   -> Adaptive feature selection only")
    print("  [L2] Multi-Scale Inception    -> Multi-scale temporal convolutions only")
    print("  [L3] Temporal Cross-Attention -> Self-attention temporal fusion only")
    print("  [L4] Residual Dense Trunk     -> Deep MLP mapping only")
    print("  [FULL] M2-WaveAttnNet (Hybrid)-> Pre-trained + Focal-loss fine-tuned\n")

    # ── Constituent models (standard training) ────────────────────────────────
    constituent_models = [
        ("L1: Channel Gating",
         StandaloneChannelGatingNet(input_dim, seq_len, OPTIMAL_CFG['hidden_dim'], OPTIMAL_CFG['dropout']),
         OPTIMAL_CFG),
        ("L2: Multi-Scale Inception Conv",
         StandaloneInceptionConvNet(input_dim, seq_len, OPTIMAL_CFG['hidden_dim'], OPTIMAL_CFG['dropout']),
         OPTIMAL_CFG),
        ("L3: Temporal Cross-Attention",
         StandaloneTemporalAttnNet(input_dim, seq_len, OPTIMAL_CFG['hidden_dim'], OPTIMAL_CFG['num_heads'], OPTIMAL_CFG['dropout']),
         OPTIMAL_CFG),
        ("L4: Residual Dense Trunk",
         StandaloneResidualTrunkNet(input_dim, seq_len, OPTIMAL_CFG['hidden_dim'], OPTIMAL_CFG['dropout']),
         OPTIMAL_CFG),
    ]

    results = []
    model_predictions = {}  # Store probabilities for confidence thresholding

    for label, model, cfg in constituent_models:
        torch.manual_seed(42)
        np.random.seed(42)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[*] Training: {label}  ({n_params:,} params) | wd={cfg['weight_decay']}, dropout={cfg['dropout']}")
        trainer = ModelTrainer(model, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                               max_epochs=cfg['epochs'], patience=cfg['patience'])
        trainer.criterion = DirectionCalibratedLoss(alpha=cfg['alpha'], beta=1.0, gamma=cfg['gamma'])
        trainer.train(train_loader, val_loader, verbose=True)

        preds      = trainer.predict(test_loader)
        pred_closes = test_raw_close * np.exp(preds['pred_returns'])
        m = compute_all_metrics(test_raw_close, pred_closes,
                                preds['true_returns'], preds['pred_returns'],
                                preds['true_directions'], preds['pred_directions'])
        results.append({
            'Architecture': label,
            'Params': n_params,
            'Test DA (%)':  round(m['Directional Accuracy (%)'], 2),
            'Test F1 (%)':  round(m['F1-Score (%)'], 2),
            'RMSE ($)':     round(m['RMSE ($)'], 2),
            'MAE ($)':      round(m['MAE ($)'], 2),
            'R2':           round(m['R2 Score'], 4),
        })
        model_predictions[label] = {
            'pred_probs': preds['pred_directions'],
            'true_dirs':  preds['true_directions']
        }
        print(f"    --> DA: {m['Directional Accuracy (%)']:.2f}% | "
              f"F1: {m['F1-Score (%)']:.2f}% | RMSE: ${m['RMSE ($)']:,.2f}")

    # ── Full Hybrid: 2-stage (pre-train → focal fine-tune) ───────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    hybrid_model = M2WaveAttnNet(
        input_dim, seq_len,
        HYBRID_CFG['hidden_dim'],
        HYBRID_CFG['num_heads'],
        HYBRID_CFG['dropout']
    )
    n_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"\n[*] Training: FULL: M2-WaveAttnNet (Hybrid)  ({n_params:,} params) — 2-stage")

    # Stage 1: self-supervised pre-training
    hybrid_model = pretrain_encoder(hybrid_model, train_loader, HYBRID_CFG)

    # Stage 2: focal-loss fine-tuning (early-stop on true val DA)
    finetune_with_focal(hybrid_model, train_loader, val_loader, HYBRID_CFG)

    # Evaluation
    hybrid_model.eval()
    pred_rets, pred_dirs, true_rets, true_dirs, true_closes = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(DEVICE)
            pr, pd_ = hybrid_model(x)
            pred_rets.append(pr.cpu().numpy())
            pred_dirs.append(pd_.cpu().numpy())
            true_rets.append(batch['y_return'].numpy())
            true_dirs.append(batch['y_dir'].numpy())
            true_closes.append(batch['y_close'].numpy())

    pred_returns    = np.concatenate(pred_rets).flatten()
    pred_directions = np.concatenate(pred_dirs).flatten()
    true_returns    = np.concatenate(true_rets).flatten()
    true_directions = np.concatenate(true_dirs).flatten()
    pred_closes     = test_raw_close * np.exp(pred_returns)

    m = compute_all_metrics(test_raw_close, pred_closes, true_returns, pred_returns,
                            true_directions, pred_directions)
    hybrid_label = 'FULL: M2-WaveAttnNet (Hybrid)'
    results.append({
        'Architecture': hybrid_label,
        'Params': n_params,
        'Test DA (%)':  round(m['Directional Accuracy (%)'], 2),
        'Test F1 (%)':  round(m['F1-Score (%)'], 2),
        'RMSE ($)':     round(m['RMSE ($)'], 2),
        'MAE ($)':      round(m['MAE ($)'], 2),
        'R2':           round(m['R2 Score'], 4),
    })
    model_predictions[hybrid_label] = {
        'pred_probs': pred_directions,
        'true_dirs':  true_directions
    }
    print(f"    --> DA: {m['Directional Accuracy (%)']:.2f}% | "
          f"F1: {m['F1-Score (%)']:.2f}% | RMSE: ${m['RMSE ($)']:,.2f}")

    # ── Table 1: Standard Academic Comparison ─────────────────────────────────
    print("\n" + "=" * 80)
    print(" TABLE 1: CONSTITUENT COMPARISON RESULTS (Standard Evaluation - 100% Coverage)")
    print("=" * 80)
    df_std = pd.DataFrame(results).sort_values('Test DA (%)', ascending=False).reset_index(drop=True)
    print(tabulate(df_std, headers='keys', tablefmt='grid', showindex=True))

    os.makedirs("results/tables", exist_ok=True)
    df_std.to_csv("results/tables/constituent_comparison.csv", index=False)
    print("\n[+] Saved standard comparison to results/tables/constituent_comparison.csv")

    # ── Table 2: Confidence Thresholding (Selective Prediction) ───────────────
    print("\n" + "=" * 80)
    print(" TABLE 2: CONFIDENCE THRESHOLDING (Selective Prediction Across Models)")
    print(" Evaluating Directional Accuracy (%) when betting only on high-conviction days")
    print("=" * 80)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    thresh_rows = []
    for model_name, pdata in model_predictions.items():
        stats = compute_confidence_threshold_stats(pdata['pred_probs'], pdata['true_dirs'], thresholds)
        row = {'Architecture': model_name}
        for s in stats:
            t = s['tau']
            row[f"τ={t:.2f} DA% (Cov%)"] = f"{s['da']:.1f}% ({s['coverage']:.0f}%)"
        thresh_rows.append(row)

    df_thresh = pd.DataFrame(thresh_rows)
    print(tabulate(df_thresh, headers='keys', tablefmt='grid', showindex=False))

    # ── Table 3: Detailed Breakdown for Full Hybrid Model ─────────────────────
    print("\n" + "=" * 80)
    print(" TABLE 3: FULL M2-WAVEATTNNET CONFIDENCE-SCALING BREAKDOWN")
    print("=" * 80)
    hybrid_stats = compute_confidence_threshold_stats(
        model_predictions[hybrid_label]['pred_probs'],
        model_predictions[hybrid_label]['true_dirs'],
        thresholds
    )
    detailed_rows = []
    for s in hybrid_stats:
        detailed_rows.append({
            'Threshold (τ)': f"p ≥ {s['tau']:.2f} or p ≤ {1.0 - s['tau']:.2f}",
            'Active Trades': f"{s['active_count']}/{s['total']}",
            'Coverage (%)': f"{s['coverage']:.1f}%",
            'Directional Accuracy (%)': f"{s['da']:.2f}%",
            'F1 Score (%)': f"{s['f1']:.2f}%"
        })
    df_hybrid_detail = pd.DataFrame(detailed_rows)
    print(tabulate(df_hybrid_detail, headers='keys', tablefmt='grid', showindex=False))
    df_hybrid_detail.to_csv("results/tables/confidence_thresholding.csv", index=False)
    print("\n[+] Saved confidence thresholding to results/tables/confidence_thresholding.csv")

    # ── Verdict ───────────────────────────────────────────────────────────────
    full_da              = df_std[df_std['Architecture'].str.contains('FULL')]['Test DA (%)'].values[0]
    best_individual_da   = df_std[~df_std['Architecture'].str.contains('FULL')]['Test DA (%)'].max()
    best_individual_name = df_std[~df_std['Architecture'].str.contains('FULL')].loc[
        df_std[~df_std['Architecture'].str.contains('FULL')]['Test DA (%)'].idxmax(), 'Architecture'
    ]

    print("\n" + "=" * 80)
    print(" VERDICT & SUMMARY")
    print("=" * 80)
    delta = full_da - best_individual_da
    print(f"  • Full Hybrid M2-WaveAttnNet (100% Coverage DA): {full_da:.2f}%")
    print(f"  • Best Individual Component ({best_individual_name}): {best_individual_da:.2f}%")
    if delta > 0:
        print(f"  ✅ Hybrid Advantage: +{delta:.2f}% DA over best standalone component.")
    else:
        print(f"  ℹ️ Baseline Delta: {delta:.2f}% DA at raw 100% coverage.")

    best_selective = max(s['da'] for s in hybrid_stats)
    best_cov = [s['coverage'] for s in hybrid_stats if s['da'] == best_selective][0]
    print(f"  🚀 Selective Conviction Trading Peak DA: {best_selective:.2f}% (Coverage: {best_cov:.1f}%)")
    print("=" * 80)

    return df_std, df_hybrid_detail


if __name__ == "__main__":
    run_constituent_comparison()
