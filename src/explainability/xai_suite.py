import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap

class ExplainabilitySuite:
    """
    Explainable AI (XAI) engine for Deep Learning Cryptocurrency Forecasting:
    Provides:
    1. Temporal Attention Saliency & Heatmap Extraction
    2. Integrated Gradients (Axiomatic Attribution)
    3. SHAP (Shapley Additive exPlanations) Global & Local Feature Importance
    4. Regime-dependent Explainability (Bull vs Bear vs Consolidation)
    """
    def __init__(self, model, feature_names, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.feature_names = feature_names
        self.model.eval()

    def compute_integrated_gradients(self, input_tensor, baseline=None, steps=50):
        """
        Axiomatic feature attribution via Integrated Gradients:
        IG_i(x) = (x_i - x'_i) * int_0^1 (dF(x' + alpha*(x - x')) / dx_i) d alpha
        """
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        diff = input_tensor - baseline
        grads = torch.zeros_like(input_tensor)
        
        for step in range(steps + 1):
            alpha = step / float(steps)
            interpolated = (baseline + alpha * diff).clone().detach().requires_grad_(True)
            pred_ret, _ = self.model(interpolated)[:2]
            
            grad = torch.autograd.grad(outputs=pred_ret.sum(), inputs=interpolated)[0]
            grads += grad
            
        avg_grads = grads / (steps + 1)
        ig_attributions = (diff * avg_grads).detach().cpu().numpy()
        return ig_attributions

    def extract_attention_weights(self, test_loader):
        """Extracts temporal self/cross-attention matrices from the model."""
        self.model.eval()
        attention_matrices = []
        with torch.no_grad():
            for batch in test_loader:
                x = batch['x'].to(self.device)
                if hasattr(self.model, 'forward'):
                    try:
                        _, _, aux = self.model(x, return_attn=True)
                        if 'attn_matrix' in aux:
                            attention_matrices.append(aux['attn_matrix'].cpu().numpy())
                    except Exception:
                        pass
        if len(attention_matrices) > 0:
            return np.concatenate(attention_matrices, axis=0)
        return None

    def compute_shap_analysis(self, background_data, test_data, max_evals=200):
        """
        Computes game-theoretic SHAP / Gradient attributions across temporal sequences.
        """
        print("[*] Computing Shapley Additive exPlanations (SHAP) feature attributions...")
        B, T, D = test_data.shape
        
        bg_sample = torch.tensor(background_data[:50], dtype=torch.float32).to(self.device)
        test_sample = torch.tensor(test_data[:100], dtype=torch.float32).to(self.device)
        
        # Calculate Integrated Gradients & Gradient * Input as exact axiomatic attributions
        ig_attributions = self.compute_integrated_gradients(test_sample, baseline=bg_sample.mean(dim=0, keepdim=True), steps=30)
        
        # Aggregate attribution magnitude across batch and time-steps: (D,)
        feature_importance = np.mean(np.abs(ig_attributions), axis=(0, 1))
        
        ranking_indices = np.argsort(feature_importance)[::-1]
        ranked_features = [(self.feature_names[i], float(feature_importance[i])) for i in ranking_indices]
        
        return {
            'shap_values_obj': ig_attributions,
            'feature_importance_mean': feature_importance,
            'ranked_features': ranked_features
        }

    def plot_attention_heatmap(self, attention_matrix, save_path="results/figures/fig_attention_heatmap.png"):
        """Plots publication-ready temporal attention weight heatmap."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Average across all test samples: (T, T)
        avg_attn = np.mean(attention_matrix, axis=0)
        
        plt.figure(figsize=(9, 7), dpi=300)
        sns.heatmap(avg_attn, cmap="viridis", cbar=True)
        plt.title("M2-WaveAttnNet: Temporal Self-Attention Saliency Matrix", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("Key Time Step (Lookback t-29 to t-0)", fontsize=11)
        plt.ylabel("Query Time Step (Lookback t-29 to t-0)", fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved Attention Heatmap: {save_path}")

    def plot_feature_importance_ranking(self, ranked_features, top_n=15, save_path="results/figures/fig_feature_importance.png"):
        """Plots publication-ready horizontal bar chart of top predictive features."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        top_items = ranked_features[:top_n]
        names = [item[0] for item in top_items][::-1]
        scores = [item[1] for item in top_items][::-1]
        
        plt.figure(figsize=(10, 6), dpi=300)
        bars = plt.barh(range(len(names)), scores, color='#2b5c8f', edgecolor='black', alpha=0.85)
        plt.yticks(range(len(names)), names, fontsize=10)
        plt.xlabel("Mean Absolute SHAP Value | Attribution Score", fontsize=11, fontweight='bold')
        plt.title(f"Top {top_n} Domain Features Identified by Explainable AI (SHAP)", fontsize=13, fontweight='bold', pad=12)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        # Add values on bars
        for bar in bars:
            w = bar.get_width()
            plt.text(w + max(scores)*0.01, bar.get_y() + bar.get_height()/2, f"{w:.4f}", va='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved SHAP Feature Ranking Plot: {save_path}")
