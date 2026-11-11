import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .proposed_sota import (
    CausalConv1d,
    MultiScaleCausalInception,
    ChannelFeatureGating,
    CrossAttentionTemporalFusion,
    PositionalEncoding
)

# =========================================================================
# 1. Constituent Layer 1: Standalone Feature Channel Gating Network
# =========================================================================
class StandaloneChannelGatingNet(nn.Module):
    """
    Constituent Layer 1 (Standalone):
    Focuses exclusively on adaptive feature importance and channel recalibration.
    Evaluates how effectively feature selection alone predicts returns without
    multi-scale temporal convolutions or attention.
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "Constituent 1: Channel Gating Only"
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        self.channel_gate = ChannelFeatureGating(input_dim, reduction=4)
        self.fc_pool = nn.AdaptiveAvgPool1d(1)
        
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.GELU()
        )
        self.head_return = nn.Linear(32, 1)
        self.head_direction = nn.Linear(32, 1)

    def forward(self, x):
        # x is (B, T, D)
        x_gated, gate_weights = self.channel_gate(x)
        pooled = self.fc_pool(x_gated.transpose(1, 2)).squeeze(-1) # (B, D)
        latent = self.trunk(pooled)
        pred_return = self.head_return(latent)
        pred_dir = torch.sigmoid(self.head_direction(latent))
        return pred_return, pred_dir


# =========================================================================
# 2. Constituent Layer 2: Standalone Multi-Scale Inception Causal ConvNet
# =========================================================================
class StandaloneInceptionConvNet(nn.Module):
    """
    Constituent Layer 2 (Standalone):
    Focuses exclusively on multi-scale causal convolutions (1d, 3d, 5d, 7d).
    Evaluates temporal pattern extraction across micro, short, and medium horizons
    without channel gating or self-attention.
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "Constituent 2: Multi-Scale Inception Conv Only"
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.inception1 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        self.inception2 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_direction = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x_in = self.stem(x.transpose(1, 2))
        inc1 = self.inception1(x_in)
        inc2 = self.inception2(inc1) + inc1 # Residual Inception
        
        pooled = self.pool(inc2).squeeze(-1) # (B, H)
        latent = self.trunk(pooled)
        pred_return = self.head_return(latent)
        pred_dir = torch.sigmoid(self.head_direction(latent))
        return pred_return, pred_dir


# =========================================================================
# 3. Constituent Layer 3: Standalone Temporal Cross-Attention Network
# =========================================================================
class StandaloneTemporalAttnNet(nn.Module):
    """
    Constituent Layer 3 (Standalone):
    Focuses exclusively on multi-head scaled dot-product temporal self-attention.
    Evaluates pure inter-timestep dependency scoring without multi-scale convolutions
    or channel gating.
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "Constituent 3: Temporal Attention Only"
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 10)
        self.attn = CrossAttentionTemporalFusion(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_direction = nn.Linear(32, 1)

    def forward(self, x):
        # x is (B, T, D)
        proj = self.pos_encoder(self.proj(x))
        attn_out, _ = self.attn(proj)
        
        last_step = attn_out[:, -1, :] # (B, H)
        latent = self.trunk(last_step)
        pred_return = self.head_return(latent)
        pred_dir = torch.sigmoid(self.head_direction(latent))
        return pred_return, pred_dir


# =========================================================================
# 4. Constituent Layer 4: Standalone Deep Residual Multi-Task Trunk
# =========================================================================
class StandaloneResidualTrunkNet(nn.Module):
    """
    Constituent Layer 4 (Standalone):
    Focuses exclusively on a deep multi-layer residual dense representation trunk.
    Evaluates non-linear multi-task mapping from flattened historical representations
    without specialized gating, convolutions, or attention.
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "Constituent 4: Residual Trunk Only"
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        flat_dim = input_dim * seq_len
        self.in_proj = nn.Linear(flat_dim, hidden_dim)
        
        # 3 Residual Dense Blocks
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_norm = nn.LayerNorm(hidden_dim)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_norm = nn.LayerNorm(hidden_dim)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        self.head_return = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.head_direction = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x is (B, T, D) -> Flatten to (B, T*D)
        B = x.shape[0]
        h = self.act(self.in_proj(x.reshape(B, -1)))
        
        # Res block 1
        res1 = self.act(self.res1_fc2(self.dropout(self.act(self.res1_norm(self.res1_fc1(h))))))
        h = h + res1
        
        # Res block 2
        res2 = self.act(self.res2_fc2(self.dropout(self.act(self.res2_norm(self.res2_fc1(h))))))
        h = h + res2
        
        pred_return = self.head_return(h)
        pred_dir = torch.sigmoid(self.head_direction(h))
        return pred_return, pred_dir
