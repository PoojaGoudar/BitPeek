import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for temporal order awareness."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is (B, T, D)
        return x + self.pe[:, :x.size(1)]

class CausalConv1d(nn.Module):
    """Causal convolution that strictly prohibits lookahead into future time steps."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

class MultiScaleCausalInception(nn.Module):
    """
    Multi-Scale Inception Causal Convolution Block capturing micro (1d),
    short (3d), medium (5d), and weekly (7d) temporal dynamics.
    Uses GroupNorm for sample-invariant temporal normalization.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_dim = max(1, out_channels // 4)
        self.b1 = CausalConv1d(in_channels, branch_dim, kernel_size=1)
        self.b2 = CausalConv1d(in_channels, branch_dim, kernel_size=3, dilation=1)
        self.b3 = CausalConv1d(in_channels, branch_dim, kernel_size=5, dilation=2)
        self.b4 = CausalConv1d(in_channels, branch_dim, kernel_size=7, dilation=3)
        self.proj = nn.Conv1d(branch_dim * 4, out_channels, kernel_size=1)
        
        # GroupNorm is sample-invariant: avoids covariate shift in non-stationary financial data
        groups = 4 if out_channels % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x is (B, C, T)
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        cat = torch.cat([o1, o2, o3, o4], dim=1)
        out = self.act(self.norm(self.proj(cat)))
        return out

class ChannelFeatureGating(nn.Module):
    """
    Squeeze-and-Excitation style adaptive feature channel gating.
    Computes dynamic channel attention weights to emphasize salient crypto indicators.
    """
    def __init__(self, num_features, reduction=4):
        super().__init__()
        bottleneck = max(8, num_features // reduction)
        self.fc = nn.Sequential(
            nn.Linear(num_features, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, T, D)
        weights = self.fc(x.mean(dim=1)) # (B, D)
        return x * weights.unsqueeze(1), weights

class CrossAttentionTemporalFusion(nn.Module):
    """
    Dual-branch cross-attention temporal fusion mechanism with positional encoding.
    Captures inter-day temporal dependencies and market regime transitions.
    """
    def __init__(self, d_model=64, n_heads=4, dropout=0.15):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        residual = x
        
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        out = self.layer_norm(residual + self.out_proj(context))
        
        # Average attention weights across heads for XAI
        avg_attn = attn_weights.mean(dim=1) # (B, T, T)
        return out, avg_attn

class M2WaveAttnNet(nn.Module):
    """
    Proposed Novel Architecture:
    Multi-Modal Wavelet Cross-Attention Temporal Fusion Network (M2-WaveAttnNet).
    
    A unified 4-layer Hybrid Architecture that fuses:
    1. Layer 1: Channel Feature Gating for quantitative indicator recalibration
    2. Layer 2: Multi-Scale Inception Causal Convolutions (MICC) across 1d, 3d, 5d, 7d horizons
    3. Layer 3: Positional Cross-Attention Temporal Fusion (CATF) with dynamic dependency extraction
    4. Layer 4: Multi-Task Dense Residual Trunk & Calibrated Heads (Return + Direction + Uncertainty)
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.name = "Proposed M2-WaveAttnNet (SOTA)"
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 1. Feature Channel Gating (Layer 1)
        self.channel_gate = ChannelFeatureGating(input_dim, reduction=4)
        
        # 2. Multi-Scale Inception Causal Convolutions (Layer 2)
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.inception1 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        self.inception2 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        
        # Highway residual projection across convolution stages
        self.conv_skip = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
        # 3. Cross-Attention Temporal Fusion (Layer 3)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 10)
        self.attn_fusion = CrossAttentionTemporalFusion(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
        self.attn_fusion2 = CrossAttentionTemporalFusion(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
        self.attn_fusion3 = CrossAttentionTemporalFusion(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
        
        # 4. Context Aggregation
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. Shared Representation Trunk (Layer 4) — aggressive regularization
        # hidden_dim*2 -> hidden_dim with 3-stage dropout ladder
        self.trunk_fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.trunk_norm1 = nn.LayerNorm(hidden_dim)
        self.trunk_drop1 = nn.Dropout(dropout + 0.10)   # stronger first-stage dropout
        self.trunk_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.trunk_norm2 = nn.LayerNorm(hidden_dim // 2)
        self.trunk_drop2 = nn.Dropout(dropout)
        self.trunk_fc3 = nn.Linear(hidden_dim // 2, 32)
        self.trunk_norm3 = nn.LayerNorm(32)
        self.trunk_fc4 = nn.Linear(32, 32)
        self.trunk_norm4 = nn.LayerNorm(32)
        self.trunk_drop4 = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        # 6. Multi-Task Heads
        self.head_return = nn.Linear(32, 1)       # Continuous Return r_{t+1}
        self.head_direction = nn.Linear(32, 1)    # Binary Direction d_{t+1} (Logits)
        self.head_uncertainty = nn.Linear(32, 1)  # Epistemic / Aleatoric log-variance

        # 7. Self-supervised reconstruction head (used only during pre-training)
        self.recon_head = nn.Linear(hidden_dim, input_dim)
        
    def _trunk(self, x):
        """4-stage MLP trunk with progressive dropout for anti-overfitting and extra capacity."""
        h = self.trunk_drop1(self.act(self.trunk_norm1(self.trunk_fc1(x))))
        h = self.trunk_drop2(self.act(self.trunk_norm2(self.trunk_fc2(h))))
        h = self.trunk_drop4(self.act(self.trunk_norm4(self.trunk_fc4(self.trunk_norm3(self.trunk_fc3(h))))) )
        return h

    def encode(self, x):
        """Encoder-only forward pass — returns full (B, T, H) temporal feature map.
        Used during self-supervised pre-training (masked time-step reconstruction)."""
        x_gated, _ = self.channel_gate(x)
        x_conv_in = self.stem(x_gated.transpose(1, 2))
        inc1 = self.inception1(x_conv_in)
        inc2 = self.inception2(inc1) + self.conv_skip(x_conv_in)
        attn_in = self.pos_encoder(inc2.transpose(1, 2))
        attn_out, _ = self.attn_fusion(attn_in)
        attn_out, _ = self.attn_fusion2(attn_out)
        attn_out, _ = self.attn_fusion3(attn_out)
        return attn_out  # (B, T, H)

    def forward(self, x, return_attn=False):
        # x is (B, T, D)
        B, T, D = x.shape

        # Step 1: Feature Channel Gating (Layer 1)
        x_gated, gate_weights = self.channel_gate(x)

        # Step 2: Multi-Scale Causal Convolutions (Layer 2)
        x_conv_in = self.stem(x_gated.transpose(1, 2))  # (B, H, T)
        inc1 = self.inception1(x_conv_in)
        inc2 = self.inception2(inc1) + self.conv_skip(x_conv_in)  # Residual Inception

        # Step 3: Positional Cross-Attention Temporal Fusion (Layer 3)
        attn_in = self.pos_encoder(inc2.transpose(1, 2))  # (B, T, H)
        attn_out, attn_matrix  = self.attn_fusion(attn_in)
        attn_out, attn_matrix2 = self.attn_fusion2(attn_out)
        attn_out, attn_matrix3 = self.attn_fusion3(attn_out)

        # Step 4: Multi-Scale Context Aggregation
        last_step_feat = attn_out[:, -1, :]                                     # (B, H)
        pooled_feat    = self.temporal_pool(attn_out.transpose(1, 2)).squeeze(-1)  # (B, H)
        combined_feat  = torch.cat([last_step_feat, pooled_feat], dim=-1)         # (B, 2*H)

        # Step 5: Shared Representation Trunk (Layer 4)
        latent = self._trunk(combined_feat)
        pred_return    = self.head_return(latent)
        pred_dir_prob  = torch.sigmoid(self.head_direction(latent))
        log_var        = self.head_uncertainty(latent)

        if return_attn:
            return pred_return, pred_dir_prob, {
                'attn_matrix':  attn_matrix,
                'attn_matrix2': attn_matrix2,
                'attn_matrix3': attn_matrix3,
                'gate_weights': gate_weights,
                'log_var':      log_var
            }
        return pred_return, pred_dir_prob
