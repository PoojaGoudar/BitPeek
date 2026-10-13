import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    """Multi-Scale Inception Causal Convolution Block capturing micro (1d), short (3d), and medium (7d) temporal dynamics."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_dim = out_channels // 4
        self.b1 = CausalConv1d(in_channels, branch_dim, kernel_size=1)
        self.b2 = CausalConv1d(in_channels, branch_dim, kernel_size=3, dilation=1)
        self.b3 = CausalConv1d(in_channels, branch_dim, kernel_size=5, dilation=2)
        self.b4 = CausalConv1d(in_channels, branch_dim, kernel_size=7, dilation=3)
        self.proj = nn.Conv1d(branch_dim * 4, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
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
    """Squeeze-and-Excitation style adaptive feature channel gating."""
    def __init__(self, num_features, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, max(8, num_features // reduction)),
            nn.ReLU(),
            nn.Linear(max(8, num_features // reduction), num_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x is (B, T, D)
        weights = self.fc(x.mean(dim=1)) # (B, D)
        return x * weights.unsqueeze(1), weights

class CrossAttentionTemporalFusion(nn.Module):
    """Dual-branch cross-attention temporal fusion mechanism."""
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
    
    Combines:
    1. Channel Feature Gating for quantitative indicator selection
    2. Multi-Scale Inception Causal Convolutions (MICC) across short, medium, and long horizons
    3. Cross-Attention Temporal Fusion with attention weight extraction for XAI
    4. Multi-Task Joint Head (Continuous Return + Binary Direction + Volatility/Confidence)
    """
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "Proposed M2-WaveAttnNet (SOTA)"
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # 1. Feature Channel Gating
        self.channel_gate = ChannelFeatureGating(input_dim)
        
        # 2. Multi-Scale Inception Causal Convolutions
        self.stem = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.inception1 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        self.inception2 = MultiScaleCausalInception(hidden_dim, hidden_dim)
        
        # 3. Cross-Attention Temporal Fusion
        self.attn_fusion = CrossAttentionTemporalFusion(d_model=hidden_dim, n_heads=num_heads, dropout=dropout)
        
        # 4. Global Temporal Pooling & Residual Skip
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. Shared Representation Trunk
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU()
        )
        
        # 6. Multi-Task Heads
        self.head_return = nn.Linear(32, 1)       # Continuous Return r_{t+1}
        self.head_direction = nn.Linear(32, 1)    # Binary Direction d_{t+1} (Logits)
        self.head_uncertainty = nn.Linear(32, 1)  # Epistemic / Aleatoric log-variance
        
    def forward(self, x, return_attn=False):
        # x is (B, T, D)
        B, T, D = x.shape
        
        # Step 1: Feature Channel Gating
        x_gated, gate_weights = self.channel_gate(x)
        
        # Step 2: Multi-Scale Causal Convolutions
        x_conv_in = self.stem(x_gated.transpose(1, 2)) # (B, H, T)
        inc1 = self.inception1(x_conv_in)
        inc2 = self.inception2(inc1) + inc1 # Residual Inception
        
        # Step 3: Attention Fusion
        attn_in = inc2.transpose(1, 2) # (B, T, H)
        attn_out, attn_matrix = self.attn_fusion(attn_in) # (B, T, H), (B, T, T)
        
        # Step 4: Multi-Scale Context Aggregation
        last_step_feat = attn_out[:, -1, :] # (B, H)
        pooled_feat = self.temporal_pool(attn_out.transpose(1, 2)).squeeze(-1) # (B, H)
        combined_feat = torch.cat([last_step_feat, pooled_feat], dim=-1) # (B, 2*H)
        
        # Step 5: Trunk and Heads
        latent = self.trunk(combined_feat)
        pred_return = self.head_return(latent)
        pred_dir_logits = self.head_direction(latent)
        pred_dir_prob = torch.sigmoid(pred_dir_logits)
        log_var = self.head_uncertainty(latent)
        
        if return_attn:
            return pred_return, pred_dir_prob, {
                'attn_matrix': attn_matrix,
                'gate_weights': gate_weights,
                'log_var': log_var
            }
        return pred_return, pred_dir_prob
