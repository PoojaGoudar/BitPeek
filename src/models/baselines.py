import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. Vanilla LSTM Model
# ==========================================
class VanillaLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Vanilla LSTM"
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        feat = self.fc_shared(out[:, -1, :])
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 2. Bi-LSTM with Attention Mechanism
# ==========================================
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoder_outputs):
        scores = self.v(torch.tanh(self.W(encoder_outputs))) # (B, T, 1)
        weights = F.softmax(scores, dim=1) # (B, T, 1)
        context = torch.sum(weights * encoder_outputs, dim=1) # (B, H)
        return context, weights

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Bi-LSTM + Attention"
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim // 2, num_layers=num_layers, 
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attn = BahdanauAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.bilstm(x) # (B, T, H)
        context, attn_weights = self.attn(out) # (B, H), (B, T, 1)
        feat = self.fc(context)
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir, attn_weights

# ==========================================
# 3. Gated Recurrent Unit (GRU)
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Gated Recurrent Unit (GRU)"
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        feat = self.layer_norm(out[:, -1, :])
        feat = self.fc_shared(feat)
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 4. Temporal Convolutional Network (TCN)
# ==========================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, input_dim: int, num_channels=[32, 64, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        self.name = "Temporal Convolutional Network (TCN)"
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size, dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        # x is (B, T, D) -> transpose to (B, D, T)
        x_t = x.transpose(1, 2)
        y = self.network(x_t)
        feat = self.fc(y[:, :, -1])
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 5. Time-Series Transformer
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.name = "Vanilla Transformer"
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        emb = self.pos_encoder(self.input_proj(x))
        out = self.transformer_encoder(emb)
        feat = self.fc(out.mean(dim=1)) # Global temporal pooling
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 6. PatchTST (Patch Time Series Transformer)
# ==========================================
class PatchTST(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 30, patch_len: int = 6, stride: int = 3, d_model: int = 64, nhead: int = 4, dropout: float = 0.2):
        super().__init__()
        self.name = "PatchTST"
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        self.patch_proj = nn.Linear(patch_len * input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Sequential(
            nn.Linear(self.num_patches * d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        # x is (B, T, D)
        B, T, D = x.shape
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            p = x[:, start : start + self.patch_len, :].reshape(B, -1)
            patches.append(p)
        patches = torch.stack(patches, dim=1) # (B, num_patches, patch_len*D)
        
        emb = self.pos_encoder(self.patch_proj(patches))
        out = self.transformer(emb)
        feat = self.head(out.reshape(B, -1))
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 7. N-BEATS (Neural Basis Expansion)
# ==========================================
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, theta_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim)
        )

    def forward(self, x):
        return self.fc(x)

class NBeatsModel(nn.Module):
    def __init__(self, input_dim: int, seq_len: int = 30, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.name = "N-BEATS"
        self.flatten_dim = seq_len * input_dim
        self.blocks = nn.ModuleList([
            NBeatsBlock(self.flatten_dim, hidden_dim, theta_dim=64, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.fc_combine = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        residuals = x_flat
        forecast_thetas = 0
        for block in self.blocks:
            theta = block(residuals)
            forecast_thetas = forecast_thetas + theta
            
        feat = self.fc_combine(forecast_thetas)
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir

# ==========================================
# 8. CNN-BiLSTM Hybrid (WaveNet / ConvLSTM)
# ==========================================
class CNNBiLSTMHybrid(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name = "CNN-BiLSTM Hybrid"
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bilstm = nn.LSTM(64, hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.head_return = nn.Linear(32, 1)
        self.head_dir = nn.Linear(32, 1)

    def forward(self, x):
        # x is (B, T, D) -> (B, D, T)
        x_t = x.transpose(1, 2)
        c1 = F.relu(self.conv1(x_t))
        c2 = F.relu(self.conv2(c1))
        # Back to (B, T, C)
        lstm_in = c2.transpose(1, 2)
        lstm_out, _ = self.bilstm(lstm_in)
        feat = self.fc(lstm_out[:, -1, :])
        pred_ret = self.head_return(feat)
        pred_dir = torch.sigmoid(self.head_dir(feat))
        return pred_ret, pred_dir
