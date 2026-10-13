import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionCalibratedLoss(nn.Module):
    """
    Direction-Calibrated Multi-Task Loss (DCMT-Loss) for Financial Time Series:
    Combines:
    1. Smooth L1 / Huber Loss (Robust Regression against crypto tail-risk jumps)
    2. Binary Cross Entropy / Focal Loss (Directional accuracy)
    3. Asymmetric Direction Penalty (ADP) that severely penalizes wrong sign predictions
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 1.2, delta: float = 0.01):
        super().__init__()
        self.alpha = alpha  # Regression weight
        self.beta = beta    # Classification weight
        self.gamma = gamma  # Asymmetric direction penalty weight
        self.huber = nn.HuberLoss(delta=delta)
        self.bce = nn.BCELoss()

    def forward(self, pred_ret, target_ret, pred_dir, target_dir):
        # 1. Robust Regression Loss
        loss_reg = self.huber(pred_ret, target_ret)
        
        # 2. Direction Binary Cross Entropy
        eps = 1e-7
        pred_dir_clamped = torch.clamp(pred_dir, eps, 1.0 - eps)
        loss_cls = self.bce(pred_dir_clamped, target_dir)
        
        # 3. Asymmetric Direction Penalty (ADP): Penalty = max(0, -pred_ret * target_ret)
        sign_mismatch = F.relu(-1.0 * pred_ret * target_ret)
        loss_penalty = torch.mean(sign_mismatch ** 2)
        
        total_loss = self.alpha * loss_reg + self.beta * loss_cls + self.gamma * loss_penalty
        return total_loss, {
            'loss_reg': loss_reg.item(),
            'loss_cls': loss_cls.item(),
            'loss_penalty': loss_penalty.item()
        }
