import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionCalibratedLoss(nn.Module):
    """
    Direction-Calibrated Multi-Task Loss (DCMT-Loss) for Financial Time Series:
    Combines:
    1. Calibrated Huber Loss (Scaled robust regression against crypto tail-risk jumps)
    2. Binary Cross Entropy (Directional classification)
    3. Asymmetric Direction Penalty (ADP) that severely penalizes wrong sign predictions
    4. Sign-Probability Consistency Regularization ensuring the regression and classification heads agree
    """
    def __init__(self, alpha: float = 50.0, beta: float = 1.0, gamma: float = 15.0,
                 delta: float = 0.02, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha  # Regression weight (scaled to balance Huber loss ~0.0002 with BCE ~0.69)
        self.beta = beta    # Classification weight
        self.gamma = gamma  # Asymmetric direction penalty weight
        self.label_smoothing = label_smoothing  # Prevents overconfident direction predictions
        self.huber = nn.HuberLoss(delta=delta)
        self.bce = nn.BCELoss()

    def forward(self, pred_ret, target_ret, pred_dir, target_dir):
        # 1. Scaled Robust Regression Loss
        loss_reg = self.huber(pred_ret, target_ret)

        # 2. Direction BCE with Label Smoothing: smoothed_target = (1-ε)*target + ε*0.5
        eps = 1e-7
        pred_dir_clamped = torch.clamp(pred_dir, eps, 1.0 - eps)
        smoothed_target = (1.0 - self.label_smoothing) * target_dir + self.label_smoothing * 0.5
        loss_cls = self.bce(pred_dir_clamped, smoothed_target)

        # 3. Asymmetric Direction Penalty (ADP)
        sign_mismatch = F.relu(-1.0 * pred_ret * target_ret)
        loss_penalty = torch.mean(sign_mismatch ** 2)

        # 4. Consistency alignment
        inconsistency = F.relu(-1.0 * (pred_dir - 0.5) * pred_ret * 50.0)
        loss_align = torch.mean(inconsistency)

        total_loss = self.alpha * loss_reg + self.beta * loss_cls + self.gamma * loss_penalty + 0.5 * loss_align
        return total_loss, {
            'loss_reg': (self.alpha * loss_reg).item(),
            'loss_cls': loss_cls.item(),
            'loss_penalty': (self.gamma * loss_penalty).item(),
            'loss_align': loss_align.item()
        }


class FocalDirectionLoss(nn.Module):
    """
    Focal Direction Multi-Task Loss — drop-in replacement for DirectionCalibratedLoss.

    Uses Focal Loss for the direction head: down-weights easy examples so the model
    concentrates gradient on hard direction-flip cases.

    Focal Loss:  FL(p_t) = -alpha_t * (1 - p_t)^focal_gamma * log(p_t)
    """
    def __init__(self,
                 alpha: float = 50.0,        # regression weight
                 beta: float = 2.0,          # direction loss weight (emphasise direction signal)
                 gamma: float = 10.0,        # ADP sign-penalty weight
                 focal_gamma: float = 2.0,   # focal exponent (standard = 2)
                 focal_alpha: float = 0.6,   # UP-class prior (crypto trends slightly up)
                 delta: float = 0.02,        # Huber delta
                 label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.huber = nn.HuberLoss(delta=delta)

    def _focal_bce(self, pred_dir: torch.Tensor, target_dir: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        p = torch.clamp(pred_dir, eps, 1.0 - eps)
        t = (1.0 - self.label_smoothing) * target_dir + self.label_smoothing * 0.5
        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
        p_t = p * target_dir + (1.0 - p) * (1.0 - target_dir)
        focal_weight = (1.0 - p_t) ** self.focal_gamma
        alpha_t = self.focal_alpha * target_dir + (1.0 - self.focal_alpha) * (1.0 - target_dir)
        return (alpha_t * focal_weight * bce).mean()

    def forward(self, pred_ret, target_ret, pred_dir, target_dir):
        loss_reg = self.huber(pred_ret, target_ret)
        loss_cls = self._focal_bce(pred_dir, target_dir)
        sign_mismatch = F.relu(-1.0 * pred_ret * target_ret)
        loss_penalty = torch.mean(sign_mismatch ** 2)
        inconsistency = F.relu(-1.0 * (pred_dir - 0.5) * pred_ret * 50.0)
        loss_align = torch.mean(inconsistency)

        total_loss = (self.alpha * loss_reg
                      + self.beta * loss_cls
                      + self.gamma * loss_penalty
                      + 0.5 * loss_align)
        return total_loss, {
            'loss_reg':     (self.alpha * loss_reg).item(),
            'loss_focal':   loss_cls.item(),
            'loss_penalty': (self.gamma * loss_penalty).item(),
            'loss_align':   loss_align.item()
        }


class MaskedReconstructionLoss(nn.Module):
    """MSE reconstruction loss for self-supervised pre-training (masked time-step prediction)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        diff = (pred - target) ** 2
        loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        return loss
