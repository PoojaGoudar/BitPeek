import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .losses import DirectionCalibratedLoss

class ModelTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, max_epochs=40, patience=10, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.criterion = DirectionCalibratedLoss()
        self.best_state = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_da': [], 'val_mae': []}

    def train(self, train_loader, val_loader, verbose=False):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=1e-5)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            # Training Phase
            self.model.train()
            train_losses = []
            for batch in train_loader:
                x = batch['x'].to(self.device)
                y_ret = batch['y_return'].to(self.device)
                y_dir = batch['y_dir'].to(self.device)

                optimizer.zero_grad()
                pred_ret, pred_dir = self.model(x)[:2]
                loss, _ = self.criterion(pred_ret, y_ret, pred_dir, y_dir)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = np.mean(train_losses)

            # Validation Phase
            val_loss, val_metrics = self.evaluate(val_loader)
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_da'].append(val_metrics['DA'])
            self.history['val_mae'].append(val_metrics['MAE'])

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"Epoch [{epoch:02d}/{self.max_epochs:02d}] - Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f} | Val DA: {val_metrics['DA']*100:.2f}% | Val MAE: {val_metrics['MAE']:.5f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"[*] Early stopping triggered at epoch {epoch}.")
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.history

    def evaluate(self, loader):
        self.model.eval()
        total_losses = []
        all_pred_rets, all_true_rets = [], []
        all_pred_dirs, all_true_dirs = [], []

        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(self.device)
                y_ret = batch['y_return'].to(self.device)
                y_dir = batch['y_dir'].to(self.device)

                pred_ret, pred_dir = self.model(x)[:2]
                loss, _ = self.criterion(pred_ret, y_ret, pred_dir, y_dir)
                total_losses.append(loss.item())

                all_pred_rets.append(pred_ret.cpu().numpy())
                all_true_rets.append(y_ret.cpu().numpy())
                all_pred_dirs.append(pred_dir.cpu().numpy())
                all_true_dirs.append(y_dir.cpu().numpy())

        all_pred_rets = np.concatenate(all_pred_rets, axis=0).flatten()
        all_true_rets = np.concatenate(all_true_rets, axis=0).flatten()
        all_pred_dirs = (np.concatenate(all_pred_dirs, axis=0).flatten() >= 0.5).astype(int)
        all_true_dirs = np.concatenate(all_true_dirs, axis=0).flatten().astype(int)

        mae = np.mean(np.abs(all_pred_rets - all_true_rets))
        da = np.mean(all_pred_dirs == all_true_dirs)
        
        return np.mean(total_losses), {'MAE': mae, 'DA': da}

    def predict(self, loader):
        self.model.eval()
        pred_rets, pred_dirs, true_rets, true_dirs, true_closes = [], [], [], [], []
        
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(self.device)
                pred_ret, pred_dir = self.model(x)[:2]
                
                pred_rets.append(pred_ret.cpu().numpy())
                pred_dirs.append(pred_dir.cpu().numpy())
                true_rets.append(batch['y_return'].numpy())
                true_dirs.append(batch['y_dir'].numpy())
                true_closes.append(batch['y_close'].numpy())

        return {
            'pred_returns': np.concatenate(pred_rets, axis=0).flatten(),
            'pred_directions': np.concatenate(pred_dirs, axis=0).flatten(),
            'true_returns': np.concatenate(true_rets, axis=0).flatten(),
            'true_directions': np.concatenate(true_dirs, axis=0).flatten(),
            'true_closes': np.concatenate(true_closes, axis=0).flatten()
        }
