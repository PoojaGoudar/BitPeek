import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.pipeline import CryptoDataPipeline
from src.evaluation.metrics import compute_all_metrics

# Load pipeline
pipeline = CryptoDataPipeline(ticker="BTC-USD", start_date="2018-01-01", end_date="2025-01-01", lookback=30)
data = pipeline.prepare_datasets(train_ratio=0.70, val_ratio=0.15, batch_size=32)

print("Features:", data['input_dim'], "Window:", data['seq_len'])
print("Train batches:", len(data['train_loader']), "Val batches:", len(data['val_loader']), "Test batches:", len(data['test_loader']))
