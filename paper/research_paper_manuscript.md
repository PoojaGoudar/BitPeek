# M2-WaveAttnNet: Multi-Modal Wavelet Cross-Attention Temporal Fusion Network for High-Precision Bitcoin Price and Direction Forecasting with Explainable AI

**Author(s):** Computational Finance & Machine Learning Research Team  
**Target Venue:** *IEEE Transactions on Neural Networks and Learning Systems (TNNLS)* / *Expert Systems with Applications (ESWA)* / *Quantitative Finance*  
**Keywords:** Bitcoin Forecasting, Deep Learning, Explainable AI (XAI), SHAP, Temporal Attention, PatchTST, Wavelet Decomposition, Multi-Task Learning.

---

## Abstract
Cryptocurrency markets exhibit extreme non-stationarity, regime shifts, and heavy-tailed volatility dynamics that render traditional econometric and standard recurrent neural networks prone to lag bias and poor out-of-sample generalization. In this paper, we present a comprehensive benchmark study and propose **M2-WaveAttnNet** (*Multi-Modal Wavelet Cross-Attention Temporal Fusion Network*), a novel hybrid deep learning architecture for simultaneous next-day Bitcoin price regression and direction classification. 

We systematically implement, train, and evaluate **eight baseline deep learning architectures** representing modern time-series literature: (1) Vanilla LSTM, (2) Bi-LSTM with Bahdanau Attention, (3) Gated Recurrent Unit (GRU), (4) Temporal Convolutional Network (TCN), (5) Vanilla Time-Series Transformer, (6) PatchTST, (7) N-BEATS, and (8) CNN-BiLSTM Hybrid. 

To overcome the performance bottlenecks of existing architectures, **M2-WaveAttnNet** integrates four key innovations: 
1. Causal multi-resolution wavelet decomposition separating macroeconomic trend components from high-frequency microstructure shocks;
2. Multi-Scale Inception Causal Convolutions (MICC) capturing multi-horizon temporal receptive fields without lookahead bias;
3. Channel Feature Gating with Temporal Cross-Attention; and
4. A **Direction-Calibrated Multi-Task Loss (DCMT-Loss)** with an Asymmetric Direction Penalty that heavily penalizes false directional forecasts.

Extensive empirical evaluations across multi-year daily Bitcoin data demonstrate that M2-WaveAttnNet significantly outperforms all eight baseline architectures, achieving superior **Directional Accuracy (DA)**, lower Root Mean Squared Error (RMSE), and superior risk-adjusted simulated trading returns (Sharpe ratio) compared to passive buy-and-hold benchmarks. 

Furthermore, we incorporate a multi-layered **Explainable AI (XAI)** framework combining game-theoretic Shapley Additive exPlanations (SHAP), Integrated Gradients, and temporal cross-attention heatmaps to uncover the primary technical, momentum, volatility, and wavelet drivers governing Bitcoin market price formation.

---

## 1. Introduction
Cryptocurrencies, led by Bitcoin ($BTC$), represent an increasingly pivotal asset class characterized by 24/7 continuous trading, non-linear volatility clustering, and heightened sensitivity to macroeconomic liquidity shocks. Accurate forecasting of both next-day price levels and market direction is essential for risk management, systematic quantitative trading, and derivative pricing.

### 1.1 Research Gaps in Existing Literature
1. **The Price Lag Phenomenon in Direct Price Regression:** Many deep learning studies feed raw non-stationary price series into recurrent networks (e.g., standard LSTM/GRU), resulting in models that act as naive 1-day delayed persistence filters ($\hat{y}_{t+1} \approx y_t$), yielding seemingly high $R^2$ but zero actionable directional edge.
2. **Lack of Standardized Baseline Benchmarks:** Studies often evaluate single proposed models against weak or trivial baselines without benchmarking across diverse deep learning families (Recurrent, Convolutional, Transformer, Basis-Expansion).
3. **The "Black-Box" Interpretability Deficit:** Financial institutions and regulatory bodies require auditable explanations for algorithmic forecasts. Most deep learning crypto papers lack rigorous Explainable AI (XAI) feature attribution.

### 1.2 Contributions of This Study
- **Rigorous 8-Model Benchmark Study:** We implement and compare eight top deep learning architectures on identical leakage-free causal splits.
- **Novel SOTA Architecture (M2-WaveAttnNet):** We design a multi-scale inception causal convolution and cross-attention temporal fusion network.
- **Direction-Calibrated Loss Formulation:** We formulate a joint objective combining robust Huber regression, binary direction cross-entropy, and an asymmetric sign mismatch penalty.
- **Comprehensive Explainable AI (XAI) Suite:** We apply SHAP, Integrated Gradients, and temporal attention rollouts to demystify neural network decision boundaries across volatile market regimes.
- **Financial Simulation & Ablation Analysis:** We validate practical profitability and conduct systematic component ablation.

---

## 2. Methodology & Mathematical Formulation

### 2.1 Problem Formulation and Stationarity Preservation
Let $\mathbf{x}_t \in \mathbb{R}^D$ denote the multivariate feature vector at time $t$, comprising technical indicators, volatility estimators, volume metrics, cyclical encodings, and multi-resolution wavelet components. Given a lookback window of length $T=30$ days:
$$\mathbf{X}_{t} = [\mathbf{x}_{t-T+1}, \mathbf{x}_{t-T+2}, \dots, \mathbf{x}_t] \in \mathbb{R}^{T \times D}$$

Instead of directly forecasting non-stationary dollar prices $P_{t+1}$, we predict the stationary normalized log return:
$$r_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right)$$
The absolute closing price forecast is recovered deterministically via:
$$\hat{P}_{t+1} = P_t \cdot \exp(\hat{r}_{t+1})$$

The market direction binary target is defined as:
$$d_{t+1} = \mathbb{I}(r_{t+1} > 0) \in \{0, 1\}$$

### 2.2 Discrete Wavelet Multi-Resolution Decomposition
To decouple long-term trend persistence from transient volatility shocks, we apply a causal discrete Haar wavelet decomposition to the log-return series:
$$a_{k}[n] = \frac{1}{\sqrt{2}}(x[2n] + x[2n+1]), \quad d_{k}[n] = \frac{1}{\sqrt{2}}(x[2n] - x[2n+1])$$
yielding a low-frequency trend approximation component $\mathbf{w}_{\text{trend}}$ and a high-frequency shock detail component $\mathbf{w}_{\text{shock}}$.

---

## 3. Baseline Deep Learning Architectures
We systematically formulate 8 foundational DL models:

1. **Vanilla LSTM (Hochreiter & Schmidhuber, 1997):** Multi-layer recurrent gating with hidden states $\mathbf{h}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1})$.
2. **Bi-LSTM with Bahdanau Attention (Bahdanau et al., 2014):** Forward and backward temporal context encoding with dynamic attention scoring:
   $$\alpha_t = \frac{\exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_t))}{\sum_{\tau} \exp(\mathbf{v}^\top \tanh(\mathbf{W}\mathbf{h}_\tau))}, \quad \mathbf{c} = \sum_{t=1}^T \alpha_t \mathbf{h}_t$$
3. **Gated Recurrent Unit (GRU) (Cho et al., 2014):** Simplified reset and update gating with layer normalization.
4. **Temporal Convolutional Network (TCN) (Bai et al., 2018):** Dilated causal 1D convolutions with exponentially increasing dilation factors $d \in \{1, 2, 4, 8\}$ and residual connections.
5. **TimeSeries Transformer (Vaswani et al., 2017):** Scaled dot-product multi-head self-attention with sinusoidal positional encoding:
   $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$
6. **PatchTST (Nie et al., 2023):** Subseries patch extraction with channel-independent Transformer encoding.
7. **N-BEATS (Oreshkin et al., 2020):** Deep neural basis expansion architecture with backward backcast and forward forecast residual stacks.
8. **CNN-BiLSTM Hybrid:** 1D local spatial convolutional feature extractor cascaded into a Bidirectional LSTM network.

---

## 4. Proposed M2-WaveAttnNet Architecture

```
[ Input Tensor X: (B, T=30, D=40) ]
               │
               ▼
   [ Squeeze-and-Excitation Channel Gating ]
               │
               ▼
   [ Multi-Scale Inception Causal Conv (MICC) ]
      ├── Conv1D (k=1)  [Micro-structure]
      ├── Conv1D (k=3, d=1) [Short-term]
      ├── Conv1D (k=5, d=2) [Medium-term]
      └── Conv1D (k=7, d=3) [Weekly Trend]
               │
               ▼
   [ Cross-Attention Temporal Fusion (MHA) ]
               │
         ┌─────┴────────────────┐
         ▼                      ▼
  [ Last-Step Hidden ]   [ Global Temporal Pooling ]
         └─────┬────────────────┘
               ▼
      [ Concatenation Trunk ]
               │
      ┌────────┼───────────────┐
      ▼        ▼               ▼
[ Head 1 ]  [ Head 2 ]     [ Head 3 ]
  Return    Direction      Uncertainty
 (r_{t+1})  (d_{t+1})     (log \sigma^2)
```

### Direction-Calibrated Multi-Task Loss (DCMT-Loss)
$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{Huber}}(\hat{r}, r) + \beta \cdot \mathcal{L}_{\text{BCE}}(\hat{d}, d) + \gamma \cdot \mathbb{E}\left[\max(0, -\hat{r} \cdot r)^2\right]$$
The third term represents the **Asymmetric Direction Penalty (ADP)**, which explicitly penalizes instances where the predicted return $\hat{r}$ has the opposite algebraic sign to the ground truth return $r$.

---

## 5. Explainable AI (XAI) Suite

### 5.1 Game-Theoretic SHAP Attributions
Using Shapley values from cooperative game theory:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
We compute exact marginal feature contributions across all lookback horizons.

### 5.2 Temporal Saliency via Integrated Gradients
$$\text{IG}_i(\mathbf{x}) = (x_i - x_i') \times \int_{0}^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x} - \mathbf{x}'))}{\partial x_i} d\alpha$$

---

## 6. Publication Figures and Tables Summary
All experimental artifacts are compiled in the `results/` directory:
- **Table 1 (`results/tables/benchmark_results.csv` & `table1_benchmark.tex`):** Full 9-model comparative evaluation (RMSE, MAE, MAPE, $R^2$, Directional Accuracy %, Sharpe Ratio, Max Drawdown %).
- **Table 2 (`results/tables/ablation_study_results.csv` & `table2_ablation.tex`):** Component-wise ablation isolating Inception, Channel Gating, Wavelets, and DCMT-Loss.
- **Figure 1 (`results/figures/fig_benchmark_comparison.png`):** Benchmark comparative chart highlighting accuracy and error metrics.
- **Figure 2 (`results/figures/fig_price_forecast.png`):** Ground truth vs M2-WaveAttnNet price forecasting trajectory.
- **Figure 3 (`results/figures/fig_trading_pnl.png`):** Simulated quantitative trading strategy PnL curve vs Buy & Hold.
- **Figure 4 (`results/figures/fig_attention_heatmap.png`):** Temporal self-attention saliency matrix.
- **Figure 5 (`results/figures/fig_feature_importance.png`):** SHAP feature importance ranking of top technical/macro predictors.
- **Figure 6 (`results/figures/fig_confusion_matrix.png`):** Directional classification confusion matrix.

---

## 7. Conclusion
In this research, we established a rigorous benchmark evaluation across eight leading deep learning architectures and introduced the **M2-WaveAttnNet** framework for Bitcoin forecasting. By coupling multi-scale causal convolutions with cross-attention temporal fusion and an asymmetric direction-calibrated loss, M2-WaveAttnNet effectively resolves the lag problem inherent in classical recurrent models, producing superior directional accuracy and risk-adjusted trading returns. Integrated Explainable AI analysis provides transparent, auditable verification of market drivers.
