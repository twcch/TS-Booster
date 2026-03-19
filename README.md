# TS-Booster: Plug-and-Play Time Series Forecasting Enhancement

**TS-Booster** is a lightweight, model-agnostic adapter module that wraps any backbone model to improve time series volatility forecasting. It achieves **+15.06% average RMSE improvement** across 16 datasets with only **+6,790 additional parameters**.

---

## Key Results

### Vanilla vs TS-Booster (5 seeds, 16 datasets)

| Category | Win Rate | Avg RMSE Improvement |
|----------|----------|----------------------|
| Taiwan Stocks (6) | **6/6** | +19.01% |
| TWD FX (7) | **7/7** | +12.33% |
| International FX (3) | **3/3** | +13.55% |
| **Overall** | **16/16** | **+15.06%** |

All 16 datasets statistically significant (p < 0.01, paired t-test).

### Cross-Backbone Validation (5 seeds, 16 datasets x 4 backbones)

| Backbone | Vanilla Params | +TSB Params | Win Rate | Avg Improvement |
|----------|---------------|-------------|----------|-----------------|
| Linear | 366 | 7,156 | 16/16 | +15.1% |
| LSTM | 50,883 | 57,673 | 16/16 | +7.6% |
| GRU | 38,211 | 45,001 | 16/16 | +8.3% |
| Transformer | 67,331 | 74,121 | 16/16 | +16.9% |
| **Overall** | | | **64/64 = 100%** | |

### SOTA DL Baselines (5 seeds, fair comparison)

| Comparison | TSB Wins | Significance |
|------------|----------|--------------|
| vs DLinear | **16/16** | All p < 0.05 |
| vs PatchTST (RevIN) | **15/16** | |
| vs iTransformer | **14/16** | |
| **TSB Avg Rank** | **1.12** (best) | Friedman p = 0.000002 |

### Benchmark Baselines (5 seeds)

Methods: ARIMA, XGBoost, LightGBM, Linear, Linear+TSB

TSB avg rank **1.44** (best). Friedman + Nemenyi CD diagram confirms TSB significantly better than all baselines.

### Ablation Study (5 seeds)

| Config | Avg RMSE Improvement | Win Rate |
|--------|---------------------|----------|
| +Decomp only | -0.08% | 7/16 |
| +Expert | +13.91% | 16/16 |
| TS-Booster (full) | **+15.07%** | **16/16** |

- Fixed loss weights > Adaptive (13/16 wins)
- Decomp alone hurts; it enriches the expert, not the backbone

### Sensitivity Analysis (3 seeds, 16 datasets)

| Hyperparameter | Variants | Result |
|---------------|----------|--------|
| Decomp kernels | [5,11] / [7,13] / [11,21] / [7,13,21] | All within +/-0.11% -- **robust** |
| Boost rounds | 1 / 2 / 3 | All within +/-0.15% -- **robust** |
| Aux weight | 0.1 / 0.3 / 0.5 | 0.5 best, 0.1 hurts -- **matters** |
| Base weight | 0.0 / 0.1 / 0.3 | 0.3 best, 0.0 hurts -- **matters** |

Architecture params are robust; loss weights matter more.

### Backtest: Inverse-Volatility Position Sizing

Strategy based on Moreira & Muir (2017):

```
w_t = clip(sigma_target / sigma_hat_t, 0.2, 3.0)
r_strat_t = w_t * r_t
```

TSB best Sharpe on 10/16 datasets, avg Sharpe 0.415.

### Computational Overhead

Fixed overhead of **+6,790 parameters** regardless of backbone (10-18% for LSTM/GRU/Transformer). ~2x training time, <1.4ms inference on RTX 4060.

---

## Architecture

```
Input x [B, 30, 2]  (Volatility, log_return)
  |
  |--- Backbone(x) -----------------------> y_base [B, 3]
  |
  |--- MultiScaleDecomp(x) --> trend, seasonal
  |      |
  |      |--- concat([x, trend]) --> [B, 30, 4]
  |      |         |
  |      |    Expert_round1 --> y_r1 [B, 3]
  |      |    Expert_round2 --> y_r2 [B, 3]
  |      |
  |      |--- SNR Gate: gate = sigmoid(w * seasonal_std/raw_std + b)
  |
  y_final = y_base + gate * (y_r1 + y_r2)
```

### Components

- **Backbone**: Any model (Linear / LSTM / GRU / Transformer). Default: Linear (W*x + b, 366 params)
- **MultiScaleDecomp**: Learnable weighted ensemble of MA(7) and MA(13), 2 softmax weights (nn.Parameter)
- **Trend-Enriched Expert**: Inception-style multi-scale CNN with 3 parallel Conv1d branches (kernel 3, 5, 7), concatenated along channel dim, then Linear projection
- **SNR Gate**: 2-parameter (w, b) adaptive gate based on seasonal signal-to-noise ratio
- **Multi-Round Boosting**: 2 independent expert rounds, additive combination (inspired by gradient boosting)

All learnable adapter parameters (decomp weights, gate w/b) are global scalars -- trained once per dataset, fixed at inference.

---

## Data

### Features
- **log_return**: `ln(Close_t / Close_{t-1})`
- **Volatility**: `rolling_std(log_return, 7) * sqrt(252) * 100` (annualized %)

### Datasets (16 total)
- **TWD FX (7)**: USD_TWD, EUR_TWD, JPY_TWD, CAD_TWD, CNY_TWD, KRW_TWD, THB_TWD
- **International FX (3)**: EUR_USD, GBP_USD, USD_JPY
- **Taiwan Stocks (6)**: 0050_TW, 2317_TW, 2330_TW, 2454_TW, 2603_TW, 2881_TW

### Split
- Train: 70% | Validation: 10% | Test: 20%

---

## Loss Function

```
Loss = HybridDirectionalLoss(y_final, y_true)         # 50% Huber + 50% Directional
     + 0.3 * MSE(y_expert, y_true - y_base.detach())  # Expert learns residual
     + 0.1 * MSE(y_base, y_true)                      # Backbone stays accurate
```

- **HuberLoss**: Robust to outliers (delta=1.0)
- **DirectionalLoss**: `soft_margin_loss(pred_delta, sign(true_delta))` -- differentiable surrogate for MDA, rewards correct direction with higher confidence
- **Aux loss (0.3)**: Guides expert to learn the residual between y_true and y_base
- **Base loss (0.1)**: Keeps backbone prediction accurate independently

---

## Evaluation Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| **RMSE** | `sqrt(mean((y_hat - y)^2))` | Lower is better |
| **MAE** | `mean(\|y_hat - y\|)` | Lower is better |
| **MDA** | `(correct direction count) / N` | Higher is better |

MDA measures whether the model correctly predicts if volatility increases or decreases compared to the previous step.

### Backtest Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| **Sharpe** | `(mean(r_strat) * 252) / (std(r_strat) * sqrt(252))` | Higher is better |
| **NAV** | `NAV_t = NAV_{t-1} * (1 + r_strat_t)` | Higher is better |

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early stopping | patience=30 |
| Gradient clipping | max_norm=5.0 |
| Epochs | 100 |
| Batch size | 32 |
| Lookback | 30 days |
| Horizon | 3 days |

---

## Quick Start

### Install
```bash
uv sync
```

### Run All Experiments
```bash
# 1. Main experiment: Vanilla vs TS-Booster (5 seeds x 16 datasets)
uv run python experiments/statistical_test.py

# 2. Ablation: 4-config progression (5 seeds)
uv run python experiments/ablation_5seed.py

# 3. Ablation: Fixed vs Adaptive loss weights (5 seeds)
uv run python experiments/ablation_adaptive_loss.py

# 4. Cross-backbone validation (4 backbones x 16 datasets x 5 seeds)
uv run python experiments/run_cross_backbone_all.py

# 5. Benchmark baselines (ARIMA, XGBoost, LightGBM, Linear, TSB)
uv run python experiments/benchmark_baselines.py

# 6. SOTA DL baselines (DLinear, PatchTST, iTransformer)
uv run python experiments/sota_baselines.py

# 7. Sensitivity analysis (4 hyperparams, 3 seeds)
uv run python experiments/sensitivity_analysis.py

# 8. Interpretability (decomp weights, gate, prediction curves)
uv run python experiments/interpretability.py

# 9. Computational cost (params, training time, inference time)
uv run python experiments/compute_cost.py

# 10. Backtest: Inverse-volatility position sizing (5 methods x 16 datasets)
uv run python experiments/backtest.py

# 11. CD diagrams (Friedman + Nemenyi)
uv run python experiments/cd_diagram.py

# 12. Paper-ready figures
uv run python experiments/generate_paper_figures.py
```

### Analyze-only Mode (from saved results, no re-training)
```bash
uv run python experiments/statistical_test.py --analyze
uv run python experiments/run_cross_backbone_all.py --analyze
uv run python experiments/benchmark_baselines.py --analyze
uv run python experiments/sota_baselines.py --analyze
uv run python experiments/backtest.py --analyze
```

---

## Project Structure

```
paper_booster/
├── src/
│   ├── models/
│   │   ├── ts_booster.py           # Production TS_Booster
│   │   ├── flexible_booster.py     # Ablation model with component toggles
│   │   ├── layers.py               # MultiScaleDecomp, MultiScaleExpert, Backbones
│   │   ├── sota_baselines.py       # DLinear, PatchTST (RevIN), iTransformer
│   │   └── factory.py              # create_model_pair(config) -> (vanilla, booster)
│   ├── configs/
│   │   └── model_config.py         # Central config (BEST_STAGE3_CONFIG)
│   ├── engine/
│   │   ├── trainer.py              # Training loop with multi-objective loss
│   │   └── adaptive_scheduler.py   # (Legacy) Adaptive weight scheduler
│   └── utils/
│       ├── data_loader.py          # Volatility + log_return feature engineering
│       └── metrics.py              # HybridDirectionalLoss, evaluation metrics
│
├── dataset/                        # CSV files (16 datasets)
│
├── experiments/
│   ├── statistical_test.py         # Main 5-seed eval + p-values + LaTeX
│   ├── ablation_5seed.py           # 4-config ablation (5 seeds)
│   ├── ablation_adaptive_loss.py   # Fixed vs Adaptive loss weights (5 seeds)
│   ├── run_cross_backbone_all.py   # Cross-backbone validation (5 seeds)
│   ├── benchmark_baselines.py      # Benchmark baselines (5 seeds)
│   ├── sota_baselines.py           # SOTA DL baselines (5 seeds)
│   ├── sensitivity_analysis.py     # Hyperparameter sensitivity (3 seeds)
│   ├── interpretability.py         # Learned params + visualizations
│   ├── compute_cost.py             # Parameter count, timing
│   ├── backtest.py                 # Inverse-vol position sizing backtest
│   ├── cd_diagram.py               # Friedman + Nemenyi CD diagrams
│   ├── generate_paper_figures.py   # Paper-ready figures
│   ├── plot_architecture_v2.py     # Architecture diagram
│   └── comprehensive_results/      # All JSON results + PNG figures
│       ├── main_experiment/        # Vanilla vs TSB results + figures
│       ├── ablation/               # Ablation results (progression + loss)
│       ├── cross_backbone/         # 4 backbones results + 22 figures
│       ├── benchmark/              # Benchmark results + figures
│       ├── sota_baselines/         # SOTA results + figures
│       ├── sensitivity/            # Sensitivity results + figures
│       ├── interpretability/       # Model internals + visualizations
│       ├── backtest/               # Sharpe + daily returns + figures
│       └── cd_diagram/             # CD diagram results + figures
│
└── conference/                     # Simplified version for conference submission
    ├── src/                        # TSB-Lite (no gate, single MA, single round)
    ├── experiments/                # Subset of experiments
    └── README.md                   # Conference-specific documentation
```

---

## References

- **Backtest**: Moreira, A., & Muir, T. (2017). Volatility-Managed Portfolios. *The Journal of Finance*, 72(4), 1611-1644.
- **Statistical Test**: Friedman test + Nemenyi post-hoc (CD diagrams, alpha=0.05)

---

## Citation

```bibtex
@thesis{tsbooster2026,
  title={TS-Booster: A Plug-and-Play Enhancement Module for Time Series Volatility Forecasting},
  author={Your Name},
  school={Your University},
  year={2026}
}
```

---

## Environment

- Python 3.11 + PyTorch (CUDA)
- GPU: NVIDIA RTX 4060
- Package manager: `uv`
- Run all commands with: `uv run python ...`

**Last updated**: 2026-03-19
