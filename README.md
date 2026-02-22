# TS-Booster: Plug-and-Play Time Series Forecasting Enhancement

**TS-Booster** is a lightweight adapter module that wraps any backbone model to improve time series forecasting. It achieves **+15.06% average RMSE improvement** across 16 datasets with only **+6,790 additional parameters**.

---

## Key Results

### Vanilla vs TS-Booster (5 seeds, 16 datasets)

| Category | Win Rate | Avg Improvement |
|----------|----------|-----------------|
| Taiwan Stocks (6) | **6/6** | +19.01% |
| TWD FX (7) | **7/7** | +12.33% |
| International FX (3) | **3/3** | +13.55% |
| **Overall** | **16/16** | **+15.06%** |

All 16 datasets statistically significant (p < 0.01, paired t-test).

### Cross-Backbone Validation (16 datasets x 4 backbones)

| Backbone | Win Rate | Avg Improvement |
|----------|----------|-----------------|
| Linear | 16/16 | +13.85% |
| LSTM | 13/16 | +9.47% |
| GRU | 16/16 | +9.04% |
| Transformer | 16/16 | +18.45% |
| **Overall** | **61/64 = 95.3%** | |

### Computational Overhead

Fixed overhead of **+6,790 parameters** regardless of backbone (10-18% for LSTM/GRU/Transformer), ~2x training time, <1.4ms inference.

---

## Architecture

```
Input x [B, 30, 2]
  |
  |--- Backbone(x) ---> y_base               (base prediction)
  |
  |--- MultiScaleDecomp(x) ---> trend, seasonal
  |      |
  |      |--- cat([x, trend]) ---> Expert_1 ---> y_r1
  |      |                    ---> Expert_2 ---> y_r2
  |      |
  |      |--- SNR Gate: gate = sigmoid(w * seasonal_std/raw_std + b)
  |
  y_final = y_base + gate * (y_r1 + y_r2)    (additive fusion)
```

Components:
- **MultiScaleDecomp**: Learnable weighted ensemble of MA(7) and MA(13)
- **Trend-Enriched Expert**: Inception-style CNN (kernel 3, 5, 7) on [raw, trend]
- **SNR Gate**: 2-parameter adaptive gate based on seasonal signal-to-noise ratio
- **Multi-Round Boosting**: 2 independent experts, additive combination

---

## Quick Start

### Install
```bash
uv sync
```

### Run Main Experiments
```bash
# Statistical significance test (Vanilla vs TS-Booster, 5 seeds x 16 datasets)
uv run python experiments/statistical_test.py

# Ablation: +Expert vs TS-Booster (5 seeds x 16 datasets)
uv run python experiments/multi_seed_test.py

# Cross-backbone validation (4 backbones x 16 datasets)
uv run python experiments/run_cross_backbone_all.py

# Interpretability analysis (decomp weights, gate params, pred curves)
uv run python experiments/interpretability.py

# Computational cost analysis
uv run python experiments/compute_cost.py

# Analyze saved results without re-training
uv run python experiments/statistical_test.py --analyze
```

---

## Project Structure

```
paper_booster/
├── src/
│   ├── models/
│   │   ├── ts_booster.py           # Production TS_Booster
│   │   ├── flexible_booster.py     # Ablation model with component toggles
│   │   ├── layers.py               # MultiScaleDecomp, Experts, Backbones
│   │   └── factory.py              # create_model_pair(config)
│   ├── configs/model_config.py     # Central config (BEST_STAGE3_CONFIG)
│   ├── engine/
│   │   ├── trainer.py              # Training loop with multi-objective loss
│   │   └── adaptive_scheduler.py   # (Legacy) Adaptive weight scheduler
│   └── utils/
│       ├── data_loader.py          # Volatility + log_return features
│       └── metrics.py              # HybridDirectionalLoss, financial metrics
│
├── dataset/                        # CSV files (16 datasets)
│
└── experiments/
    ├── statistical_test.py         # Main eval + p-values + LaTeX table
    ├── multi_seed_test.py          # +Expert vs TS-Booster (ablation)
    ├── run_cross_backbone_all.py   # Cross-backbone validation
    ├── interpretability.py         # Learned params + visualizations
    ├── compute_cost.py             # Parameter count, timing
    ├── ablation_adaptive_loss.py   # Fixed vs Adaptive loss weights
    └── comprehensive_results/      # JSON results + figures
```

---

## Training Details

### Loss Function
```
Loss = HybridDirectionalLoss(y_final, y_true)         # 50% Huber + 50% Directional
     + 0.3 * MSE(y_expert, y_true - y_base.detach())  # Expert learns residual
     + 0.1 * MSE(y_base, y_true)                      # Backbone stays accurate
```

### Hyperparameters
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: patience=30
- Gradient clipping: max_norm=5.0
- Epochs: 100

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

**Last updated**: 2026-02-22
