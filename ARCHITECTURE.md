# TS-Booster: Architecture & Technical Documentation

## Overview

TS-Booster is a **model-agnostic time series boosting framework** that wraps any backbone model and enhances its predictions through decomposition-enriched expert modules with adaptive gating. The core insight is using time series decomposition not to split the prediction task (as in DLinear), but to **enrich** auxiliary expert modules with structural information.

### Key Formula

```
ŷ = y_base + gate · y_expert
```

Where:
- `y_base`: backbone prediction on raw input
- `y_expert`: sum of multi-round expert predictions on enriched input
- `gate`: per-sample adaptive gate based on signal-to-noise ratio (SNR)

---

## Architecture Flow

```
Input x ∈ (B, S, C)
│
├──────────────────────────────────── Backbone(x) → y_base ∈ (B, P)
│                                         │
├── MultiScaleDecomp[7,13](x)             │
│   ├── trend   ∈ (B, S, C)              │
│   └── seasonal ∈ (B, S, C)             │
│                                         │
├── Concat: x_enriched = [x, trend]       │
│   → (B, S, 2C)                          │
│                                         │
├── Expert Round 1(x_enriched) → y_r1     │
├── Expert Round 2(x_enriched) → y_r2     │
│   y_expert = y_r1 + y_r2               │
│                                         │
├── SNR Gate:                             │
│   snr = std(seasonal) / std(x)          │
│   gate = σ(w · snr + b)                │
│   y_expert = gate · y_expert            │
│                                         │
└── Fusion: ŷ = y_base + y_expert ────────┘
```

---

## Components

### 1. Input & Data Preprocessing

**File**: `src/utils/data_loader.py`

Input features (2 channels):
- **Volatility**: Rolling 7-day standard deviation of log returns, annualized (×√252×100)
- **Log Return**: ln(Close_t / Close_{t-1})

Data pipeline:
1. Compute log returns and rolling volatility from raw Close prices
2. StandardScaler fitted on training set only (no data leakage)
3. Sliding window: lookback=30, horizon=3
4. Train/Val/Test split: 70%/10%/20% (chronological)

### 2. Backbone (Any Architecture)

**File**: `src/models/layers.py` — `LinearBackbone`, `LSTMBackbone`, `GRUBackbone`, `TransformerBackbone`

The backbone receives raw input `x ∈ (B, 30, 2)` and produces `y_base ∈ (B, 3)`.

TS-Booster is model-agnostic — the backbone can be any time series model. Four backbones are validated:

| Backbone | Architecture | Parameters |
|----------|-------------|------------|
| **Linear** | Flatten → FC(60, 6) → reshape → take channel 0 | ~360 |
| **LSTM** | 2-layer LSTM(2→64) → FC(64, 3) | ~35K |
| **GRU** | 2-layer GRU(2→64) → FC(64, 3) | ~27K |
| **Transformer** | Linear(2→64) → 2-layer TransformerEncoder(d=64, h=4) → FC(64, 3) | ~70K |

The backbone sees **only raw input** — it is never modified by TS-Booster. This is the same backbone used in the "Vanilla" baseline, ensuring fair comparison.

### 3. Multi-Scale Decomposition

**File**: `src/models/layers.py` — `MultiScaleDecomp`

```python
class MultiScaleDecomp(nn.Module):
    def __init__(self, kernels=[7, 13]):
        self.avgs = [AvgPool1d(k) for k in kernels]
        self.weights = nn.Parameter(torch.zeros(len(kernels)))  # learnable

    def forward(self, x):
        w = softmax(self.weights)          # [0.5, 0.5] initially
        trend = Σ w_i · AvgPool(x, k_i)   # weighted ensemble of MAs
        seasonal = x - trend               # residual
        return trend, seasonal
```

**Design choices**:
- **Kernel sizes [7, 13]**: Correspond to ~1 week and ~2 weeks of trading data (seq_len=30). Kernel=3 was tested but hurt performance on THB_TWD (+8.56% → +0.21%).
- **Learnable weights**: 2 parameters with softmax normalization. The model learns the optimal blend of short-term (7-day) and medium-term (13-day) moving averages per dataset.
- **Not dataset-adaptive per sample**: Per-sample adaptive weights (via MLP) were tested but overfitted. Simple dataset-level nn.Parameter works better.

**Purpose**: Decomposition is NOT used for splitting prediction (DLinear-style). It produces:
- **trend**: Used to enrich expert input (denoised structural reference)
- **seasonal**: Used by SNR gate to measure enrichment quality

### 4. Decomposition-Enriched Expert

**File**: `src/models/layers.py` — `MultiScaleExpert`

The expert receives enriched input `x_enriched = [x, trend] ∈ (B, 30, 4)` — raw signal concatenated with trend component.

```python
class MultiScaleExpert(nn.Module):
    """Inception-style Multi-Scale CNN Expert"""
    def __init__(self, seq_len=30, pred_len=3, input_channels=4,
                 hidden_dim=32, kernels=[3, 5, 7]):
        branch_dim = hidden_dim // len(kernels)  # 10 per branch (32//3)
        self.branches = [
            Conv1d(4, 10, kernel_size=3) + BN + LeakyReLU + Dropout,
            Conv1d(4, 10, kernel_size=5) + BN + LeakyReLU + Dropout,
            Conv1d(4, 10, kernel_size=7) + BN + LeakyReLU + Dropout,
        ]
        self.fc = Linear(30 * 30, 3)  # seq_len * concat_dim → pred_len
```

**Architecture**: Inception-style with 3 parallel Conv1d branches at kernel sizes [3, 5, 7], capturing patterns at different temporal scales. Features are concatenated and projected to prediction length via a single FC layer.

**Key design**: The FC layer outputs `pred_len` directly (not `pred_len × channels`). This avoids wasting parameters on multi-channel output when only channel 0 (volatility) is the prediction target.

**Why trend enrichment (not seasonal)**:
- Trend provides a **denoised structural reference** — the expert sees both raw signals and their smoothed version, enabling it to learn residual patterns.
- Seasonal component in financial data (seq_len=30) is mostly noise, not true periodicity.
- Raw + trend is NOT redundant: `trend = weighted_avg(raw)`, so the expert receives both the noisy detail and the denoised summary, learning to exploit the gap.

### 5. Multi-Round Boosting

**File**: `src/models/ts_booster.py` — `TS_Booster.experts`

```python
self.experts = nn.ModuleList([
    MultiScaleExpert(...) for _ in range(n_boost_rounds)  # n=2
])

# Forward:
y_expert_total = 0
for expert in self.experts:
    y_expert_total += expert(x_enriched)
```

Two independent expert modules receive the **same enriched input** and their outputs are **additively combined**. This is NOT sequential residual boosting (where round 2 targets the residual of round 1) — both rounds target the full residual jointly.

**Why not sequential residual boosting**: Tested and found to hurt. With only n=2 rounds, constraining round 2 to target round 1's residual limits cooperation. Joint targeting allows both rounds to specialize on different aspects of the same residual.

**Parameter overhead**: Each expert has ~300 parameters. With n=2 rounds, the expert module adds ~600 params. Combined with decomposition (~2 params), total overhead is ~602 params (9.2% over backbone).

### 6. SNR Gate (Signal-to-Noise Ratio Gate)

**File**: `src/models/ts_booster.py` — SNR gating logic

```python
# Per-sample adaptive gating
seasonal_std = seasonal.std(dim=time).mean(dim=channel)  # scalar per sample
raw_std = x.std(dim=time).mean(dim=channel)              # scalar per sample
snr = seasonal_std / (raw_std + 1e-8)
gate = sigmoid(w · snr + b)         # w, b are learnable (2 params)
y_expert = gate · y_expert           # scale expert contribution
```

**Intuition**: The gate measures how "decomposable" each sample is:
- `seasonal = x - trend`, so `seasonal_std = std(x - trend)` = how much the raw signal deviates from its smoothed version.
- High seasonal_std relative to raw_std → the sample has rich structural patterns → trust the trend-enriched expert more (gate → 1).
- Low seasonal_std → the signal is already smooth → trend adds little value → suppress expert (gate → 0).

**Why SNR gate relates to trend enrichment**: `seasonal_std` directly measures the **informativeness of trend enrichment**. If `std(x - trend)` is high, the trend is very different from raw, meaning the expert receives genuinely new information.

**Only 2 learnable parameters** (`w` and `b`): Minimal overfitting risk, per-sample adaptive behavior.

### 7. Additive Fusion

```python
ŷ = y_base + gate · y_expert
```

Simple additive fusion. The backbone provides the base prediction; the gated expert provides a correction term. This ensures:
- **Backbone is never degraded**: In the worst case (gate → 0), TS-Booster reduces to the vanilla backbone.
- **Monotonic improvement**: The expert can only add information, not replace the backbone.
- **Interpretability**: `gate · y_expert` directly measures the expert's contribution.

---

## Training Pipeline

**File**: `src/engine/trainer.py` — `train_v11()`

### Loss Function: HybridDirectionalLoss

**File**: `src/utils/metrics.py`

```python
L = (1 - α) · L_huber(ŷ, y) + α · L_direction(ŷ, y, x_{t-1})
```

- **Huber Loss** (δ=1.0): Robust to outliers compared to MSE. Balances L1 (outlier-robust) and L2 (smooth gradient) behavior.
- **Directional Loss** (Soft Margin): Penalizes incorrect directional predictions (up vs. down). Uses `soft_margin_loss(pred_delta, sign(true_delta))` only on samples where `|true_delta| > threshold`.
- **Weight** `α=0.5`: Equal emphasis on magnitude accuracy and directional correctness.

### Adaptive Multi-Objective Loss Weighting

**File**: `src/engine/adaptive_scheduler.py` — `AdaptiveWeightScheduler`

The total training loss has 3 objectives:

```
L_total = L_main + w_aux · L_expert + w_base · L_base
```

- `L_main`: HybridDirectionalLoss on final output `ŷ`
- `L_expert`: MSE between expert output and backbone residual `(y - y_base.detach())`
- `L_base`: MSE between backbone output and target

The scheduler dynamically adjusts `w_aux` and `w_base`:
1. **Warmup phase** (first 20% of epochs): Low auxiliary weights, let the model learn basic patterns.
2. **Search phase**: Monitor base vs. expert loss gaps. If expert lags → increase `w_aux`. If base lags → increase `w_base`.
3. **EMA smoothing** (α=0.7) on loss gaps + **momentum** (β=0.7) on weight updates to prevent oscillation.

### Optimizer & Scheduling

- **AdamW** (lr=0.001, weight_decay=1e-4)
- **ReduceLROnPlateau** (factor=0.5, patience=5, min_lr=1e-5)
- **Gradient clipping**: max_norm=5.0
- **Early stopping**: patience=30 epochs on validation loss

---

## Hyperparameters (Final Configuration)

**File**: `src/configs/model_config.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `seq_len` | 30 | ~6 weeks of trading data |
| `pred_len` | 3 | 3-day ahead prediction |
| `input_channels` | 2 | Volatility + log_return |
| `decomp_kernels` | [7, 13] | ~1 week + ~2 weeks MA |
| `expert_kernels` | [3, 5, 7] | Multi-scale CNN branches |
| `expert_hidden_dim` | 32 | 10-11 per CNN branch |
| `n_boost_rounds` | 2 | 2 independent expert rounds |
| `expert_type` | cnn | Inception-style CNN (vs MLP) |
| `expert_gate` | snr | SNR-based adaptive gate |
| `enrich_mode` | trend | Trend enrichment (vs seasonal) |
| `dropout` | 0.3 | Applied in expert branches |
| `lr` | 0.001 | AdamW learning rate |
| `weight_decay` | 1e-4 | L2 regularization |
| `dir_weight` | 0.5 | Directional loss weight |
| `epochs` | 100 | With early stopping (patience=30) |

---

## Ablation Study (4 Configurations)

| Config | Backbone | Decomp | Expert | Expert Input | SNR Gate |
|--------|----------|--------|--------|-------------|----------|
| **Vanilla** | 1× on raw | - | - | - | - |
| **+Decomp** | 2× on trend/seasonal | MultiScale[7,13] | - | - | - |
| **+Expert** | 1× on raw | - | 2-round CNN | raw (2ch) | - |
| **TS-Booster** | 1× on raw | MultiScale[7,13] | 2-round CNN | [raw, trend] (4ch) | SNR |

Key observations:
- **+Decomp (-0.1%)**: Splitting prediction hurts. Validates "enrichment, not splitting".
- **+Expert (+12.0%)**: Expert alone provides significant boost. Multi-round CNN is the primary driver.
- **TS-Booster (+13.8%)**: Decomp enrichment + SNR gate provides additional systematic improvement. Statistically significant (Wilcoxon p=0.0046).

---

## Experimental Results Summary

### Ablation (16 datasets, single seed)
- TS-Booster best in **12/16** datasets
- Average improvement over Vanilla: **+13.8%**
- Average improvement over +Expert: **+1.8%** (Wilcoxon p=0.005)

### Multi-Seed Robustness (5 seeds × 16 datasets)
- **13/16 datasets** with positive mean improvement (81.3%)
- 0/16 datasets with statistically significant degradation
- Stocks: 6/6 = 100% positive, avg +2.83%
- FX: 7/10 = 70% positive, avg +0.35%

### Cross-Backbone (4 backbones × 16 datasets)
- **61/64 wins (95.3%)**
- Linear: 16/16, avg +13.8%
- LSTM: 13/16, avg +9.5%
- GRU: 16/16, avg +9.0%
- Transformer: 16/16, avg +18.5%

---

## File Structure

```
src/
├── models/
│   ├── ts_booster.py         # Production TS-Booster module
│   ├── flexible_booster.py   # Ablation variant with component toggles
│   ├── layers.py             # MultiScaleDecomp, MultiScaleExpert, Backbones
│   └── factory.py            # Model factory (creates Vanilla + Boosted pairs)
├── engine/
│   ├── trainer.py            # Training loop (train_v11)
│   └── adaptive_scheduler.py # Adaptive multi-objective loss weighting
├── utils/
│   ├── data_loader.py        # Data preprocessing & sliding window
│   └── metrics.py            # HybridDirectionalLoss, evaluation metrics
├── configs/
│   └── model_config.py       # Central hyperparameter configuration
└── preprocess/
    ├── get_fx.py             # FX data fetching
    ├── get_stock.py          # Stock data fetching
    └── getUSDTWD.py          # USD/TWD specific fetcher

experiments/
├── comprehensive_evaluation.py   # Ablation + cross-backbone evaluation
├── run_ablation_all.py           # Batch ablation runner (16 datasets)
├── run_cross_backbone_all.py     # Batch cross-backbone runner
├── multi_seed_test.py            # 5-seed robustness test
├── visualize_comprehensive.py    # Result visualization
├── plot_architecture_v2.py       # Architecture diagram generator
└── generate_paper_figures.py     # Paper figure generator

dataset/                          # 16 CSV files (7 TWD FX + 3 intl FX + 6 TW stocks)
```
