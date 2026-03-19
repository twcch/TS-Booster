# TS-Booster: Plug-and-Play Time Series Forecasting Enhancement

**TS-Booster** is a lightweight, model-agnostic adapter module that wraps any backbone model to improve time series volatility forecasting. It achieves **+15.07% average RMSE improvement** across 16 datasets with only **+6,790 additional parameters**.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Experiment 1: Main Experiment — Vanilla vs TS-Booster](#experiment-1-main-experiment--vanilla-vs-ts-booster)
3. [Experiment 2: Ablation — Component Progression](#experiment-2-ablation--component-progression)
4. [Experiment 3: Ablation — Fixed vs Adaptive Loss](#experiment-3-ablation--fixed-vs-adaptive-loss)
5. [Experiment 4: Cross-Backbone Validation](#experiment-4-cross-backbone-validation)
6. [Experiment 5: Benchmark Baselines](#experiment-5-benchmark-baselines)
7. [Experiment 6: SOTA DL Baselines](#experiment-6-sota-dl-baselines)
8. [Experiment 7: Sensitivity Analysis](#experiment-7-sensitivity-analysis)
9. [Experiment 8: Interpretability](#experiment-8-interpretability)
10. [Experiment 9: Backtest — Inverse-Volatility Position Sizing](#experiment-9-backtest--inverse-volatility-position-sizing)
11. [Experiment 10: Computational Cost](#experiment-10-computational-cost)
12. [Experiment 11: CD Diagrams](#experiment-11-cd-diagrams)
13. [Data & Training Details](#data--training-details)
14. [Quick Start](#quick-start)
15. [Project Structure](#project-structure)

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

### Loss Function

```
Loss = HybridDirectionalLoss(y_final, y_true)         # 50% Huber + 50% Directional
     + 0.3 * MSE(y_expert, y_true - y_base.detach())  # Expert learns residual
     + 0.1 * MSE(y_base, y_true)                      # Backbone stays accurate
```

### Evaluation Metrics

| Metric | Formula | Direction |
|--------|---------|-----------|
| **RMSE** | `sqrt(mean((y_hat - y)^2))` | Lower is better |
| **MAE** | `mean(\|y_hat - y\|)` | Lower is better |
| **MDA** | `(correct direction count) / N` | Higher is better |

---

## Experiment 1: Main Experiment — Vanilla vs TS-Booster

**Setup**: Linear backbone, 5 seeds, 16 datasets, paired t-test.

### RMSE

| Dataset | Vanilla | TS-Booster | Improvement | p-value |
|---------|---------|------------|-------------|---------|
| USD_TWD | 1.8102 | 1.5282 | +15.58% | < 0.0001 |
| JPY_TWD | 2.4169 | 2.0504 | +15.16% | < 0.0001 |
| KRW_TWD | 1.8220 | 1.6839 | +7.58% | 0.0003 |
| CAD_TWD | 1.9763 | 1.7736 | +10.26% | < 0.0001 |
| CNY_TWD | 1.8653 | 1.5986 | +14.30% | < 0.0001 |
| EUR_TWD | 2.0276 | 1.8049 | +10.98% | < 0.0001 |
| THB_TWD | 1.9364 | 1.6950 | +12.47% | 0.0001 |
| EUR_USD | 1.5952 | 1.3999 | +12.24% | < 0.0001 |
| GBP_USD | 1.9204 | 1.6570 | +13.72% | 0.0001 |
| USD_JPY | 2.2813 | 1.9459 | +14.70% | < 0.0001 |
| 0050_TW | 10.3195 | 7.5071 | +27.25% | 0.0001 |
| 2317_TW | 7.8338 | 6.6660 | +14.91% | < 0.0001 |
| 2330_TW | 7.1627 | 5.8629 | +18.15% | < 0.0001 |
| 2454_TW | 8.8069 | 7.5864 | +13.86% | < 0.0001 |
| 2603_TW | 12.8879 | 10.2181 | +20.72% | < 0.0001 |
| 2881_TW | 6.4564 | 5.2176 | +19.19% | < 0.0001 |

### MAE

| Dataset | Vanilla | TS-Booster | Improvement |
|---------|---------|------------|-------------|
| USD_TWD | 0.9506 | 0.7791 | +18.05% |
| JPY_TWD | 1.5365 | 1.3122 | +14.60% |
| KRW_TWD | 1.2121 | 1.1301 | +6.77% |
| CAD_TWD | 1.1164 | 0.9968 | +10.71% |
| CNY_TWD | 0.9244 | 0.7676 | +16.96% |
| EUR_TWD | 1.1713 | 1.0374 | +11.44% |
| THB_TWD | 1.1884 | 1.0818 | +8.97% |
| EUR_USD | 1.1471 | 0.9973 | +13.06% |
| GBP_USD | 1.3346 | 1.1157 | +16.41% |
| USD_JPY | 1.5569 | 1.2939 | +16.89% |
| 0050_TW | 6.9901 | 5.2817 | +24.44% |
| 2317_TW | 5.1343 | 4.3787 | +14.72% |
| 2330_TW | 4.9729 | 3.9901 | +19.76% |
| 2454_TW | 6.2890 | 5.3932 | +14.24% |
| 2603_TW | 8.2888 | 6.4979 | +21.61% |
| 2881_TW | 4.1451 | 3.3684 | +18.74% |

### MDA

| Dataset | Vanilla | TS-Booster | Diff |
|---------|---------|------------|------|
| USD_TWD | 0.6256 | 0.7327 | +10.71pp |
| JPY_TWD | 0.6435 | 0.7036 | +6.01pp |
| KRW_TWD | 0.5814 | 0.6572 | +7.57pp |
| CAD_TWD | 0.6079 | 0.6988 | +9.10pp |
| CNY_TWD | 0.6400 | 0.7229 | +8.29pp |
| EUR_TWD | 0.6263 | 0.7161 | +8.98pp |
| THB_TWD | 0.5998 | 0.7114 | +11.16pp |
| EUR_USD | 0.6232 | 0.7136 | +9.04pp |
| GBP_USD | 0.6467 | 0.7218 | +7.51pp |
| USD_JPY | 0.6183 | 0.7149 | +9.66pp |
| 0050_TW | 0.5931 | 0.6624 | +6.92pp |
| 2317_TW | 0.6218 | 0.6980 | +7.61pp |
| 2330_TW | 0.6198 | 0.7199 | +10.01pp |
| 2454_TW | 0.6236 | 0.7253 | +10.17pp |
| 2603_TW | 0.6308 | 0.7228 | +9.20pp |
| 2881_TW | 0.6286 | 0.7060 | +7.73pp |

### Summary

| Category | Datasets | RMSE Improvement | MAE Improvement | MDA Diff |
|----------|----------|-----------------|-----------------|----------|
| TWD FX | 7 | +12.33% | +12.50% | +8.77pp |
| International FX | 3 | +13.55% | +15.45% | +8.74pp |
| Taiwan Stocks | 6 | +19.01% | +18.92% | +8.61pp |
| **Overall** | **16** | **+15.07%** | **+15.46%** | **+8.73pp** |

**Win rate: 16/16 = 100%, ALL p < 0.01**

---

## Experiment 2: Ablation — Component Progression

**Setup**: 4 configs, 5 seeds, 16 datasets. Configs: Vanilla (baseline), +Decomp, +Expert, TS-Booster (full).

### RMSE by Config

| Dataset | Vanilla | +Decomp | +Expert | TS-Booster |
|---------|---------|---------|---------|------------|
| USD_TWD | 1.8102 | 1.8118 | 1.5407 | **1.5282** |
| JPY_TWD | 2.4169 | 2.4158 | 2.0684 | **2.0504** |
| KRW_TWD | 1.8220 | 1.8361 | 1.6894 | **1.6839** |
| CAD_TWD | 1.9763 | 1.9729 | 1.7926 | **1.7736** |
| CNY_TWD | 1.8653 | 1.8664 | 1.6190 | **1.5986** |
| EUR_TWD | 2.0276 | 2.0220 | 1.8225 | **1.8049** |
| THB_TWD | 1.9364 | 1.8906 | 1.6975 | **1.6950** |
| EUR_USD | 1.5952 | 1.6000 | 1.4184 | **1.3999** |
| GBP_USD | 1.9204 | 1.9170 | **1.6363** | 1.6570 |
| USD_JPY | 2.2813 | 2.2851 | 1.9600 | **1.9459** |
| 0050_TW | 10.3195 | 10.5432 | 7.9932 | **7.5071** |
| 2317_TW | 7.8338 | 7.8573 | 6.7679 | **6.6660** |
| 2330_TW | 7.1627 | 7.1454 | 5.9805 | **5.8629** |
| 2454_TW | 8.8069 | 8.7931 | 7.7013 | **7.5864** |
| 2603_TW | 12.8879 | 12.9565 | 10.4491 | **10.2181** |
| 2881_TW | 6.4564 | 6.4823 | 5.3615 | **5.2176** |

### Improvement vs Vanilla

| Config | Avg RMSE Improvement | Win Rate (vs Vanilla) | Best on |
|--------|---------------------|----------------------|---------|
| +Decomp | -0.08% | 7/16 | 0/16 |
| +Expert | +13.91% | 16/16 | 1/16 (GBP_USD) |
| **TS-Booster** | **+15.07%** | **16/16** | **15/16** |

**Key insight**: Decomp alone hurts (-0.08%) because it enriches the expert input, not the backbone. The expert is the main driver (+13.91%), and the SNR gate adds +1.16pp on top.

---

## Experiment 3: Ablation — Fixed vs Adaptive Loss

**Setup**: 5 seeds, 16 datasets. Compares fixed loss weights (aux=0.3, base=0.1) vs adaptive scheduling.

| Dataset | Fixed RMSE | Adaptive RMSE | Delta | Winner |
|---------|-----------|--------------|-------|--------|
| USD_TWD | **1.5282** | 1.5322 | +0.26% | Fixed |
| JPY_TWD | 2.0504 | **2.0465** | -0.19% | Adaptive |
| KRW_TWD | **1.6839** | 1.6996 | +0.92% | Fixed |
| CAD_TWD | **1.7736** | 1.7829 | +0.52% | Fixed |
| CNY_TWD | **1.5986** | 1.6111 | +0.77% | Fixed |
| EUR_TWD | 1.8049 | **1.7788** | -1.47% | Adaptive |
| THB_TWD | **1.6950** | 1.7825 | +4.91% | Fixed |
| EUR_USD | 1.3999 | **1.3872** | -0.92% | Adaptive |
| GBP_USD | **1.6570** | 1.6634 | +0.39% | Fixed |
| USD_JPY | **1.9459** | 1.9556 | +0.49% | Fixed |
| 0050_TW | **7.5071** | 7.6569 | +1.96% | Fixed |
| 2317_TW | **6.6660** | 6.8987 | +3.37% | Fixed |
| 2330_TW | **5.8629** | 6.0028 | +2.33% | Fixed |
| 2454_TW | **7.5864** | 7.6464 | +0.78% | Fixed |
| 2603_TW | **10.2181** | 10.3713 | +1.48% | Fixed |
| 2881_TW | **5.2176** | 5.2199 | +0.04% | Fixed |

**Fixed wins 13/16 (avg +0.98% better). Simpler and better.**

---

## Experiment 4: Cross-Backbone Validation

**Setup**: 4 backbones x 16 datasets x 5 seeds. Shows TSB is model-agnostic.

### RMSE Improvement (%) by Backbone

| Dataset | Linear | LSTM | GRU | Transformer |
|---------|--------|------|-----|-------------|
| USD_TWD | +15.58% | +0.16% | +1.95% | +16.07% |
| JPY_TWD | +15.16% | +7.19% | +5.16% | +21.77% |
| KRW_TWD | +7.58% | +14.13% | +18.62% | +12.60% |
| CAD_TWD | +10.26% | +5.68% | +2.02% | +14.99% |
| CNY_TWD | +14.30% | +10.70% | +12.81% | +22.71% |
| EUR_TWD | +10.98% | +1.98% | +0.35% | +19.28% |
| THB_TWD | +12.47% | +6.03% | +7.64% | +12.50% |
| EUR_USD | +12.24% | +5.34% | +5.84% | +15.17% |
| GBP_USD | +13.72% | +3.33% | +3.37% | +16.30% |
| USD_JPY | +14.70% | +7.48% | +5.88% | +19.82% |
| 0050_TW | +27.25% | +31.42% | +34.72% | +5.86% |
| 2317_TW | +14.91% | +6.38% | +8.38% | +19.27% |
| 2330_TW | +18.15% | +6.63% | +6.43% | +22.07% |
| 2454_TW | +13.86% | +6.41% | +7.19% | +16.01% |
| 2603_TW | +20.72% | +2.38% | +5.25% | +18.03% |
| 2881_TW | +19.19% | +6.59% | +6.51% | +18.68% |
| **Average** | **+15.07%** | **+7.61%** | **+8.26%** | **+16.95%** |
| **Wins** | **16/16** | **16/16** | **16/16** | **16/16** |

**64/64 = 100% win rate across all backbone-dataset combinations.**

---

## Experiment 5: Benchmark Baselines

**Setup**: 5 methods, 5 seeds (ARIMA deterministic), 16 datasets.

### RMSE (bold = best per row)

| Dataset | Linear | Linear+TSB | XGBoost | LightGBM | ARIMA |
|---------|--------|------------|---------|----------|-------|
| USD_TWD | 1.8102 | **1.5282** | 2.6402 | 2.6217 | 2.0988 |
| JPY_TWD | 2.4169 | **2.0504** | 2.3648 | 2.2635 | 2.6196 |
| KRW_TWD | 1.8220 | 1.6839 | **1.6527** | 1.8415 | 1.9199 |
| CAD_TWD | 1.9763 | **1.7736** | 2.2534 | 2.0388 | 2.1843 |
| CNY_TWD | 1.8653 | **1.5986** | 2.7182 | 2.8169 | 2.1961 |
| EUR_TWD | 2.0276 | **1.8049** | 2.3553 | 2.1367 | 2.0898 |
| THB_TWD | 1.9364 | **1.6950** | 1.9643 | 1.8590 | 2.1573 |
| EUR_USD | 1.5952 | **1.3999** | 1.4511 | 1.4545 | 1.6175 |
| GBP_USD | 1.9204 | **1.6570** | 2.0539 | 1.8205 | 1.9505 |
| USD_JPY | 2.2813 | **1.9459** | 2.1074 | 2.0603 | 2.3639 |
| 0050_TW | 10.3195 | 7.5071 | 13.3053 | 17.2928 | **6.8347** |
| 2317_TW | 7.8338 | **6.6660** | 6.8555 | 6.7564 | 7.9330 |
| 2330_TW | 7.1627 | **5.8629** | 6.5803 | 6.4998 | 7.3559 |
| 2454_TW | 8.8069 | **7.5864** | 8.1698 | 8.3409 | 8.7451 |
| 2603_TW | 12.8879 | **10.2181** | 15.9997 | 14.7999 | 13.7978 |
| 2881_TW | 6.4564 | **5.2176** | 5.9644 | 5.8013 | 6.5417 |

### Summary

| Method | Avg Rank | Best Count |
|--------|----------|------------|
| **Linear+TSB** | **1.12** | **14/16** |
| Linear | 3.19 | 0/16 |
| XGBoost | 3.56 | 1/16 |
| LightGBM | 3.06 | 0/16 |
| ARIMA | 4.06 | 1/16 |

---

## Experiment 6: SOTA DL Baselines

**Setup**: 3 SOTA models (reimplemented with faithful key components), MSE loss + Adam + per-model LR from original papers, 5 seeds, 16 datasets.

### RMSE (bold = best per row)

| Dataset | Linear | Linear+TSB | DLinear | PatchTST | iTransformer |
|---------|--------|------------|---------|----------|--------------|
| USD_TWD | 1.8102 | **1.5282** | 1.8800 | 1.7881 | 2.6345 |
| JPY_TWD | 2.4169 | **2.0504** | 2.5200 | 2.5135 | 2.4946 |
| KRW_TWD | 1.8220 | 1.6839 | 2.0195 | 1.8985 | **1.6314** |
| CAD_TWD | 1.9763 | **1.7736** | 2.0125 | 2.0043 | 2.0979 |
| CNY_TWD | 1.8653 | **1.5986** | 2.0378 | 1.8499 | 2.6839 |
| EUR_TWD | 2.0276 | **1.8049** | 2.0239 | 2.0308 | 2.1997 |
| THB_TWD | 1.9364 | **1.6950** | 2.1465 | 2.0027 | 1.9162 |
| EUR_USD | 1.5952 | **1.3999** | 1.5887 | 1.5913 | 1.5480 |
| GBP_USD | 1.9204 | **1.6570** | 1.9582 | 1.8688 | 1.7836 |
| USD_JPY | 2.2813 | **1.9459** | 2.3262 | 2.3624 | 2.3088 |
| 0050_TW | 10.3195 | 7.5071 | 8.4974 | **6.9266** | 9.6912 |
| 2317_TW | 7.8338 | **6.6660** | 8.1284 | 7.8791 | 7.5569 |
| 2330_TW | 7.1627 | **5.8629** | 7.2912 | 7.1407 | 6.7043 |
| 2454_TW | 8.8069 | **7.5864** | 9.2533 | 9.1181 | 8.7745 |
| 2603_TW | 12.8879 | **10.2181** | 12.7887 | 13.0628 | 14.9238 |
| 2881_TW | 6.4564 | **5.2176** | 6.5120 | 6.2596 | 6.0176 |

### Summary

| Method | Avg Rank | Best Count |
|--------|----------|------------|
| **Linear+TSB** | **1.12** | **14/16** |
| Linear | 3.25 | 0/16 |
| DLinear | 4.12 | 0/16 |
| PatchTST | 3.38 | 1/16 |
| iTransformer | 3.12 | 1/16 |

### Head-to-Head vs TSB

| Comparison | TSB Wins |
|------------|----------|
| vs DLinear | **16/16** |
| vs PatchTST | **15/16** |
| vs iTransformer | **15/16** |

---

## Experiment 7: Sensitivity Analysis

**Setup**: 3 seeds, 16 datasets, 4 hyperparameters. Default config: kernels=[7,13], rounds=2, aux=0.3, base=0.1.

### Decomp Kernels

| Variant | Avg RMSE | vs Default | Wins |
|---------|----------|------------|------|
| [5, 11] | 3.7611 | +0.11% | 6/16 |
| **[7, 13]** (default) | **3.7651** | -- | 3/16 |
| [11, 21] | 3.7686 | -0.09% | 5/16 |
| [7, 13, 21] | 3.7675 | -0.06% | 2/16 |

All within +/-0.11% -- **robust**.

### Boost Rounds

| Variant | Avg RMSE | vs Default | Wins |
|---------|----------|------------|------|
| 1 | 3.7705 | -0.14% | 10/16 |
| **2** (default) | **3.7651** | -- | 2/16 |
| 3 | 3.7708 | -0.15% | 4/16 |

All within +/-0.15% -- **robust**.

### Aux Weight

| Variant | Avg RMSE | vs Default | Wins |
|---------|----------|------------|------|
| 0.1 | 3.8766 | -2.96% | 0/16 |
| **0.3** (default) | **3.7651** | -- | 1/16 |
| 0.5 | 3.7278 | +0.99% | **15/16** |

Loss weights **matter** -- 0.1 significantly hurts.

### Base Weight

| Variant | Avg RMSE | vs Default | Wins |
|---------|----------|------------|------|
| 0.0 | 3.8439 | -2.09% | 0/16 |
| **0.1** (default) | **3.7651** | -- | 6/16 |
| 0.3 | 3.7550 | +0.27% | **10/16** |

**Conclusion**: Architecture params (kernels, rounds) are very robust. Loss weights matter more -- removing aux or base loss significantly hurts performance.

---

## Experiment 8: Interpretability

**Setup**: Single seed, per-dataset analysis of learned parameters.

### Decomposition Weights & SNR Gate

| Dataset | W_7 | W_13 | SNR_w | SNR_b | Gate Mean | Gate Std |
|---------|-----|------|-------|-------|-----------|----------|
| USD_TWD | **0.619** | 0.381 | 0.256 | -0.748 | 0.369 | 0.004 |
| JPY_TWD | **0.547** | 0.453 | 0.425 | -0.605 | 0.436 | 0.009 |
| KRW_TWD | **0.534** | 0.466 | 0.509 | -0.494 | 0.487 | 0.007 |
| CAD_TWD | **0.545** | 0.455 | 0.337 | -0.648 | 0.409 | 0.007 |
| CNY_TWD | **0.638** | 0.362 | 0.454 | -0.548 | 0.455 | 0.008 |
| EUR_TWD | 0.474 | **0.526** | 0.324 | -0.650 | 0.408 | 0.007 |
| THB_TWD | **0.617** | 0.383 | 0.345 | -0.614 | 0.420 | 0.005 |
| EUR_USD | **0.563** | 0.437 | 0.266 | -0.713 | 0.380 | 0.006 |
| GBP_USD | 0.493 | **0.507** | 0.439 | -0.560 | 0.451 | 0.008 |
| USD_JPY | **0.500** | 0.500 | 0.314 | -0.685 | 0.395 | 0.006 |
| 0050_TW | **0.557** | 0.443 | 0.745 | -0.253 | **0.596** | 0.010 |
| 2317_TW | **0.658** | 0.342 | 0.469 | -0.518 | 0.468 | 0.010 |
| 2330_TW | **0.549** | 0.452 | 0.285 | -0.680 | 0.389 | 0.005 |
| 2454_TW | **0.500** | 0.500 | 0.421 | -0.564 | 0.445 | 0.007 |
| 2603_TW | **0.575** | 0.425 | 0.406 | -0.581 | 0.436 | 0.008 |
| 2881_TW | **0.524** | 0.476 | 0.352 | -0.612 | 0.420 | 0.008 |

### Key Observations

- **W_7 > W_13** on 13/16 datasets -- short-term decomposition preferred (exceptions: EUR_TWD, GBP_USD)
- **0050_TW** has highest gate mean (0.596) and highest RMSE improvement (+27.25%) -- stronger expert reliance correlates with larger gains
- **Gate std is tiny** (0.004-0.010) -- gate operates as a per-dataset scalar, not per-sample adaptive
- **SNR_w always positive, SNR_b always negative** -- consistent sigmoid gate behavior across all datasets

---

## Experiment 9: Backtest — Inverse-Volatility Position Sizing

**Setup**: Single seed, 5 methods, 16 datasets. Strategy: `w_t = clip(sigma_target / sigma_hat, 0.2, 3.0)` with sigma_target = 15%. Reference: Moreira & Muir (2017).

### Sharpe Ratio (bold = best per row)

| Dataset | Hist. Vol | XGBoost | LightGBM | Linear | Linear+TSB |
|---------|-----------|---------|----------|--------|------------|
| USD_TWD | 0.406 | 0.376 | 0.390 | 0.361 | **0.415** |
| JPY_TWD | -0.642 | -0.669 | -0.673 | -0.648 | **-0.640** |
| KRW_TWD | -0.585 | -0.549 | -0.548 | -0.597 | **-0.527** |
| CAD_TWD | 0.112 | 0.103 | 0.105 | **0.138** | 0.119 |
| CNY_TWD | 0.278 | 0.283 | 0.251 | 0.277 | **0.288** |
| EUR_TWD | 0.117 | 0.145 | 0.136 | 0.123 | **0.214** |
| THB_TWD | 0.093 | 0.124 | 0.099 | 0.127 | **0.148** |
| EUR_USD | -0.217 | -0.150 | -0.164 | -0.202 | **-0.100** |
| GBP_USD | -0.129 | -0.125 | -0.136 | -0.174 | **-0.121** |
| USD_JPY | 1.084 | 1.091 | 1.094 | 1.098 | **1.121** |
| 0050_TW | 1.510 | 1.532 | 1.477 | 1.500 | **1.597** |
| 2317_TW | **0.799** | 0.790 | 0.778 | 0.700 | 0.717 |
| 2330_TW | **1.107** | 0.960 | 0.980 | 1.029 | 0.950 |
| 2454_TW | **0.755** | 0.685 | 0.733 | 0.661 | 0.680 |
| 2603_TW | 0.324 | 0.620 | **0.680** | 0.311 | 0.573 |
| 2881_TW | **1.374** | 1.229 | 1.220 | 1.229 | 1.201 |

### Summary

| Method | Avg Sharpe | Best Count |
|--------|-----------|------------|
| **Linear+TSB** | **0.415** | **10/16** |
| Hist. Vol | 0.399 | 4/16 |
| XGBoost | 0.403 | 0/16 |
| LightGBM | 0.401 | 1/16 |
| Linear | 0.371 | 1/16 |

---

## Experiment 10: Computational Cost

**Setup**: Per-backbone parameter count, training time, inference time on RTX 4060.

| Backbone | Vanilla Params | +TSB Params | Overhead | Inference (Vanilla) | Inference (+TSB) | Train Ratio |
|----------|---------------|-------------|----------|--------------------|--------------------|-------------|
| Linear | 366 | 7,156 | +6,790 (+1855%) | 0.07ms | 1.01ms | 2.39x |
| LSTM | 50,883 | 57,673 | +6,790 (+13.3%) | 0.29ms | 1.24ms | 2.54x |
| GRU | 38,211 | 45,001 | +6,790 (+17.8%) | 0.23ms | 1.30ms | 2.49x |
| Transformer | 67,331 | 74,121 | +6,790 (+10.1%) | 0.37ms | 1.40ms | 1.73x |

**TSB overhead is fixed at +6,790 parameters** regardless of backbone. For LSTM/GRU/Transformer, this is only 10-18% overhead. Inference stays under 1.4ms. Training is ~2x slower.

---

## Experiment 11: CD Diagrams

**Setup**: Friedman test + Nemenyi post-hoc, alpha=0.05.

| Comparison Group | Friedman p-value | Critical Difference | TSB vs Others |
|-----------------|-----------------|--------------------|----|
| SOTA (5 methods) | 0.000002 | 1.52 | TSB sig. better than ALL |
| Benchmark (5 methods) | 0.000002 | 1.52 | TSB sig. better than ALL |

No other pairwise differences are significant (all within CD).

---

## Data & Training Details

### Features
- **log_return**: `ln(Close_t / Close_{t-1})`
- **Volatility**: `rolling_std(log_return, 7) * sqrt(252) * 100` (annualized %)

### Datasets (16 total)
- **TWD FX (7)**: USD_TWD, EUR_TWD, JPY_TWD, CAD_TWD, CNY_TWD, KRW_TWD, THB_TWD
- **International FX (3)**: EUR_USD, GBP_USD, USD_JPY
- **Taiwan Stocks (6)**: 0050_TW, 2317_TW, 2330_TW, 2454_TW, 2603_TW, 2881_TW

### Data Split
- Train: 70% | Validation: 10% | Test: 20%

### Training Hyperparameters

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
