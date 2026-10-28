# TS-Booster

**Official code for the paper**
**_TS-Booster: A Residual Correction Module for Short-Horizon Rolling Volatility Forecasting across Neural Backbones_**
(Chih-Chien Hsieh, Shih-Chieh Lin, Chung-Hsun Cheng, Mu-Yen Chen; manuscript under review, 2026)

[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Built on Time-Series-Library](https://img.shields.io/badge/Built%20on-Time--Series--Library-8A2BE2.svg)](https://github.com/thuml/Time-Series-Library)

This repository contains everything needed to reproduce the experiments in the paper: the proposed **TS-Booster** module on its **four neural hosts** (Linear, LSTM, GRU, Transformer) with its **ablation**, **gate** and **control** variants, the paper's own **DLinear / PatchTST / iTransformer** reimplementations, the **statistical and machine-learning comparators**, the **data pipeline** that builds the 16 realized-volatility series, a **batch runner** with resumable sweeps and verified artifacts, the **significance tests**, and an **analysis layer** that recomputes every table and figure of the paper from the run records.

---

## Table of Contents

- [Overview](#overview)
- [What makes the benchmark _fair_](#what-makes-the-benchmark-fair)
- [TS-Booster — the architecture](#ts-booster--the-architecture)
- [Benchmark models](#benchmark-models)
- [Datasets](#datasets)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the benchmark](#reproducing-the-benchmark)
- [Evaluation and outputs](#evaluation-and-outputs)
- [Analysis — the paper's tables and figures](#analysis--the-papers-tables-and-figures)
- [Adding a model](#adding-a-model)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contributing](#contributing)

---

## Overview

Short-horizon volatility is what risk models actually consume, and its rolling realized version is an awkward target: the 7-day realized volatility of daily log returns is heavy-tailed and regime-dependent, and at a 1–3 step horizon most of the returns inside the target window are already known at the forecast origin. The usual response is a bigger backbone. This paper asks a different question:

> **Can a small, host-agnostic residual-correction module — a learnable decomposition, a few CNN experts trained on the host's residual, and a data-dependent gate — improve whatever neural backbone is already in use on short-horizon rolling-volatility forecasts, and which of its parts do the work?**

To answer it, the repo provides two things:

1. A **fair benchmark** — one data pipeline, chronological split, scaler, objective, seed set and metric set shared by every neural model, and the same test windows for the classical comparators, so differences in results reflect the **model**, not the harness.
2. **TS-Booster plus its probes** — the module wraps a host forecaster (`--tsb_backbone linear | lstm | gru | transformer`) and adds a gated sum of **decomposition-enriched CNN experts** to the host's forecast. It ships with **ablation variants** that remove the decomposition or the experts, **gate variants** that fix or learn the gate, **width-matched enrichment controls**, a **frozen decomposition**, a **parameter-matched MLP**, and framework options for the reviewer-driven controls (a future-only target, observed-window inputs, an MSE anchoring term for models without experts). Sliding TS-Booster across these axes isolates *which components matter* and *whether the gain survives a fair comparison*.

The repository is a focused fork of [thuml/Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library): it keeps TSLib's experiment harness, layer library and 41-model zoo, and hardens the harness for publication-grade experiments — reproducible run identity, verified artifacts, resumable batch sweeps, and an extended evaluation record (accuracy, directional accuracy, timing, parameters, memory).

---

## What makes the benchmark _fair_

Every neural model in this repository is trained and evaluated under an **identical protocol**, and the classical comparators are scored on exactly the same test windows, so accuracy differences are attributable to the method and not to incidental advantages.

| Dimension | Shared setting |
|---|---|
| **Data loader** | `Dataset_Custom` ([`data_provider/data_loader.py`](./data_provider/data_loader.py)) for all neural models; the classical comparators ([`tools/baselines/classical_baselines.py`](./tools/baselines/classical_baselines.py)) take `run.py`'s flags and read the same splits and windows |
| **Splits** | Chronological: the first **70%** trains, the last **20%** tests, the rest validates; validation and test windows are extended back by `seq_len` for look-back context |
| **Scaling** | `StandardScaler` **fit on the training split only**, then applied to all splits; CSVs are read with exact float round-tripping |
| **Inputs / target** | `--features MS`: two channels (`log_return`, `Volatility`), single target `Volatility` (`--c_out 1`). The loader moves the target to the last column; the TS-Booster models reorder it to channel 0 internally, as in the paper's code. `--features S --enc_in 1` uses the target alone |
| **Window** | `seq_len 30`, `label_len 0`, `pred_len 3` — the paper's main setting; the sensitivity study varies the horizon and the look-back |
| **Objective** | `--loss HYBRID`: `(1 − w) · Huber + w · soft-margin direction term` on steps whose true change from the last observed value exceeds a threshold (`--huber_delta 1`, `--dir_weight 0.5`, `--dir_threshold 0.01`). TS-Booster adds its auxiliary terms **in training only**; validation and checkpoint selection use `--loss` alone |
| **Training** | AdamW (`lr 1e-3`, `weight_decay 1e-4`), ReduceLROnPlateau (`factor 0.5`, `patience 5`, `min_lr 1e-5`), gradient clipping at 5, `--train_drop_last`, batch 32, up to 100 epochs with early-stopping patience 30, expert dropout 0.3. The paper's deep comparators train with MSE and Adam over a learning-rate grid chosen **per dataset on validation** ([`tools/select_by_validation.py`](./tools/select_by_validation.py)) |
| **Seeds** | `--rand_seed` ∈ {42, 123, 456, 789, 2024}; `--itr N` runs `rand_seed + 0..N−1` and records each actual seed; deterministic comparators run once (seed 0) |
| **Metrics** | MSE, RMSE, MAE, MAPE, MSPE, R² and **MDA** (directional accuracy against the last observed value) in **both** the original and the normalized scale — pooled, per channel and per horizon step — plus training / inference time, parameter count, model size and GPU / RSS memory peaks |
| **Run identity** | `experiment_id` = SHA-256 of the resolved config, the model's effective config, the code hash, the data hash and the package versions; every execution is a new attempt directory, and `metadata.json` is marked complete last. Batch sweeps verify artifacts and re-run anything missing or invalid |

> **Reproducibility note.** `run.py` seeds Python `random`, NumPy and PyTorch. cuDNN non-determinism is left on by default; `--deterministic` additionally turns on `torch.use_deterministic_algorithms`. The port of TS-Booster was checked against the paper's original code: the same seed builds the same weights and gives bit-identical outputs ([`tests/test_tsbooster.py`](./tests/test_tsbooster.py)), and the classical comparators reproduce the original RMSE / MAE / MDA on matching splits ([`tests/test_tsbooster_controls.py`](./tests/test_tsbooster_controls.py)).

**Known differences from the paper's original code** (deliberate, and recorded here so numbers can be compared):

- The test split starts at `n − int(0.2n)` rather than `int(0.8n)`: when `n` is not a multiple of 5 the validation set gains one window and the test set loses one; the training split and the scaler are identical.
- The validation loss is averaged over valid elements (the original averaged per batch), and early stopping treats an equal score as an improvement (the original required a strict decrease).
- The Taiwan stocks are adjusted from TWSE data instead of Yahoo Finance's adjusted closes, and therefore start on 2003-07-01 (see [Datasets](#datasets)).
- GPU determinism is opt-in (`--deterministic`) rather than always on.

---

## TS-Booster — the architecture

**File:** [`models/my_models/tsbooster/TSBooster.py`](./models/my_models/tsbooster/TSBooster.py) · **`--model TSBooster`** · building blocks in [`layers/TSBooster_layers.py`](./layers/TSBooster_layers.py)

TS-Booster is a **residual-correction module wrapped around a host forecaster**. The host (`--tsb_backbone`) forecasts the target from the raw look-back window exactly as it would on its own. A **learnable multi-scale decomposition** splits the window into trend and seasonal parts; **K Inception-style CNN experts** read the window concatenated with its trend and learn what the host leaves behind; a scalar **SNR gate**, computed from the ratio of the seasonal part's spread to the window's spread, scales their sum before it is added to the host forecast.

**Forward pass** (`B` = batch, `L` = `seq_len`, `C` = `enc_in`, `H` = `pred_len`):

```
x [B, L, C]                                    # look-back window; target in channel 0 inside the module
  │
  ├─ Host(x) ────────────────────────────────────────────────────────► y_base [B, H]
  │
  ├─ MultiScaleDecomp(x) ──────────────────────────────────────────────
  │     trend    = Σ_j softmax(θ)_j · MA_{k_j}(x)     # kernels --tsb_decomp_kernels (7, 13), learnable θ
  │     seasonal = x − trend
  │
  ├─ Experts ──────────────────────────────────────────────────────────
  │     x_expert = [x ‖ trend]                        # --tsb_enrich trend (seasonal / both / raw / zeros)
  │     r = Σ_k Expert_k(x_expert)                    # K = --tsb_n_experts Inception CNNs, kernels 3/5/7, width 32
  │
  ├─ SNR gate ─────────────────────────────────────────────────────────
  │     g = sigmoid(w · std(seasonal) / std(x) + b)   # one scalar per sample; (w, b) initialized to (1, 0)
  │
  └─ y = y_base + g · r                               ──► forecast [B, H]

training loss = HYBRID(y, target) + 0.3 · MSE(g · r, target − sg(y_base)) + 0.1 · MSE(y_base, target)
```

(`std` is taken over time and averaged over channels; `sg` is a stop-gradient; the two weights are `--tsb_aux_weight` and `--tsb_base_weight`.)

The design choices that define the module — and that the variants probe — are:

- **Residual correction, not replacement.** The host is untouched; the experts are supervised on the residual `target − sg(y_base)`, and a small host term keeps `y_base` an honest forecast on its own. Validation and checkpoint selection never see the auxiliary terms.
- **Decomposition-enriched experts.** The experts see `[x ‖ trend]`, where the trend is a softmax-weighted mixture of zero-padded centered moving averages whose mixture weights are learned. The `raw` and `zeros` enrichment modes keep the width but remove the decomposition content, to test whether the content — rather than the extra input width — is what helps.
- **A data-dependent gate with two parameters.** The SNR gate scales the correction by how much of the window's variation is seasonal. The gate variants replace it with a constant or a single learned scalar.
- **Host-agnostic.** The same module and the same flags wrap the Linear, LSTM, GRU and Transformer hosts. The hosts' internal settings (RNN dropout 0.2, Transformer dropout 0.1 without positional encoding) are fixed as in the paper and written to each run's `effective_config.json`.

### Ablation variants

**Folder:** [`models/my_models/tsbooster/ablation_variant/`](./models/my_models/tsbooster/ablation_variant/). Each is a valid `--model` name; all take `--tsb_backbone`.

| `--model` | Paper name | Keeps | Resulting behavior |
|---|---|---|---|
| `TSBooster-vanilla` | Vanilla *host* | host only | The backbone alone — the reference every variant is measured against |
| `TSBooster-woexpert` | +Decomp | decomposition | Two identically initialized copies of the host forecast the trend and seasonal parts and their outputs are summed; no experts, no gate |
| `TSBooster-wodecomp` | +Expert | experts | The experts read the raw window `[x]`; without a decomposition there is no gate |
| `TSBooster` | TS-Booster | both | Decomposition-enriched experts with the SNR gate |

### Gate variants

**Folder:** [`models/my_models/tsbooster/gate_variant/`](./models/my_models/tsbooster/gate_variant/). These keep both branches and change only `g`.

| `--model` | Gate | Mechanism |
|---|---|---|
| `TSBooster-nogate` | None | `g = 1`: the expert sum is added unscaled |
| `TSBooster-constgate` | Const | `g = 0.5`, fixed |
| `TSBooster-learnedgate` | Learned | `g = sigmoid(p)`, one learned scalar, `p` initialized to 1 — static across samples |

### Controls

| Control | How | Tests |
|---|---|---|
| `MLP7K` | 64-48 MLP with 7,171 parameters ([`baselines/MLP7K.py`](./models/my_models/tsbooster/baselines/MLP7K.py)) | Whether TS-Booster+Linear (7,156 parameters) beats a parameter-matched network without the structure |
| `--tsb_enrich raw` / `zeros` | Expert input `[x ‖ x]` / `[x ‖ 0]`, same width and parameter count as `trend`; the decomposition still feeds the gate | Whether the decomposition **content** matters |
| `--tsb_decomp_fixed` | Mixture weights frozen at equal values (a buffer, 2 fewer parameters) | Whether learning the mixture matters |
| `--tsb_expert_type mlp` | MLP experts over the flattened window instead of CNNs | The expert design |
| `--pred_offset k` | Skip `k` steps after the origin and forecast steps `k+1 .. k+H`; with `k = 6`, `H = 3` no target window overlaps the observed returns | How much of the gain comes from already-known returns |
| `--train_mse_weight w` | Add `w · MSE(forecast, target)` to the training loss only; `w = 0.1` with `--loss HYBRID` gives a model without experts TS-Booster's anchoring term | Architecture × objective attribution |
| observed-window inputs | `obs_vol_h1..h3` features from the data builder (`--observed_window_features 3`, train with `--enc_in 5`) | Whether handing the host the known part of the target window closes the gap |

### Hyperparameters

TSLib flags are reused wherever they mean the same thing; TS-Booster-only settings use `--tsb_*` flags; settings that were hard-coded in the paper stay in the model and are written to `effective_config.json`. Defaults are the paper's configuration.

| Flag | Default | Meaning |
|---|---|---|
| `--tsb_backbone` | `linear` | Host model: `linear`, `lstm`, `gru` or `transformer` |
| `--d_model` / `--e_layers` | `512` / `2` | Width and depth of the LSTM, GRU and Transformer hosts (the paper: `64` / `2`); ignored by the Linear host |
| `--n_heads` / `--d_ff` | `8` / `2048` | Transformer host only (the paper: `4` / `128`) |
| `--dropout` | `0.1` | Expert dropout (the paper: `0.3`) |
| `--tsb_decomp_kernels` | `7 13` | Moving-average kernels of the learnable decomposition (positive odd integers) |
| `--tsb_expert_kernels` | `3 5 7` | Conv kernels of each CNN expert |
| `--tsb_expert_hidden` | `32` | Expert width, split evenly across the kernels |
| `--tsb_n_experts` | `2` | Number of boosting experts |
| `--tsb_expert_type` | `cnn` | `cnn` or `mlp` |
| `--tsb_enrich` | `trend` | Expert input: `trend` (`[x ‖ trend]`), `seasonal`, `both`, or the width-matched controls `raw` and `zeros` |
| `--tsb_decomp_fixed` | off | Freeze the decomposition at equal kernel weights |
| `--tsb_aux_weight` / `--tsb_base_weight` | `0.3` / `0.1` | Weights of the residual-expert and host terms of the auxiliary loss |

TS-Booster forecasts one target: use `--features MS --c_out 1` (the target is the last CSV column after loading) or `--features S --enc_in 1`.

**Parameter counts** at `seq_len 30`, `pred_len 3`, two channels, as in the paper's compute study ([`tests/test_tsbooster.py`](./tests/test_tsbooster.py) checks them):

| Host (`--tsb_backbone`) | Host flags | Vanilla | TS-Booster |
|---|---|---|---|
| `linear` | — | 366 | 7,156 |
| `lstm` | `--d_model 64 --e_layers 2` | 50,883 | 57,673 |
| `gru` | `--d_model 64 --e_layers 2` | 38,211 | 45,001 |
| `transformer` | `--d_model 64 --e_layers 2 --n_heads 4 --d_ff 128` | 67,331 | 74,121 |

---

## Benchmark models

TS-Booster is compared against three families of methods, all scored on the same windows.

### (a) The paper's deep comparators — `models/my_models/tsbooster/baselines/`

The paper's own reimplementations of three deep forecasters, ported so that they reproduce its numbers. Their settings are **fixed** (dropout 0.1 included); they ignore TSLib's width flags (`--d_model`, `--n_heads`, `--e_layers`, `--d_ff`, `--dropout`) and record what they use in `effective_config.json`.

| `--model` | Architecture | Parameters |
|---|---|---|
| `DLinear-paper` | Zero-padded moving average (kernel 25) splits each channel into trend and remainder; per-channel linear maps forecast both | 372 |
| `PatchTST-paper` | Patches of length 5, stride 3, learnable position embedding, shared 3-layer Transformer (`d_model` 128, 8 heads, `d_ff` 256), RevIN on the input, flatten head | 402,823 |
| `iTransformer-paper` | One token per variate (linear embedding plus a learnable variate embedding), 2-layer Transformer (`d_model` 128, 4 heads, `d_ff` 256) across variates, linear head on the target token, no input normalization | 269,571 |

TSLib's own `DLinear`, `PatchTST` and `iTransformer` differ in padding, patching, normalization and embeddings, so they are separate models; they can be run as a check beyond the paper.

### (b) Statistical and machine-learning comparators — `tools/baselines/classical_baselines.py`

[`tools/baselines/classical_baselines.py`](./tools/baselines/classical_baselines.py) fits the paper's non-neural comparators on the training split and evaluates them on the framework's test windows. It takes `run.py`'s flags, runs as a `long_term_forecast` / `is_training 1` run and writes the standard artifacts, so sweeps, reports and significance tests treat these methods like models. The settings are the paper's and fixed (listed in `effective_config.json`). The statistical models read the whole series up to the forecast origin, not only the look-back window.

| `--model` | Method |
|---|---|
| `Persistence` | `y_t` for every step |
| `ObservedWindowRule` | The paper's Eq. 19: the sample standard deviation of the target window's already-observed returns, annualized |
| `EWMA` | RiskMetrics variance of the returns, λ = 0.94; flat single-day volatility (a reference comparison) |
| `HAR-RV` | OLS of `y_{t+h}` on the daily value and the 5- and 22-day means of the target, one fit per step |
| `GARCH` / `GJR-GARCH` | (1,1) fitted with `arch` on the training returns, analytic h-step variance from the filtered state (reference comparisons) |
| `ARIMA` | `pmdarima.auto_arima` on the training and validation targets, updated with each observed test value |
| `XGBoost` / `LightGBM` | One regressor per step on the flattened scaled window, ≤ 500 trees, depth 6, learning rate 0.05, early stopping on the validation windows; seeded by `--rand_seed` |
| `EWMA-aligned` / `GARCH-aligned` / `GJR-GARCH-aligned` | Revision control: the same variance models forecasting the rolling target itself, combining the window's known returns with model variances of its future returns (`E[S²]`) |

EWMA and the GARCH models forecast single-day conditional volatility rather than the rolling target, which is why the paper reports them as reference comparisons and the `-aligned` variants exist.

### (c) The TSLib model zoo — `models/tslib_models/`

The 41 TSLib models are kept and runnable through the same harness: 34 trainable baselines (Autoformer, Crossformer, DLinear, FEDformer, Informer, iTransformer, Koopa, Mamba, PatchTST, SegRNN, TimeMixer, TimesNet, TimeXer, …) and **7 pretrained foundation models** run **zero-shot** through the `zero_shot_forecast` task with `--is_training 0`:

| `--model` | Weights (Hugging Face Hub) | Requirements file |
|---|---|---|
| `Chronos` / `Chronos2` | `amazon/chronos-bolt-base` / `amazon/chronos-2` | `requirements/foundation-transformers.txt` |
| `TimeMoE` | `Maple728/TimeMoE-50M` | `requirements/foundation-transformers.txt` |
| `Sundial` | `thuml/sundial-base-128m` | `requirements/foundation-transformers.txt` |
| `TimesFM` | `google/timesfm-2.5-200m-pytorch` | `requirements/foundation-timesfm.txt` |
| `TiRex` | `NX-AI/TiRex` | `requirements/foundation-tirex.txt` |
| `Moirai` | `Salesforce/moirai-2.0-R-small` (uni2ts) | `requirements/foundation-moirai.txt`, then `uni2ts` with `--no-deps` |

Provider metadata (model id, `--model_revision`, device, dtype) is recorded in each run; pin an immutable revision for a reproducible comparison.

---

## Datasets

The paper uses **16 daily series**: 10 exchange rates and 6 Taiwan equities. The forecast target is the **7-day realized volatility of daily log returns**, annualized and in percent:

```
log_return_t = ln(C_t / C_{t−1})                                  # first value set to 0
Volatility_t = std(log_return_{t−6}, …, log_return_t; ddof=1) · √252 · 100
```

Each CSV under `dataset/volatility/` has the columns `date, log_return, Volatility`; rows without a full 7-day window are dropped.

| File | Instrument | Type | Source |
|---|---|---|---|
| `USD_TWD` | US dollar / New Taiwan dollar | FX | Investing.com (investpy) |
| `EUR_TWD` | Euro / New Taiwan dollar | FX | Investing.com |
| `JPY_TWD` | Japanese yen / New Taiwan dollar | FX | Investing.com |
| `CAD_TWD` | Canadian dollar / New Taiwan dollar | FX | Investing.com |
| `CNY_TWD` | Chinese yuan / New Taiwan dollar | FX | Investing.com; 2003-12 → 2005-07 gap filled from FRED cross rates |
| `KRW_TWD` | South Korean won / New Taiwan dollar | FX | Investing.com |
| `THB_TWD` | Thai baht / New Taiwan dollar | FX | Investing.com; faulty 2001, 2007-05-28 and 2007-11 → 2008-02 quotes replaced from FRED cross rates |
| `EUR_USD` | Euro / US dollar | FX | Investing.com |
| `GBP_USD` | British pound / US dollar | FX | Investing.com |
| `USD_JPY` | US dollar / Japanese yen | FX | Investing.com |
| `0050_TW` | Yuanta Taiwan Top 50 ETF | Equity | TWSE closes via FinMind, adjusted |
| `2317_TW` | Hon Hai Precision Industry | Equity | TWSE via FinMind, adjusted |
| `2330_TW` | Taiwan Semiconductor Manufacturing | Equity | TWSE via FinMind, adjusted |
| `2454_TW` | MediaTek | Equity | TWSE via FinMind, adjusted |
| `2603_TW` | Evergreen Marine | Equity | TWSE via FinMind, adjusted |
| `2881_TW` | Fubon Financial Holding | Equity | TWSE via FinMind, adjusted |

The exchange rates run from 2000-01 (the experiments use the paper's snapshot, ending 2026-02-16); the equities start on 2003-07-01, when the exchange's ex-rights records begin (0050 listed in June 2003).

**Building the data.** `dataset/` is git-ignored; the tools in [`tools/data/`](./tools/data/) rebuild it with a provenance manifest next to every file:

```bash
# 1. FX closes from Investing.com (investpy), with provenance manifests
python tools/data/fetch_tsbooster_data.py --output dataset/raw
# 2. CNY/TWD gap and THB/TWD bad quotes repaired with FRED cross rates (every change listed in the manifest)
python tools/data/repair_fx.py --input dataset/raw --output dataset/raw --overwrite
# 3. Taiwan stocks: TWSE closes adjusted with the exchange's reference prices, from 2003-07-01
python tools/data/adjust_tw_prices.py --output dataset/raw --overwrite
# 4. date, log_return, Volatility; warns about suspicious daily jumps
python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility
# 5. Optional: + obs_vol_h1..h3, the known part of each target window (Eq. 19), for the observed-window experiment
python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility_obs --observed_window_features 3
```

**Why the stocks are not Yahoo Finance's adjusted series.** Yahoo's adjusted closes for these stocks move beyond the TWSE daily price limit (7% before June 2015, 10% after) — 0050 by −139% on 2014-01-02 and 2603 by −17.8% on 2022-09-19 (an unadjusted capital reduction inside the test period), which distort both the volatility target and the training-set scaler. [`adjust_tw_prices.py`](./tools/data/adjust_tw_prices.py) adjusts the exchange's own closes for dividends, capital reductions and splits with its reference prices (factor = reference price ÷ last close before the event) and stops if any move still breaks the limit. The builder reports, but does not alter, daily moves beyond `--jump_threshold`; check its warnings before training.

**Loading convention.** CSVs are read through the `custom` provider with multivariate-input / single-target settings:

| Flag | Value | Meaning |
|---|---|---|
| `--data` | `custom` | use `Dataset_Custom` |
| `--root_path` / `--data_path` | `./dataset/volatility/` / `USD_TWD.csv` | one series per run |
| `--features` | `MS` | both columns in, single target out |
| `--target` | `Volatility` | predict the volatility (moved to the last column internally) |
| `--freq` | `b` | business-day frequency for the temporal embedding |
| `--enc_in` / `--dec_in` / `--c_out` | `2` / `2` / `1` | two inputs, one target output (`5` / `5` / `1` with the observed-window features) |
| `--inverse` | set | make the original scale primary; both scales are always recorded |

> `run.py` inherits TSLib's defaults (`--data ETTh1`, `--features M`, `--target OT`, `--freq h`), so the volatility configuration must be passed explicitly, as shown below.

---

## Repository structure

```text
TS-Booster/
├── run.py                              # Single entry point — CLI validation, seeding, device & task dispatch
├── models/
│   ├── capabilities.py                 # Which tasks each model implements (declared, checked at build time)
│   ├── my_models/tsbooster/
│   │   ├── TSBooster.py                # proposed model (--model TSBooster, --tsb_backbone linear|lstm|gru|transformer)
│   │   ├── ablation_variant/           # TSBooster-vanilla · TSBooster-woexpert · TSBooster-wodecomp
│   │   ├── gate_variant/               # TSBooster-nogate · TSBooster-constgate · TSBooster-learnedgate
│   │   └── baselines/                  # MLP7K · DLinear-paper · PatchTST-paper · iTransformer-paper
│   └── tslib_models/                   # 41 TSLib models, including 7 pretrained foundation models
├── layers/                             # TSBooster_layers.py, TSBooster_baselines.py, TSLib layers, foundation_adapter.py
├── exp/                                # Task pipelines; exp_basic.py (model registry); common/{runtime,training,evaluation}.py
├── data_provider/                      # data_factory.py, data_loader.py (Dataset_Custom), preparation.py
├── utils/                              # metrics.py (incl. MDA), forecast_losses.py (HYBRID), visualization.py, ADFtest.py, …
├── tools/
│   ├── batch_runner.py                 # Resumable sweeps with verified artifacts (sweep_main)
│   ├── baselines/classical_baselines.py# Statistical and ML comparators on the framework's windows
│   ├── data/                           # fetch_tsbooster_data.py, repair_fx.py, adjust_tw_prices.py, build_volatility_dataset.py
│   ├── stats/                          # significance.py (paired / DM / Friedman–Nemenyi), descriptive_stats.py
│   ├── benchmark.py                    # Parameter counts and forward latency (the paper's compute study)
│   ├── select_by_validation.py         # Per-dataset learning-rate choice for the deep comparators
│   ├── parse_result_to_xlsx.py         # Excel / CSV reports with mean ± std over seeds
│   └── summarize_m4.py                 # Inherited: official M4 summary
├── analysis/                           # The paper's tables, figures and reviewer analyses, from results/ (run_all.py)
├── tests/                              # CPU contract, parity and end-to-end tests
├── scripts/
│   ├── check.py                        # Syntax check + the full test suite
│   └── <task>/                         # Inherited TSLib reproduction scripts (standard benchmarks — see note)
├── requirements/                       # Split dependency groups: core, dev, tools, baselines, foundation-*, reqs_1..4
├── Dockerfile · docker-compose.yml     # CUDA 12.4 / PyTorch 2.6.0 image with the full stack
└── LICENSE · CONTRIBUTING.md
```

> **Note on `scripts/<task>/`.** These shell scripts are inherited from upstream TSLib and target the _standard_ academic benchmarks (ETT, ECL, Weather, M4, anomaly / classification datasets). They are **not** the experiments of this paper, which run through `run.py` and the batch runner documented below.

`dataset/`, `results/`, `checkpoints/`, `test_results/`, `batch_runs/`, `benchmark_results/` and `analysis/output/` are git-ignored outputs.

---

## Installation

**Python 3.11 – 3.13** (verified on 3.11). Dependencies are split by purpose under [`requirements/`](./requirements/).

### Minimal install (TS-Booster, its variants and the deep comparators)

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/core.txt -r requirements/dev.txt   # torch 2.6.0 + the scientific stack, pytest
python scripts/check.py                                                   # syntax check + the full CPU test suite
```

> **CUDA:** `requirements/core.txt` pins `torch==2.6.0` from PyPI (the CPU / MPS wheel on macOS). On a Linux CUDA host install the CUDA 12.4 build first, then the rest:
> ```bash
> pip install -r requirements/reqs_1.txt          # torch==2.6.0+cu124 via the PyTorch index
> pip install -r requirements/core.txt -r requirements/dev.txt
> ```

### Tools and comparators

```bash
python -m pip install -r requirements/tools.txt      # investpy, yfinance, statsmodels, openpyxl: data download, statistics, Excel
python -m pip install -r requirements/baselines.txt  # arch, pmdarima, xgboost, lightgbm: the statistical and ML comparators
```

### Optional model providers

| File | Enables |
|---|---|
| `requirements/reformer.txt` | TSLib's `Reformer` |
| `requirements/uea.txt` | `.ts` classification datasets (not needed for the paper) |
| `requirements/foundation-transformers.txt` | `Chronos`, `Chronos2`, `TimeMoE`, `Sundial` |
| `requirements/foundation-timesfm.txt` / `foundation-tirex.txt` | `TimesFM` / `TiRex` |
| `requirements/foundation-moirai.txt` | `Moirai` runtime deps; then `pip install --no-deps uni2ts==2.0.0` |

The full four-step environment (`requirements/reqs_1..4.txt`) installs everything at once; `reqs_3.txt` pins a `mamba_ssm` wheel for Linux x86-64 + CUDA 12 + torch 2.6 only, and `reqs_4.txt` must be installed with `--no-deps`.

### Docker (optional)

```bash
docker compose up -d --build
docker compose exec dev_tslib bash
```

The image builds on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, installs `reqs_1..4`, runs with GPU passthrough (`NVIDIA_VISIBLE_DEVICES=all`), `shm_size: 8gb`, the repository mounted at `/workspace` and a persistent Hugging Face cache volume.

### Verifying the install

[`scripts/check.py`](./scripts/check.py) parses every Python file and runs the full test suite on the CPU — model contracts, data splits, metric formulas, artifact integrity, batch resume, the TS-Booster parity tests and the analysis layer on a synthetic campaign — without downloading anything. On a GPU host, `python scripts/check.py --device cuda` (or `--device mps`) also runs the real train → standalone-test smoke on that device; `--spawn` adds real DataLoader workers.

---

## Reproducing the benchmark

All experiments run through [`run.py`](./run.py), which validates the flags, seeds the RNGs, resolves the device and dispatches to the task pipeline.

### Train and evaluate TS-Booster with the paper protocol

```bash
python run.py --task_name long_term_forecast --is_training 1 \
  --model_id USD_TWD --model TSBooster --tsb_backbone linear \
  --data custom --root_path ./dataset/volatility/ --data_path USD_TWD.csv \
  --features MS --target Volatility --freq b --enc_in 2 --dec_in 2 --c_out 1 \
  --seq_len 30 --label_len 0 --pred_len 3 --dropout 0.3 \
  --loss HYBRID --optimizer adamw --weight_decay 1e-4 --learning_rate 1e-3 \
  --lradj plateau --min_lr 1e-5 --clip_grad 5 --train_drop_last \
  --train_epochs 100 --patience 30 --batch_size 32 --inverse --rand_seed 42
```

The remaining HYBRID and plateau settings (`--huber_delta 1 --dir_weight 0.5 --dir_threshold 0.01`, `--lr_factor 0.5 --lr_patience 5 --lr_threshold 1e-3`) default to the paper's values. `--itr 5` runs seeds `42..46` in one call; the paper's seed set is run as separate seeds by the batch runner.

### Hosts, ablations, gates and controls

Swap `--model` and the host flags; every other flag stays the same:

```bash
python run.py --model TSBooster --tsb_backbone lstm --d_model 64 --e_layers 2                       ...  # TS-Booster+LSTM
python run.py --model TSBooster --tsb_backbone transformer --d_model 64 --e_layers 2 --n_heads 4 --d_ff 128 ...
python run.py --model TSBooster-vanilla --tsb_backbone gru --d_model 64 --e_layers 2                  ...  # the host alone
python run.py --model TSBooster-woexpert --tsb_backbone linear                                      ...  # +Decomp
python run.py --model TSBooster-learnedgate --tsb_backbone linear                                   ...  # a gate variant
python run.py --model TSBooster --tsb_backbone linear --tsb_enrich zeros                            ...  # a width-matched control
python run.py --model MLP7K                                                                         ...  # the parameter-matched control
```

### The paper's deep comparators

The `-paper` models train with MSE and Adam over a learning-rate grid; the paper then picks each model's learning rate **per dataset** by the mean validation RMSE of seeds 42, 123 and 456:

```bash
python run.py --model PatchTST-paper --loss MSE --optimizer adam --learning_rate 1e-3 ...   # repeat over the grid and the seeds
python tools/select_by_validation.py results --output results_lr_selected \
  --models DLinear-paper PatchTST-paper iTransformer-paper                                 # selection.json lists every score
```

The selected results root holds the chosen learning rate's runs for every seed plus, unchanged, every other model's runs, so the reports and significance tests below read it directly.

### Statistical and machine-learning comparators

```bash
python tools/baselines/classical_baselines.py --task_name long_term_forecast --is_training 1 \
  --model HAR-RV --model_id USD_TWD --data custom --root_path ./dataset/volatility/ --data_path USD_TWD.csv \
  --features MS --target Volatility --freq b --seq_len 30 --label_len 0 --pred_len 3 --inverse
```

Swap `--model` for any name in the [comparator table](#b-statistical-and-machine-learning-comparators--toolsbaselinesclassical_baselinespy). The volatility definition (`--vol_window 7 --annualization 252 --vol_scale 100 --return_column log_return`) defaults to the data builder's.

### Zero-shot foundation models

Install the provider's requirements file, then run with no training (`--task_name zero_shot_forecast --is_training 0`); the weights are fetched from the Hugging Face Hub (`--model_revision`, `--model_cache_dir`, `--local_files_only`, `--runtime_dtype`):

```bash
python run.py --task_name zero_shot_forecast --is_training 0 \
  --model_id USD_TWD --model Chronos --data custom \
  --root_path ./dataset/volatility/ --data_path USD_TWD.csv \
  --features MS --target Volatility --freq b --enc_in 2 --dec_in 2 --c_out 1 \
  --seq_len 30 --label_len 0 --pred_len 3 --inverse
```

### Evaluate a saved checkpoint

A standalone test reuses the training run's model and data flags with `--is_training 0 --checkpoint_run <run directory name>` (and the same `--checkpoints` root). It loads the best weights, records their hash and writes a new evaluation attempt.

### Device selection

- `--gpu_type auto` (default) picks CUDA → MPS → CPU; an explicitly requested device that is unavailable is an error.
- **Apple Silicon:** `--gpu_type mps`. **CPU:** `--no_use_gpu`.
- `--gpu N` selects a CUDA device; `--use_multi_gpu --devices 0,1` wraps non-foundation models in `DataParallel`.

### Full sweep

The paper reports results over **16 series × horizons × 5 seeds × models**, scheduled by plan files built on [`tools/batch_runner.py`](./tools/batch_runner.py). The paper's own plan files are not part of this repository; the runner and the protocol above are everything needed to write one. A plan is a Python file that holds every setting and runs without arguments:

```python
# run_batch_main.py
from tools.batch_runner import sweep_main

DATASETS = ["USD_TWD", "EUR_TWD", "JPY_TWD", "CNY_TWD", "THB_TWD", "KRW_TWD", "CAD_TWD",
            "EUR_USD", "GBP_USD", "USD_JPY",
            "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
SEEDS = [42, 123, 456, 789, 2024]
PROTOCOL = {
    "task_name": "long_term_forecast", "is_training": 1,
    "root_path": "./dataset/volatility/", "data": "custom", "features": "MS", "target": "Volatility", "freq": "b",
    "enc_in": 2, "dec_in": 2, "c_out": 1, "embed": "timeF",
    "loss": "HYBRID", "huber_delta": 1.0, "dir_weight": 0.5, "dir_threshold": 0.01,
    "optimizer": "adamw", "learning_rate": 0.001, "weight_decay": 0.0001,
    "lradj": "plateau", "lr_factor": 0.5, "lr_patience": 5, "lr_threshold": 0.001, "min_lr": 1e-05,
    "clip_grad": 5.0, "train_drop_last": True, "train_epochs": 100, "patience": 30, "batch_size": 32,
    "dropout": 0.3, "d_model": 64, "e_layers": 2, "n_heads": 4, "d_ff": 128,
    "inverse": True, "num_workers": 0, "itr": 1, "des": "Exp", "no_plots": True,
}
EXPERIMENTS = {
    "cross_backbone": {
        "description": "each host alone and with TS-Booster",
        "data_names": DATASETS, "windows": [[30, 0]], "pred_lens": [3], "rand_seeds": SEEDS,
        "models": [
            {"model": "TSBooster-vanilla", "grid": {"tsb_backbone": ["linear", "lstm", "gru", "transformer"]}},
            {"model": "TSBooster", "grid": {"tsb_backbone": ["linear", "lstm", "gru", "transformer"]}},
        ],
        "shared": PROTOCOL,
    },
    "har_rv": {
        "description": "a deterministic comparator, run once",
        "data_names": DATASETS, "windows": [[30, 0]], "pred_lens": [3], "rand_seeds": [0], "deterministic": True,
        "models": ["HAR-RV"], "script": "tools/baselines/classical_baselines.py",
        "shared": {k: v for k, v in PROTOCOL.items()
                   if k in {"task_name", "is_training", "root_path", "data", "features", "target", "freq",
                            "embed", "inverse", "num_workers", "des", "no_plots"}},
    },
}

if __name__ == "__main__":
    raise SystemExit(sweep_main(experiments=EXPERIMENTS, max_parallel=None, run_dir_name="long_term_forecast"))
```

```bash
python run_batch_main.py --list                                  # experiments and their run counts
python run_batch_main.py --dry_run                               # print the commands only
python run_batch_main.py                                         # run everything; rerun to resume
python run_batch_main.py --experiment cross_backbone --datasets USD_TWD --seeds 42 --set train_epochs=2
python run_batch_main.py --gpus 0 1 --report results/report.xlsx
```

A sweep is `data_names × windows × pred_lens × rand_seeds × models`; a model entry is a name or a dict of run options, and a `"grid"` inside it expands into every combination of its value lists. `"script"` points an experiment at another entry point (its options are validated against that script's parser) and `"deterministic": true` marks an experiment whose methods run once (`"rand_seeds": [0]`), which `--seeds` leaves alone. Every option is checked against the parser before anything runs, and missing data files stop the sweep up front. Runs execute as `--max_parallel` subprocesses (default: one per available CPU, honouring a container's CPU quota, at most 32 per GPU) with `--threads_per_job` CPU threads each; `--gpus` spreads runs over the emptiest GPUs. Each experiment is keyed by its exact command and the code, data and environment hashes in a manifest under `batch_runs/<run_dir>/`; the runner locks the manifest, keeps a log per attempt, verifies the return code, the JSON identity, the required files and the finiteness of predictions and targets, and re-runs anything missing or invalid in a new directory — so running the same file again resumes where it stopped, and runs shared by two experiments train once.

---

## Evaluation and outputs

Every attempt writes `results/<setting>__cfg<hash>__attempt<id>/`:

| File | Content |
|---|---|
| `result.json` | `summary` (headline numbers below), `metrics`, `details` (both metric scales, per-channel / per-horizon reports, training history with the learning rate per epoch, timing scopes, model diagnostics such as the decomposition weights and gate statistics) |
| `metadata.json` | Status, identity (`experiment_id`, `group_id`, `attempt_id`), code / data hashes, package versions, device, timestamps |
| `config.json`, `requested_config.json`, `effective_config.json` | Resolved, requested and model-effective configuration |
| `pred.npy`, `true.npy`, `metrics.npy` | Forecasts, targets, legacy six-value metrics (MAE, MSE, RMSE, MAPE, MSPE, R²) |
| `last.npy` | Last observed value per window (for MDA and persistence comparisons) |
| `gate.npy` | TS-Booster's expert gate per test sample |
| `checkpoints/<setting>/checkpoint.pth`, `training_state.pth` | Best model by validation loss, and the optimizer / scheduler / RNG state at that point |

`summary` contains `mse`, `rmse`, `mae`, `mape`, `mspe`, `r2`, `mda` (primary scale) plus `metrics_original` / `metrics_normalized`, `train_time_s`, `inference_time_s`, `inference_ms_per_sample`, `inference_samples_per_s`, `evaluation_time_s`, `params`, `trainable_params`, `model_size_mb`, `epochs_trained`, `best_epoch`, `train_peak_gpu_mb`, `eval_peak_gpu_mb` and `process_peak_rss_mb`. MAPE and MSPE are ratios; undefined values (a zero denominator, a constant target for R²) are `null` with explicit metric states rather than silently skipped samples. Forecast figures go to `test_results/<setting>/` unless `--no_plots`.

### Reports

```bash
python tools/parse_result_to_xlsx.py results --output report.xlsx --scale original   # raw + summary sheets
python tools/parse_result_to_xlsx.py results --output report.csv --scale original    # report.csv + report_summary.csv
```

The raw sheet keeps every attempt; the summary uses, per configuration / dataset / scale, the latest complete attempt of each seed and reports mean ± std over seeds. Different hyperparameters are never pooled.

### Significance tests

[`tools/stats/significance.py`](./tools/stats/significance.py) ports the paper's tests and reads the run records directly:

```bash
python tools/stats/significance.py --scale original paired --baseline TSBooster-vanilla --candidate TSBooster
python tools/stats/significance.py --scale original paired --baseline HAR-RV --candidate TSBooster
python tools/stats/significance.py --scale original dm --baseline TSBooster-vanilla --candidate TSBooster --seed 42
python tools/stats/significance.py --results results_lr_selected --scale original friedman \
  --methods TSBooster DLinear-paper PatchTST-paper iTransformer-paper --plot cd.png
python tools/stats/descriptive_stats.py -i dataset/raw -o results/descriptive_stats     # the paper's Table 2
```

| Subcommand | Test |
|---|---|
| `paired` | Per dataset: one-sided paired t-test over seeds (H1: candidate better) — or a one-sample test against a deterministic comparator with a single run — with a 95% CI of the mean % gap, paired Cohen's d and Holm correction across datasets; overall: Wilcoxon signed-rank, binomial sign test and one-sample t-test on the gaps |
| `dm` | Diebold–Mariano with Newey–West variance and the Harvey–Leybourne–Newbold correction on per-window MSE |
| `friedman` | Friedman χ² on the seed-mean ranks per dataset and the Nemenyi critical difference, with an optional CD diagram |

Methods are `MODEL[:key=value,...]` matched against run configs, e.g. `TSBooster:tsb_backbone=lstm` (list values join with `+`, e.g. `tsb_decomp_kernels=5+11`); `--where pred_len=3` filters every run. The formulas are cross-checked against the paper's analysis scripts in [`tests/test_tsbooster_tools.py`](./tests/test_tsbooster_tools.py).

### Compute cost

```bash
python tools/benchmark.py --device cuda --models-only --forwards 200 --rounds 3 --output benchmark_results/gpu_rtx4090.json
```

measures parameter counts and batch-32 forward latency of every host with and without TS-Booster and of the deep comparators at the paper's window; the `efficiency` analysis module reads the newest GPU measurement in `benchmark_results/`.

---

## Analysis — the paper's tables and figures

Once the campaign has run, every table and figure of the paper and every reviewer analysis is recomputed from `results/` and `dataset/` without a GPU:

```bash
python analysis/run_all.py                  # everything, into analysis/output/<module>/
python analysis/run_all.py --only backtest  # one module
python analysis/run_all.py --bootstrap 500  # quicker Ledoit–Wolf tests
```

| Module | Covers |
|---|---|
| `main_results` | Tables 3–4, Fig. 2: each host alone and with TS-Booster |
| `comparators` | Tables 5–7, Figs. 3–5: statistical, machine-learning and deep comparators, the validation-chosen learning rates, ARIMA orders, the Eq. 19 reference |
| `ablation` | Figs. 8–9: component and gate ablations, architecture × objective, incremental significance |
| `sensitivity` | Fig. 7, §4.5: horizon, look-back, decomposition controls, expert kernels, loss terms |
| `efficiency` | Table 8, Fig. 6: parameters, latency and the parameter-matched comparison |
| `gate` | Fig. 10, §4.7: decomposition weights and gate behaviour, within- / between-dataset gate variance |
| `target_information` | Improvement per step, the future-only target, observed-window inputs |
| `backtest` | Fig. 11: inverse-volatility backtest over every seed, with transaction costs, volatility targets and comparators, and Ledoit–Wolf tests of the Sharpe ratios |
| `case_studies` | 0050, KRW/TWD and the dataset summary |

Each module writes its tables as `.csv`, `.md` and `.tex` (booktabs body, ready for the manuscript), its figures as `.png` and `.pdf`, and a `.json` with every number; `analysis/output/SUMMARY.md` collects the numbers the paper quotes next to the paper's own values. The statistical protocol is the paper's: one-sided paired t-tests with Holm correction (one-sample tests against deterministic comparators), Diebold–Mariano with the HLN correction, Friedman and Nemenyi ranks, and the Ledoit–Wolf studentized block bootstrap for Sharpe ratios. Missing runs stop the command with their list unless `--allow-missing`. [`tests/test_analysis.py`](./tests/test_analysis.py) checks the layer on a synthetic campaign.

---

## Adding a model

The registry ([`exp/exp_basic.py`](./exp/exp_basic.py)) **auto-discovers** models: drop a `.py` file anywhere under [`models/`](./models/) that defines a class named `Model`, and its filename (without `.py`) becomes the `--model` string — no manual registration, and the module is imported only when selected. Duplicate names across folders are an error, and an unknown `--model` raises a `ValueError` listing the discovered models. Declare the tasks the model implements with a module-level `SUPPORTED_TASKS` tuple (or `Model.supported_tasks`); the existing models are declared in [`models/capabilities.py`](./models/capabilities.py), and an unsupported model / task pair fails before training. Fixed or ignored hyperparameters belong in the model's `effective_config` dict, which is written to every run — do not turn a setting the paper fixed into a silently tunable flag. A model that trains with extra terms sets `has_auxiliary_loss = True` and implements `auxiliary_loss(aux, target)`; the pipeline calls `forward(..., return_aux=True)` in training and adds the result to `--loss`.

---

## Citation

The paper is under review. Until it is published, please cite this repository:

```bibtex
@misc{hsieh2026tsbooster,
  title  = {{TS-Booster}: A Residual Correction Module for Short-Horizon Rolling Volatility Forecasting across Neural Backbones},
  author = {Hsieh, Chih-Chien and Lin, Shih-Chieh and Cheng, Chung-Hsun and Chen, Mu-Yen},
  year   = {2026},
  note   = {Manuscript under review},
  url    = {https://github.com/twcch/TS-Booster}
}
```

---

## Acknowledgements

This project is a focused fork of the [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) by THUML @ Tsinghua University. It reuses TSLib's experiment harness, layer library and model zoo; the hardened experiment framework (run identity, artifacts, batch sweeps, reports) follows the author's FinTSLib. Foundation-model support builds on the upstream Chronos, Moirai (uni2ts), TimesFM, TiRex, Sundial and TimeMoE projects. Price data come from Investing.com (via investpy), the Taiwan Stock Exchange (via the FinMind API) and FRED.

---

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE).

- Copyright © 2026 Chih-Chien Hsieh
- Copyright © 2021 THUML @ Tsinghua University (Time-Series-Library)

---

## Contributing

This is a **personal research repository** accompanying a publication, so external pull requests are not accepted. It is open source under MIT and **fork-friendly** — you are welcome to fork it, adapt it, and build on the benchmark. Bug reports and questions can be raised as Issues.
