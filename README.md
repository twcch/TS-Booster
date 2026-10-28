# TS-Booster

**TS-Booster** is a plug-and-play enhancement module for time-series forecasting hosts, evaluated on realized-volatility forecasting for exchange rates and Taiwan equities. This repository runs TS-Booster inside a [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) framework hardened after [FinTSLib](https://github.com/twcch): reproducible run identity, verified artifacts, batch sweeps, and an extended evaluation record (accuracy, directional accuracy, timing, parameters, memory).

[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

```text
x [B, L, C]  (target channel first inside the model)
  ├── Host(x) ──────────────────────────────────────────► y_base [B, H]
  ├── MultiScaleDecomp(x): softmax-weighted MA(7), MA(13) ─► trend, seasonal
  │      [x || trend] ─► Expert_1 + … + Expert_K (Inception CNN, kernels 3/5/7)
  └── gate = sigmoid(w · std(seasonal)/std(x) + b)
y = y_base + gate · Σ Expert_k
loss = Hybrid(y) + 0.3 · MSE(Σ Expert, y − sg(y_base)) + 0.1 · MSE(y_base, y)
```

---

## Key Features

- **TS-Booster as a native TSLib model** — `--model TSBooster` with any of four backbones (Linear, LSTM, GRU, Transformer), plus every ablation of the paper as its own model name. The port is checked against the journal code: same seed, same weights, bit-identical outputs.
- **Every comparator of the paper** — its reimplementations of DLinear, PatchTST and iTransformer as models, and persistence, EWMA, HAR-RV, GARCH, GJR-GARCH, ARIMA, XGBoost and LightGBM through `tools/baselines/classical_baselines.py`, all on the framework's exact windows and reproducing the paper code's numbers.
- **TSLib-compatible hyperparameters** — TSLib flags are reused wherever they mean the same thing (`seq_len`, `pred_len`, `enc_in`, `dropout`, `d_model`, `e_layers`, `n_heads`, `d_ff`); TS-Booster-only settings use `--tsb_*` flags; settings that were hard-coded in the paper stay in the model and are written to `effective_config.json`.
- **General training options** — hybrid directional loss (`--loss HYBRID`), AdamW, ReduceLROnPlateau, gradient clipping and `--train_drop_last` work for every long-term forecasting model, so baselines can be trained with the same objective.
- **Complete run records** — MSE, RMSE, MAE, MAPE, MSPE, R² and MDA in original and normalized scales (pooled, per channel, per horizon), training/inference time, parameter count, model size, memory peaks, training history, configuration, code/data hashes and package versions.
- **Batch sweeps** for every paper experiment, with resume and artifact verification; Excel/CSV reports with mean ± std over seeds.
- **Separate research tools** — data crawlers and dataset builders in `tools/data/`, statistical tests (paired t / Wilcoxon / sign / Holm, Diebold–Mariano with HLN, Friedman–Nemenyi) and descriptive statistics in `tools/stats/`.
- **The TSLib model zoo** — 41 baselines including pretrained foundation models (zero-shot), under `models/tslib_models/`.

---

## Models

| `--model` | Paper name | Description |
| --- | --- | --- |
| `TSBooster` | TS-Booster+*backbone* | Backbone + decomposition-enriched boosting experts + SNR gate |
| `TSBooster-vanilla` | Vanilla *backbone* (Linear, LSTM, …) | The backbone alone |
| `TSBooster-woexpert` | +Decomp | Two identically initialized backbones on trend and seasonal parts |
| `TSBooster-wodecomp` | +Expert | Experts on the raw window, no gate |
| `TSBooster-nogate` / `-constgate` / `-learnedgate` | Gate ablation | Gate fixed at 1 / fixed at 0.5 / one learned scalar |
| `MLP7K` | MLP-7K, parameter-matched control | 64-48 MLP with 7,171 parameters |
| `DLinear-paper` | DLinear (Table 6) | The paper's reimplementation: zero-padded moving average (kernel 25), per-channel linear maps |
| `PatchTST-paper` | PatchTST (Table 6) | Patch length 5, stride 3, learnable position embedding, 3 layers × d_model 128, RevIN |
| `iTransformer-paper` | iTransformer (Table 6) | Variate tokens with a learnable variate embedding, 2 layers × d_model 128, no normalization |

The four backbones (`--tsb_backbone`, the paper's "host") are models in their own right — `TSBooster-vanilla` runs one alone, `TSBooster` adds the module to it — and the experiment files label each entry with the paper's name. The `-paper` comparators keep the paper's fixed settings (dropout 0.1 included) and ignore TSLib's width flags; TSLib's own `DLinear`, `PatchTST` and `iTransformer` differ in padding, patching, normalization and embeddings.

Parameters at `seq_len=30, pred_len=3`, two channels, as in Table 8 (`tests/test_tsbooster.py` checks them):

| Backbone (`--tsb_backbone`) | Backbone flags | Vanilla | TS-Booster |
| --- | --- | --- | --- |
| `linear` | — | 366 | 7,156 |
| `lstm` | `--d_model 64 --e_layers 2` | 50,883 | 57,673 |
| `gru` | `--d_model 64 --e_layers 2` | 38,211 | 45,001 |
| `transformer` | `--d_model 64 --e_layers 2 --n_heads 4 --d_ff 128` | 67,331 | 74,121 |

`DLinear-paper`, `PatchTST-paper` and `iTransformer-paper` have 372, 402,823 and 269,571 parameters.

TS-Booster flags (defaults = paper configuration):

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tsb_backbone` | `linear` | Backbone (host) model |
| `--tsb_decomp_kernels` | `7 13` | Moving-average kernels of the learnable decomposition |
| `--tsb_expert_kernels` | `3 5 7` | Conv kernels of each CNN expert |
| `--tsb_expert_hidden` | `32` | Expert width, split across kernels |
| `--tsb_n_experts` | `2` | Boosting experts |
| `--tsb_expert_type` | `cnn` | `cnn` or `mlp` |
| `--tsb_enrich` | `trend` | Expert input `[x ‖ trend]`; `seasonal`, `both`, or the width-matched controls `raw` (`[x ‖ x]`) and `zeros` (`[x ‖ 0]`) |
| `--tsb_decomp_fixed` | off | Freeze the decomposition at equal kernel weights |
| `--tsb_aux_weight` / `--tsb_base_weight` | `0.3` / `0.1` | Auxiliary training terms |
| `--dropout` | `0.1` (paper: `0.3`) | Expert dropout |

TS-Booster forecasts one target: use `--features MS --c_out 1` (the target is the last CSV column after loading) or `--features S`.

Framework options added for the revision experiments (any encoder-only long-term model):

| Flag | Meaning |
| --- | --- |
| `--pred_offset k` | Skip k steps after the origin and forecast steps k+1..k+H; with k=6, H=3 every target window lies after the origin |
| `--train_mse_weight w` | Add w·MSE(forecast, target) to the training loss only; w=0.1 with `--loss HYBRID` matches TS-Booster's anchoring term for models without experts |

### Statistical and machine-learning comparators

`tools/baselines/classical_baselines.py` fits the paper's non-neural comparators (Table 5) on the training split and evaluates them on the framework's test windows. It takes run.py's flags, runs as `--task_name long_term_forecast --is_training 1` and writes the standard artifacts, so batch sweeps, reports and significance tests treat these methods like models. The settings are the paper's and fixed (see `effective_config.json`); with matching splits every method reproduces the paper code exactly (`tests/test_tsbooster_controls.py`).

| `--model` | Method |
| --- | --- |
| `Persistence` | y_t for every step |
| `ObservedWindowRule` | Eq. 19, the observed part of the rolling target window |
| `EWMA` | RiskMetrics, λ = 0.94; flat single-day volatility (a reference comparison) |
| `HAR-RV` | OLS on the daily value and 5- and 22-day means of the target, one fit per step |
| `GARCH` / `GJR-GARCH` | (1,1) fitted with `arch` on training returns, analytic h-step variance (reference comparisons) |
| `ARIMA` | `pmdarima.auto_arima` on training + validation targets, updated after each test observation |
| `XGBoost` / `LightGBM` | One regressor per step on the flattened window, ≤ 500 trees, depth 6, learning rate 0.05, early stopping on validation; seeded by `--rand_seed` |
| `EWMA-aligned` / `GARCH-aligned` / `GJR-GARCH-aligned` | Revision control (R2.5): the same variance models forecasting the rolling target itself, from the window's known returns and model variances of its future returns (`E[S²]`, zero-mean returns) |

---

## Data

The paper uses 16 daily series: 10 exchange rates (USD, EUR, JPY, CAD, CNY, KRW, THB against TWD; EUR/USD, GBP/USD, USD/JPY) and 6 Taiwan stocks (0050, 2317, 2330, 2454, 2603, 2881). The target is the 7-day realized volatility of log returns, annualized and in percent.

```bash
# FX closes from investpy, with provenance manifests (the experiments use the paper's snapshot, 2000-01 .. 2026-02-16)
python tools/data/fetch_tsbooster_data.py --output dataset/raw
# CNY/TWD gap and THB/TWD bad quotes repaired with FRED cross rates (every change listed in the manifest)
python tools/data/repair_fx.py --input dataset/raw --output dataset/raw --overwrite
# Taiwan stocks: TWSE closes adjusted with the exchange's reference prices, from 2003-07-01
python tools/data/adjust_tw_prices.py --output dataset/raw --overwrite
# date, log_return, Volatility; warns about suspicious jumps
python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility
# + obs_vol_h1..h3: the known part of each target window (Eq. 19), for the observed_window experiment
python tools/data/build_volatility_dataset.py --input dataset/raw --output dataset/volatility_obs --observed_window_features 3
```

The Taiwan stocks do not come from Yahoo Finance as in the paper: its adjusted series move beyond the TWSE daily price limit (7% before June 2015, 10% after), e.g. 0050 by −139% on 2014-01-02 and 2603 by −17.8% on 2022-09-19, an unadjusted capital reduction in the test period. `adjust_tw_prices.py` adjusts TWSE closes (via the FinMind API) for dividends, capital reductions and splits with the exchange's reference prices and stops if any move still breaks the limit. Those records start in July 2003, so the stock series start on 2003-07-01 instead of 2000 (0050 listed in June 2003). `repair_fx.py` fills CNY/TWD's missing 2003-12 .. 2005-07 quotes (the yuan's dollar peg) and replaces THB/TWD quotes that alternate in early 2001, spike on 2007-05-28 or stall from 2007-11 to 2008-02, using USD/TWD over FRED's dollar rates.

`dataset/` is git-ignored. Check the warnings before training: an unadjusted split distorts the volatility target and the training-set scaler. CSVs are read with exact float round-tripping, so models see the values the builder computed.

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements/core.txt -r requirements/dev.txt
python -m pip install -r requirements/tools.txt     # data download, statistics, Excel
python -m pip install -r requirements/baselines.txt # arch, pmdarima, xgboost, lightgbm (Table 5)
python scripts/check.py                              # syntax + full CPU test suite
```

Optional groups: `uea.txt`, `reformer.txt`, `foundation-*.txt`. The full four-step environment and the Docker image (`pytorch/pytorch:2.6.0-cuda12.4`) follow FinTSLib (`requirements/reqs_1..4.txt`, `Dockerfile`). On a CUDA host, `python scripts/check.py --device cuda` also runs the real train → standalone-test smoke on the GPU.

---

## Quick Start

### One run with the paper protocol

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

`--inverse` makes the original scale primary; both scales are always recorded. `--itr N` runs seeds `rand_seed + 0..N-1`. Evaluate a saved checkpoint with `--is_training 0 --checkpoint_run <run directory name>`.

### Experiments: `run_batch_long_term_forecast_01.py` … `_08.py`

The paper and the reviewers' requests are planned as eight self-contained files. Each holds every setting (data paths, parallelism, the protocol, datasets, seeds, models and their hyperparameters) and runs without arguments; run them in order:

```bash
python run_batch_long_term_forecast.py      # setup check first: every model and comparator once, 2 epochs
python run_batch_long_term_forecast_01.py   # ... then 01 to 08
```

| File | Content | Paper / reviewer | New runs |
| --- | --- | --- | --- |
| `_01` | Four backbones alone and with TS-Booster; component ablation | Tables 3–4, Figs. 2, 8 | 800 |
| `_02` | Statistical and ML comparators, the Eq. 19 rule, the paper's DLinear/PatchTST/iTransformer over the LR grid | Tables 5–6 | 1,712 |
| `_03` | Architecture × objective; deep comparators with TS-Booster's objective; EWMA/GARCH/GJR-GARCH on the rolling target | R1.1, R2.1, R2.5 | 768 |
| `_04` | Observed-window inputs; the future-only target (steps 7–9) with backbones and comparators | R1.3, R1.4, R2.4 | 1,264 |
| `_05` | Decomposition controls; +Decomp with nonlinear backbones and frozen weights | R1.9, R2.3 | 640 |
| `_06` | The paper's later batch at 120 epochs: MLP-7K, gates, horizon, lookback | Figs. 6, 7, 9 | 960 |
| `_07` | Sensitivity, loss components, expert kernels (3 seeds) | Sec. 4.5, R2.3 | 768 |
| `_08` | TSLib's DLinear/PatchTST/iTransformer over the LR grid (optional) | beyond the paper | 1,440 |

All files share `results/`, `checkpoints/` and the manifest in `batch_runs/long_term_forecast/`: a run that two experiments share trains once, and running a file again resumes it (only missing or invalid runs execute). Missing data files stop a file before anything runs, and one line per finished run shows progress and an ETA. Every run is a `long_term_forecast` run (`tests/test_tsbooster_controls.py` builds each distinct configuration once and checks that the files share one protocol). Statistics, reports and Table 6's learning-rate choice come afterwards, from `results/`.

An experiment is FinTSLib's sweep: datasets × windows × horizons × seeds × models. A model entry is a model name or a dict of run options; a `"grid"` inside it expands into every combination of its value lists. Classical comparators use `"script": "tools/baselines/classical_baselines.py"`; `"deterministic": true` runs them once (seed 0). The optional flags still work, e.g. `--list`, `--dry_run`, `--experiment main`, `--datasets USD_TWD --seeds 42 --set train_epochs=2`, `--gpus 0 1`, `--report results/report.xlsx`.

Runs execute as `MAX_PARALLEL` concurrent processes, each limited to `THREADS_PER_JOB` CPU threads. By default (`None`) there is one run per available CPU with one thread each, at most 32 per GPU. Available CPUs honour a container's CPU quota, so a Vast.ai RTX 4090 instance that shows 64 CPUs but allows 15.36 runs 15 at once. Measured there on 32 short runs, 4 / 8 / 12 / 16 / 20 parallel runs finished 14.9 / 24.8 / 28.7 / 32.8 / 28.6 runs per minute: past the quota, CPU throttling costs more than the extra runs gain, and two threads per run did not help. On an instance with a 35-CPU quota, 16 / 24 / 32 parallel runs finished 35.8 / 38.7 / 40.4 runs per minute on 64 short runs, with the GPU near saturation and 16 GB of its memory in use at 32, hence the cap.

Table 6 selects each deep comparator's learning rate per dataset by the mean validation RMSE of seeds 42, 123 and 456, as the paper does; `tools/select_by_validation.py` writes a results root with the chosen learning rate's runs (every seed) and all other runs, for reports and significance tests:

```bash
python tools/select_by_validation.py results --output results_lr_selected \
  --models DLinear-paper PatchTST-paper iTransformer-paper      # selection.json lists every score
python tools/parse_result_to_xlsx.py results_lr_selected --output table6.xlsx --scale original
```

### Analysis: `analysis/run_all.py`

Once the eight files have run, every table and figure of the paper and every reviewer analysis is recomputed from `results/` and `dataset/` without a GPU:

```bash
python analysis/run_all.py                  # everything, into analysis/output/<module>/
python analysis/run_all.py --only backtest  # one module
python analysis/run_all.py --bootstrap 500  # quicker Ledoit-Wolf tests
```

| Module | Covers |
| --- | --- |
| `main_results` | Tables 3-4, Fig. 2 |
| `comparators` | Tables 5-7, Figs. 3-5, Table 6's learning rates (validation), ARIMA orders, the Eq. 19 reference |
| `ablation` | Figs. 8-9, architecture x objective (R1.1/R2.1), incremental significance (R2.6) |
| `sensitivity` | Fig. 7, Section 4.5, decomposition controls, expert kernels, loss terms (R1.9/R2.3) |
| `efficiency` | Table 8 parameter counts and latency (from the newest `benchmark_results/*.json` made by `tools/benchmark.py --device cuda`), Fig. 6 |
| `gate` | Fig. 10, Section 4.7, within/between-dataset gate variance (R1.2/R2.2) |
| `target_information` | Improvement per step, the future-only target, observed-window inputs (R1.3/R1.4/R2.4) |
| `backtest` | Fig. 11 with every seed, transaction costs, volatility targets and comparators (R1.7) |
| `case_studies` | 0050 (R1.5/R1.11), KRW/TWD, dataset summary |

Each module writes its tables as `.csv`, `.md` and `.tex` (booktabs body, ready for the manuscript), its figures as `.png` and `.pdf`, and a `.json` with every number; `analysis/output/SUMMARY.md` collects the numbers the paper quotes next to the paper's own values. The statistical protocol is the paper's: one-sided paired t tests with Holm correction (one-sample tests against deterministic comparators), Diebold-Mariano with the HLN correction, Friedman and Nemenyi ranks, and the Ledoit-Wolf studentized block bootstrap for Sharpe ratios. Missing runs stop the command with their list unless `--allow-missing`. `tests/test_analysis.py` checks the layer on a synthetic campaign expanded from the eight plan files.

---

## Outputs

Every attempt writes `results/<setting>__cfg<hash>__attempt<id>/`:

| File | Content |
| --- | --- |
| `result.json` | `summary` (headline numbers below), `metrics`, `details` (both metric scales, per-channel/per-horizon reports, training history with LR, timing scopes, model diagnostics) |
| `metadata.json` | Status, identity, code/data hashes, package versions, device, timestamps |
| `config.json`, `requested_config.json`, `effective_config.json` | Resolved, requested and model-effective configuration |
| `pred.npy`, `true.npy`, `metrics.npy` | Forecasts, targets, legacy six-value metrics (MAE, MSE, RMSE, MAPE, MSPE, R²) |
| `last.npy` | Last observed value per window (for MDA and persistence baselines) |
| `gate.npy` | TS-Booster expert gate per test sample |

`summary` contains: `mse`, `rmse`, `mae`, `mape`, `mspe`, `r2`, `mda` (primary scale) plus `metrics_original` / `metrics_normalized`, `train_time_s`, `inference_time_s`, `inference_ms_per_sample`, `inference_samples_per_s`, `evaluation_time_s`, `params`, `trainable_params`, `model_size_mb`, `epochs_trained`, `best_epoch`, `train_peak_gpu_mb`, `eval_peak_gpu_mb`, `process_peak_rss_mb`. MAPE/MSPE are ratios; undefined values are `null` with explicit metric states. Figures go to `test_results/<setting>/` unless `--no_plots`.

```bash
python tools/parse_result_to_xlsx.py results --output report.xlsx --scale original   # raw + summary sheets
python tools/parse_result_to_xlsx.py results --output report.csv --scale original    # report.csv + report_summary.csv
```

---

## Statistics

```bash
python tools/stats/significance.py --scale original paired --baseline TSBooster-vanilla --candidate TSBooster
python tools/stats/significance.py --scale original dm --baseline TSBooster-vanilla --candidate TSBooster --seed 42
python tools/stats/significance.py --results results_lr_selected --scale original friedman \
  --methods TSBooster DLinear-paper PatchTST-paper iTransformer-paper --plot cd.png
python tools/stats/descriptive_stats.py -i dataset/raw -o results/descriptive_stats
```

Methods are `MODEL[:key=value,...]` matched against run configs, e.g. `TSBooster:tsb_backbone=lstm` (list values join with `+`, e.g. `tsb_decomp_kernels=5+11`); `--where pred_len=3` filters every run. A deterministic baseline with a single run, such as `HAR-RV` or `Persistence` from `statistical`, is compared with a one-sample test against each candidate seed, as in the paper. The formulas are those of the journal analysis scripts and are cross-checked in `tests/test_tsbooster_tools.py`. `analysis/` (above) applies these tests to every comparison of the paper in one command.

---

## Repository Structure

```text
TS-Booster/
├── run.py                              # CLI, validation, seeding, task dispatch
├── run_batch_long_term_forecast.py     # Setup check: every model and comparator once
├── run_batch_long_term_forecast_0N.py  # The experiment plan, 01-08, each self-contained
├── run_batch_zero_shot_forecast.py     # Zero-shot foundation-model sweep
├── models/
│   ├── capabilities.py                 # Model/task contracts
│   ├── tslib_models/                   # 41 TSLib baselines and foundation models
│   └── my_models/tsbooster/            # TS-Booster, ablation/gate variants, MLP7K, the paper's deep comparators
├── layers/                             # Shared layers; TSBooster_layers.py, TSBooster_baselines.py
├── exp/                                # Task experiments + common/{runtime,training,evaluation}.py
├── data_provider/                      # Dataset classes and loaders
├── utils/                              # metrics.py (incl. MDA), forecast_losses.py, visualization.py
├── tools/
│   ├── batch_runner.py, parse_result_to_xlsx.py, select_by_validation.py, benchmark.py, summarize_m4.py
│   ├── baselines/                      # Statistical and ML comparators (classical_baselines.py)
│   ├── data/                           # Crawlers and dataset builders
│   └── stats/                          # Significance tests and descriptive statistics
├── analysis/                           # The paper's tables, figures and reviewer analyses from results/ (run_all.py)
├── tests/                              # CPU contract, parity and end-to-end tests
├── requirements/                       # Split dependency groups
└── docs/EXPERIMENT_WORKFLOW.md         # Protocol, artifacts and TS-Booster details (Traditional Chinese)
```

See [docs/EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md) for the evaluation protocol, identity rules and known differences from the journal code.

---

## Acknowledgements

Built on [Time-Series-Library](https://github.com/thuml/Time-Series-Library) by THUML; the experiment framework follows FinTSLib. Foundation-model support builds on the upstream Chronos, Moirai (uni2ts), TimesFM, TiRex, Sundial and TimeMoE projects.

## License

Released under the MIT License. See [`LICENSE`](./LICENSE).
