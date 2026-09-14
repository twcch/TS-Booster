"""
Benchmark Baselines for TS-Booster Paper.

Compares Linear+TSB against common non-DL baselines (ARIMA, XGBoost, LightGBM)
and vanilla Linear across 16 datasets.

DL results (Linear vanilla & Linear+TSB) are loaded from saved 5-seed results
to avoid redundant GPU training.

Usage:
  uv run python experiments/benchmark_baselines.py              # Full run
  uv run python experiments/benchmark_baselines.py --analyze    # Tables from saved JSON
  uv run python experiments/benchmark_baselines.py --skip-arima # Skip slow ARIMA
"""
import sys
import json
import warnings
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

CATEGORIES = {
    "Taiwan Stocks": ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"],
    "TWD FX": ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD"],
    "International FX": ["EUR_USD", "GBP_USD", "USD_JPY"],
}

DL_RESULTS_PATH = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
RESULTS_PATH = Path("experiments/comprehensive_results/benchmark/benchmark_baselines_results.json")
FIGURES_DIR = Path("experiments/comprehensive_results/benchmark/figures")

LOOKBACK = 30
HORIZON = 3
VOL_WINDOW = 7
SEEDS = [42, 123, 456, 789, 2024]


def extract_data(ds):
    """Load dataset and return train/val/test arrays + scaler, matching prepare_data() splits."""
    path = Path(f"dataset/{ds}.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)
    df["Volatility"] = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["Volatility", "log_return"]
    features = df[feature_cols].values

    n_samples = len(features)
    split_1 = int(n_samples * 0.70)
    split_2 = int(n_samples * 0.80)

    scaler = StandardScaler()
    scaler.fit(features[:split_1])
    features_scaled = scaler.transform(features)

    # Create sequences (same as data_loader.py)
    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback : i + lookback + horizon, 0])  # volatility only
        return np.array(X), np.array(y)

    train_feat = features_scaled[:split_1]
    val_feat = features_scaled[split_1 - LOOKBACK : split_2]
    test_feat = features_scaled[split_2 - LOOKBACK :]

    X_train, y_train = create_sequences(train_feat, LOOKBACK, HORIZON)
    X_val, y_val = create_sequences(val_feat, LOOKBACK, HORIZON)
    X_test, y_test = create_sequences(test_feat, LOOKBACK, HORIZON)

    # Last observation (volatility) for each test sample — for MDA
    last_obs_test = X_test[:, -1, 0]  # scaled volatility at t

    # Raw volatility series for ARIMA (unscaled)
    vol_series = df["Volatility"].values
    vol_train_end = split_1
    vol_val_end = split_2

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "last_obs_test": last_obs_test,
        "scaler": scaler,
        "vol_series": vol_series,
        "vol_train_end": vol_train_end,
        "vol_val_end": vol_val_end,
    }


def compute_metrics(preds, targets, scaler, last_obs=None):
    """Inverse-transform volatility predictions and compute RMSE, MAE, MDA."""
    preds_inv = preds * scaler.scale_[0] + scaler.mean_[0]
    targets_inv = targets * scaler.scale_[0] + scaler.mean_[0]
    rmse = float(np.sqrt(mean_squared_error(targets_inv.flatten(), preds_inv.flatten())))
    mae = float(np.mean(np.abs(targets_inv.flatten() - preds_inv.flatten())))
    # MDA: compare direction vs last observation
    if last_obs is not None:
        last_obs_inv = last_obs * scaler.scale_[0] + scaler.mean_[0]
        pred_dir = np.sign(preds_inv - last_obs_inv[:, None])
        actual_dir = np.sign(targets_inv - last_obs_inv[:, None])
        mda = float(np.mean(pred_dir == actual_dir))
    else:
        mda = None
    return {"rmse": rmse, "mae": mae, "mda": mda}


def compute_metrics_raw(preds, targets, last_obs=None):
    """Compute RMSE, MAE, MDA on raw (unscaled) values."""
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    if last_obs is not None:
        pred_dir = np.sign(preds - last_obs[:, None])
        actual_dir = np.sign(targets - last_obs[:, None])
        mda = float(np.mean(pred_dir == actual_dir))
    else:
        mda = None
    return {"rmse": rmse, "mae": mae, "mda": mda}


def run_xgboost(data, seed=42):
    """Train 3 per-step XGBRegressors with early stopping, return metrics dict."""
    from xgboost import XGBRegressor

    X_train = data["X_train"].reshape(len(data["X_train"]), -1)
    X_val = data["X_val"].reshape(len(data["X_val"]), -1)
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)

    preds = np.zeros_like(data["y_test"])

    for step in range(HORIZON):
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=30,
            random_state=seed,
            verbosity=0,
        )
        model.fit(
            X_train, data["y_train"][:, step],
            eval_set=[(X_val, data["y_val"][:, step])],
            verbose=False,
        )
        preds[:, step] = model.predict(X_test)

    return compute_metrics(preds, data["y_test"], data["scaler"], data["last_obs_test"])


def run_lightgbm(data, seed=42):
    """Train 3 per-step LGBMRegressors with early stopping, return metrics dict."""
    from lightgbm import LGBMRegressor

    X_train = data["X_train"].reshape(len(data["X_train"]), -1)
    X_val = data["X_val"].reshape(len(data["X_val"]), -1)
    X_test = data["X_test"].reshape(len(data["X_test"]), -1)

    preds = np.zeros_like(data["y_test"])

    for step in range(HORIZON):
        model = LGBMRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        )
        model.fit(
            X_train, data["y_train"][:, step],
            eval_set=[(X_val, data["y_val"][:, step])],
            callbacks=[
                __import__("lightgbm").early_stopping(30, verbose=False),
                __import__("lightgbm").log_evaluation(0),
            ],
        )
        preds[:, step] = model.predict(X_test)

    return compute_metrics(preds, data["y_test"], data["scaler"], data["last_obs_test"])


def run_arima(data):
    """Fit auto_arima on train+val volatility, rolling 3-step forecast on test. Return metrics dict."""
    from pmdarima import auto_arima

    vol = data["vol_series"]
    train_val_end = data["vol_val_end"]

    # Fit on train+val portion
    train_val_vol = vol[:train_val_end]
    test_vol = vol[train_val_end:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = auto_arima(
            train_val_vol,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_order=10,
            max_p=5, max_q=5,
        )

    # Rolling forecast: predict HORIZON steps, then update with actual value
    n_test = len(test_vol)
    if n_test < HORIZON:
        return None

    n_windows = n_test - HORIZON + 1
    preds = np.zeros((n_windows, HORIZON))

    for i in range(n_windows):
        fc = model.predict(n_periods=HORIZON)
        preds[i] = fc
        model.update(test_vol[i : i + 1])

    # Targets: raw volatility windows
    targets = np.zeros((n_windows, HORIZON))
    for i in range(n_windows):
        targets[i] = test_vol[i : i + HORIZON]

    # Last observation for MDA (raw scale): for window i the forecasts target
    # test_vol[i : i+HORIZON], so the last known value is vol[train_val_end + i - 1]
    # (test_vol[i] itself is the h=1 target and must not be used as the anchor).
    last_obs = np.array([vol[train_val_end + i - 1] for i in range(n_windows)]) if n_windows > 0 else None

    # ARIMA works on raw scale
    return compute_metrics_raw(preds, targets, last_obs)


def load_existing_dl_results():
    """Load vanilla/booster metrics from saved 5-seed results."""
    if not DL_RESULTS_PATH.exists():
        print(f"WARNING: {DL_RESULTS_PATH} not found. DL results will be missing.")
        return {}, {}

    with open(DL_RESULTS_PATH) as f:
        raw = json.load(f)

    vanilla_metrics = {}
    booster_metrics = {}
    for ds in ALL_DATASETS:
        if ds in raw:
            v = raw[ds]["vanilla"]
            b = raw[ds]["booster"]
            if isinstance(v, dict):
                vanilla_metrics[ds] = {k: v[k] for k in ["rmse", "mae", "mda"] if k in v}
            else:
                vanilla_metrics[ds] = {"rmse": v}
            if isinstance(b, dict):
                booster_metrics[ds] = {k: b[k] for k in ["rmse", "mae", "mda"] if k in b}
            else:
                booster_metrics[ds] = {"rmse": b}

    return vanilla_metrics, booster_metrics


def _get_metric_mean(results, ds, method, metric="rmse"):
    """Get mean of a metric, handling both old (scalar) and new (dict) formats."""
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        arr = val.get(metric, [])
        if not arr:
            return None
        return float(np.mean(arr))
    if metric == "rmse":
        return float(val) if not isinstance(val, list) else float(np.mean(val))
    return None


def _get_metric_list(results, ds, method, metric="rmse"):
    """Get list of a metric."""
    val = results.get(ds, {}).get(method)
    if val is None:
        return None
    if isinstance(val, dict):
        return val.get(metric, None)
    if metric == "rmse":
        return val if isinstance(val, list) else [val]
    return None


def run_experiments(args):
    """Run all baseline experiments across 16 datasets × 5 seeds."""
    # Load existing results to preserve ARIMA when --skip-arima
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}

    # Load DL results (already 5-seed arrays)
    vanilla_metrics, booster_metrics = load_existing_dl_results()

    for ds in ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")

        # Preserve existing ARIMA results when --skip-arima
        existing_arima = None
        if args.skip_arima and ds in results and "ARIMA" in results[ds]:
            existing_arima = results[ds]["ARIMA"]

        # Store DL results
        results[ds] = {}
        if ds in vanilla_metrics:
            results[ds]["Linear"] = vanilla_metrics[ds]
        if ds in booster_metrics:
            results[ds]["Linear+TSB"] = booster_metrics[ds]

        # Restore preserved ARIMA
        if existing_arima is not None:
            results[ds]["ARIMA"] = existing_arima

        data = extract_data(ds)
        if data is None:
            print(f"  SKIP (file not found)")
            continue

        print(f"  Samples: train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}")

        # XGBoost — 5 seeds
        t0 = time.time()
        xgb_metrics = {"rmse": [], "mae": [], "mda": []}
        for seed in SEEDS:
            m = run_xgboost(data, seed=seed)
            for k in xgb_metrics:
                xgb_metrics[k].append(m[k])
        results[ds]["XGBoost"] = xgb_metrics
        elapsed = time.time() - t0
        print(f"  XGBoost    : RMSE={np.mean(xgb_metrics['rmse']):.4f} MAE={np.mean(xgb_metrics['mae']):.4f} MDA={np.mean(xgb_metrics['mda']):.4f}  ({elapsed:.1f}s)")

        # LightGBM — 5 seeds
        t0 = time.time()
        lgb_metrics = {"rmse": [], "mae": [], "mda": []}
        for seed in SEEDS:
            m = run_lightgbm(data, seed=seed)
            for k in lgb_metrics:
                lgb_metrics[k].append(m[k])
        results[ds]["LightGBM"] = lgb_metrics
        elapsed = time.time() - t0
        print(f"  LightGBM   : RMSE={np.mean(lgb_metrics['rmse']):.4f} MAE={np.mean(lgb_metrics['mae']):.4f} MDA={np.mean(lgb_metrics['mda']):.4f}  ({elapsed:.1f}s)")

        # ARIMA — deterministic (1 run)
        if not args.skip_arima:
            t0 = time.time()
            arima_m = run_arima(data)
            elapsed = time.time() - t0
            if arima_m is not None:
                results[ds]["ARIMA"] = {"rmse": [arima_m["rmse"]], "mae": [arima_m["mae"]], "mda": [arima_m["mda"]]}
                print(f"  ARIMA      : RMSE={arima_m['rmse']:.4f} MAE={arima_m['mae']:.4f} MDA={arima_m['mda']:.4f}  ({elapsed:.1f}s)")
            else:
                print(f"  ARIMA      : SKIP (insufficient test data)")

        # DL summary
        if ds in vanilla_metrics:
            print(f"  Linear     : RMSE={np.mean(vanilla_metrics[ds]['rmse']):.4f}  (5 seeds)")
        if ds in booster_metrics:
            print(f"  Linear+TSB : RMSE={np.mean(booster_metrics[ds]['rmse']):.4f}  (5 seeds)")

        # Incremental save
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


def analyze(results):
    """Print terminal table, category breakdown, and LaTeX table."""
    methods = ["ARIMA", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]

    print("\n" + "=" * 120)
    print("BENCHMARK BASELINES: All Methods × 16 Datasets (5 seeds where applicable)")
    print("=" * 120)

    # --- Terminal table ---
    header = f"{'Dataset':<12}"
    for m in methods:
        header += f" | {m:>12}"
    header += f" | {'TSB vs Best':>12}"
    print(f"\n{header}")
    print("-" * len(header))

    tsb_vs_best_gaps = []

    for ds in ALL_DATASETS:
        if ds not in results:
            continue

        line = f"  {ds:<12}"
        rmse_means = {}

        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                line += f" | {val:>12.4f}"
                rmse_means[m] = val
            else:
                line += f" | {'N/A':>12}"

        # TSB vs best external baseline
        external = {k: v for k, v in rmse_means.items() if k not in ("Linear", "Linear+TSB")}
        if external and "Linear+TSB" in rmse_means:
            best_ext = min(external.values())
            gap = (best_ext - rmse_means["Linear+TSB"]) / best_ext * 100
            tsb_vs_best_gaps.append(gap)
            tag = "WIN" if gap > 0 else "LOSS"
            line += f" | {gap:>+10.2f}% {tag}"
        else:
            line += f" | {'N/A':>12}"

        print(line)

    print("-" * len(header))
    if tsb_vs_best_gaps:
        wins = sum(1 for g in tsb_vs_best_gaps if g > 0)
        print(f"  TSB vs Best External Baseline: {wins}/{len(tsb_vs_best_gaps)} wins, avg gap = {np.mean(tsb_vs_best_gaps):+.2f}%")

    # --- Pairwise wins ---
    print(f"\n  TSB vs each method (RMSE wins):")
    for m in methods:
        if m == "Linear+TSB":
            continue
        wins, total = 0, 0
        for ds in ALL_DATASETS:
            tsb_val = _get_metric_mean(results, ds, "Linear+TSB")
            other_val = _get_metric_mean(results, ds, m)
            if tsb_val is not None and other_val is not None:
                total += 1
                if tsb_val < other_val:
                    wins += 1
        if total > 0:
            print(f"    vs {m:<12}: {wins}/{total} ({wins/total*100:.0f}%)")

    # --- Category breakdown ---
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    for cat_name, cat_datasets in CATEGORIES.items():
        print(f"\n  {cat_name}:")
        cat_means = {m: [] for m in methods}
        for ds in cat_datasets:
            for m in methods:
                val = _get_metric_mean(results, ds, m)
                if val is not None:
                    cat_means[m].append(val)

        line = "    Avg RMSE:"
        for m in methods:
            if cat_means[m]:
                line += f"  {m}={np.mean(cat_means[m]):.4f}"
        print(line)

    # --- Rank analysis ---
    print(f"\n{'='*80}")
    print("AVERAGE RANK ACROSS DATASETS")
    print("=" * 80)

    ranks_per_method = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        available = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available[m] = val
        if not available:
            continue
        sorted_methods = sorted(available, key=lambda k: available[k])
        for rank, m in enumerate(sorted_methods, 1):
            ranks_per_method[m].append(rank)

    for m in methods:
        if ranks_per_method[m]:
            avg_rank = np.mean(ranks_per_method[m])
            print(f"  {m:<12}: avg rank = {avg_rank:.2f}  (ranks: {ranks_per_method[m]})")

    # --- LaTeX table ---
    print(f"\n{'='*80}")
    print("LATEX TABLE (for paper)")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{RMSE comparison across baselines and TS-Booster on 16 datasets (5-seed mean).}")
    print(r"\label{tab:baselines}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(methods) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        ds_latex = ds.replace("_", r"\_")

        available_rmse = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available_rmse[m] = val
        best_rmse = min(available_rmse.values()) if available_rmse else None

        cells = []
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                if best_rmse is not None and abs(val - best_rmse) < 1e-8:
                    cells.append(r"\textbf{" + f"{val:.4f}" + "}")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("--")

        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    # Average row
    print(r"\midrule")
    avg_cells = []
    avg_rmse = {}
    for m in methods:
        vals = [_get_metric_mean(results, ds, m) for ds in ALL_DATASETS]
        vals = [v for v in vals if v is not None]
        if vals:
            avg = np.mean(vals)
            avg_rmse[m] = avg
            avg_cells.append(f"{avg:.4f}")
        else:
            avg_cells.append("--")
    if avg_rmse:
        best_avg = min(avg_rmse.values())
        avg_cells_final = []
        for m, c in zip(methods, avg_cells):
            if m in avg_rmse and abs(avg_rmse[m] - best_avg) < 1e-8:
                avg_cells_final.append(r"\textbf{" + c + "}")
            else:
                avg_cells_final.append(c)
        avg_cells = avg_cells_final
    print(r"\textit{Average} & " + " & ".join(avg_cells) + r" \\")

    # Average rank row
    rank_cells = []
    for m in methods:
        if ranks_per_method[m]:
            rank_cells.append(f"{np.mean(ranks_per_method[m]):.2f}")
        else:
            rank_cells.append("--")
    print(r"\textit{Avg. Rank} & " + " & ".join(rank_cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
               "EUR_USD", "GBP_USD", "USD_JPY"]


def plot_tsb_vs_best_baseline(results):
    """Horizontal bar: TSB RMSE improvement % over best external baseline per dataset."""
    methods_ext = ["ARIMA", "XGBoost", "LightGBM"]

    datasets, diffs, colors, best_names = [], [], [], []
    for ds in ALL_DATASETS:
        tsb = _get_metric_mean(results, ds, "Linear+TSB")
        if tsb is None:
            continue
        ext_rmse = {}
        for m in methods_ext:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                ext_rmse[m] = val
        if not ext_rmse:
            continue
        best_m = min(ext_rmse, key=lambda k: ext_rmse[k])
        best_val = ext_rmse[best_m]
        gap = (best_val - tsb) / best_val * 100  # positive = TSB wins
        datasets.append(ds)
        diffs.append(gap)
        colors.append('#43A047' if gap > 0 else '#E53935')
        best_names.append(best_m)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(datasets))
    bars = ax.barh(y, diffs, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # All labels outside bars
    for i, (bar, diff, bm) in enumerate(zip(bars, diffs, best_names)):
        w = bar.get_width()
        label = f'{diff:+.1f}%'
        if w >= 0:
            ax.text(w + 0.4, i, label,
                    ha='left', va='center', fontsize=9, fontweight='bold')
        else:
            ax.text(w - 0.4, i, label,
                    ha='right', va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='#424242', linewidth=1)

    # FX / Stocks separator
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axhline(y=n_fx - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.5)

    wins = sum(1 for d in diffs if d > 0)
    # Add enough padding for labels
    max_val = max(diffs) if diffs else 10
    min_val = min(diffs) if diffs else -10
    ax.set_xlim(min_val - 5, max_val + 6)

    ax.set_xlabel('RMSE Improvement over Best External Baseline (%)', fontsize=11)
    ax.set_title(f'Linear+TSB vs Best Baseline: RMSE Comparison ({wins}/{len(diffs)} wins)',
                 fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'tsb_vs_best_baseline.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: tsb_vs_best_baseline.png")


def plot_average_rank(results):
    """Bar chart: average rank of each method across 16 datasets."""
    methods = ["ARIMA", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]
    method_colors = ["#9E9E9E", "#FF7043", "#66BB6A", "#42A5F5", "#AB47BC"]

    ranks_per_method = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        available = {}
        for m in methods:
            val = _get_metric_mean(results, ds, m)
            if val is not None:
                available[m] = val
        if not available:
            continue
        sorted_methods = sorted(available, key=lambda k: available[k])
        for rank, m in enumerate(sorted_methods, 1):
            ranks_per_method[m].append(rank)

    avg_ranks = []
    labels = []
    bar_colors = []
    for m, c in zip(methods, method_colors):
        if ranks_per_method[m]:
            avg_ranks.append(np.mean(ranks_per_method[m]))
            labels.append(m)
            bar_colors.append(c)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_ranks, color=bar_colors, edgecolor='#37474F', linewidth=0.8, width=0.6)

    for bar, rank in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Average Rank (lower = better)', fontsize=11)
    ax.set_title('Average RMSE Rank Across 16 Datasets', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(avg_ranks) + 0.8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'average_rank.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: average_rank.png")


def plot_metric_comparison(results, metric, higher_is_better=False):
    """Grouped bar chart: all methods × 16 datasets for a given metric."""
    methods = ["ARIMA", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]
    method_colors = ["#9E9E9E", "#FF7043", "#66BB6A", "#42A5F5", "#AB47BC"]

    datasets_with_data = []
    for ds in ALL_DATASETS:
        has_any = any(_get_metric_mean(results, ds, m, metric) is not None for m in methods)
        if has_any:
            datasets_with_data.append(ds)

    if not datasets_with_data:
        print(f"  SKIP {metric} chart (no data)")
        return

    n_ds = len(datasets_with_data)
    n_methods = len(methods)
    bar_width = 0.15
    x = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(max(14, n_ds * 0.9), 6))

    # Find best method per dataset
    best_per_ds = {}
    for i, ds in enumerate(datasets_with_data):
        ds_vals = {}
        for m in methods:
            v = _get_metric_mean(results, ds, m, metric)
            if v is not None:
                ds_vals[m] = v
        if ds_vals:
            best_per_ds[i] = max(ds_vals, key=lambda k: ds_vals[k]) if higher_is_better else min(ds_vals, key=lambda k: ds_vals[k])

    bar_containers = []
    for j, (m, color) in enumerate(zip(methods, method_colors)):
        vals = []
        for ds in datasets_with_data:
            v = _get_metric_mean(results, ds, m, metric)
            vals.append(v if v is not None else 0)
        offset = (j - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=m, color=color, edgecolor='white', linewidth=0.5)
        bar_containers.append((m, bars))

    # Dim non-best bars
    for m, bars in bar_containers:
        for i, bar in enumerate(bars):
            if best_per_ds.get(i) != m:
                bar.set_alpha(0.25)

    metric_upper = metric.upper()
    better = "higher" if higher_is_better else "lower"
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel(f'{metric_upper} ({better} is better)', fontsize=11)
    ax.set_title(f'{metric_upper} Comparison: All Methods × 16 Datasets', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_with_data, fontsize=8, rotation=45, ha='right')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'benchmark_{metric}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: benchmark_{metric}.png")


def generate_figures(results):
    """Generate all benchmark figures."""
    print("\nGenerating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_tsb_vs_best_baseline(results)
    plot_average_rank(results)
    plot_metric_comparison(results, "rmse", higher_is_better=False)
    plot_metric_comparison(results, "mae", higher_is_better=False)
    plot_metric_comparison(results, "mda", higher_is_better=True)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark baselines for TS-Booster paper")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    parser.add_argument("--skip-arima", action="store_true", help="Skip slow ARIMA fitting")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments(args)

    analyze(results)
    generate_figures(results)
