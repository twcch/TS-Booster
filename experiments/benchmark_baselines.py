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

DL_RESULTS_PATH = Path("experiments/comprehensive_results/statistical_test_results.json")
RESULTS_PATH = Path("experiments/comprehensive_results/benchmark/benchmark_baselines_results.json")

LOOKBACK = 30
HORIZON = 3
VOL_WINDOW = 7


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

    # Raw volatility series for ARIMA (unscaled)
    vol_series = df["Volatility"].values
    vol_train_end = split_1
    vol_val_end = split_2

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "vol_series": vol_series,
        "vol_train_end": vol_train_end,
        "vol_val_end": vol_val_end,
    }


def compute_rmse(preds, targets, scaler):
    """Inverse-transform volatility predictions and compute RMSE."""
    # scaler was fit on [Volatility, log_return]; volatility is col 0
    preds_inv = preds * scaler.scale_[0] + scaler.mean_[0]
    targets_inv = targets * scaler.scale_[0] + scaler.mean_[0]
    return np.sqrt(mean_squared_error(targets_inv.flatten(), preds_inv.flatten()))


def run_xgboost(data):
    """Train 3 per-step XGBRegressors with early stopping, return RMSE."""
    from xgboost import XGBRegressor

    # Flatten [N, 30, 2] -> [N, 60]
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
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_train, data["y_train"][:, step],
            eval_set=[(X_val, data["y_val"][:, step])],
            verbose=False,
        )
        preds[:, step] = model.predict(X_test)

    return compute_rmse(preds, data["y_test"], data["scaler"])


def run_lightgbm(data):
    """Train 3 per-step LGBMRegressors with early stopping, return RMSE."""
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
            random_state=42,
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

    return compute_rmse(preds, data["y_test"], data["scaler"])


def run_arima(data):
    """Fit auto_arima on train+val volatility, rolling 3-step forecast on test. Return RMSE."""
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

    # We need to align with the sequence-based test set
    # Test sequences start at index 0 of test_feat and go to len-LOOKBACK-HORIZON+1
    # But for ARIMA we forecast the raw volatility series
    # Number of forecast windows = n_test - HORIZON + 1
    n_windows = n_test - HORIZON + 1
    preds = np.zeros((n_windows, HORIZON))

    for i in range(n_windows):
        fc = model.predict(n_periods=HORIZON)
        preds[i] = fc
        # Update model with actual value (rolling)
        model.update(test_vol[i : i + 1])

    # Targets: raw volatility windows
    targets = np.zeros((n_windows, HORIZON))
    for i in range(n_windows):
        targets[i] = test_vol[i : i + HORIZON]

    # ARIMA works on raw scale, so compute RMSE directly (no inverse-transform needed)
    return np.sqrt(mean_squared_error(targets.flatten(), preds.flatten()))


def load_existing_dl_results():
    """Load vanilla/booster RMSE from saved 5-seed results (mean across seeds)."""
    if not DL_RESULTS_PATH.exists():
        print(f"WARNING: {DL_RESULTS_PATH} not found. DL results will be missing.")
        return {}, {}

    with open(DL_RESULTS_PATH) as f:
        raw = json.load(f)

    vanilla_rmse = {}
    booster_rmse = {}
    for ds in ALL_DATASETS:
        if ds in raw:
            vanilla_rmse[ds] = float(np.mean(raw[ds]["vanilla"]))
            booster_rmse[ds] = float(np.mean(raw[ds]["booster"]))

    return vanilla_rmse, booster_rmse


def run_experiments(args):
    """Run all baseline experiments across 16 datasets."""
    # Load any previously saved results for crash recovery
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing results from {RESULTS_PATH}")
    else:
        results = {}

    # Load DL results
    vanilla_rmse, booster_rmse = load_existing_dl_results()

    for ds in ALL_DATASETS:
        if ds not in results:
            results[ds] = {}

        # Store DL results
        if ds in vanilla_rmse:
            results[ds]["Linear"] = vanilla_rmse[ds]
        if ds in booster_rmse:
            results[ds]["Linear+TSB"] = booster_rmse[ds]

        # Skip if all baselines already computed
        needed = ["XGBoost", "LightGBM"]
        if not args.skip_arima:
            needed.append("ARIMA")
        if all(k in results[ds] for k in needed):
            print(f"  {ds:12s} | Already computed, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Dataset: {ds}")
        print(f"{'='*60}")

        data = extract_data(ds)
        if data is None:
            print(f"  SKIP (file not found)")
            continue

        print(f"  Samples: train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}")

        # XGBoost
        if "XGBoost" not in results[ds]:
            t0 = time.time()
            rmse = run_xgboost(data)
            elapsed = time.time() - t0
            results[ds]["XGBoost"] = rmse
            print(f"  XGBoost   : RMSE={rmse:.6f}  ({elapsed:.1f}s)")

        # LightGBM
        if "LightGBM" not in results[ds]:
            t0 = time.time()
            rmse = run_lightgbm(data)
            elapsed = time.time() - t0
            results[ds]["LightGBM"] = rmse
            print(f"  LightGBM  : RMSE={rmse:.6f}  ({elapsed:.1f}s)")

        # ARIMA
        if not args.skip_arima and "ARIMA" not in results[ds]:
            t0 = time.time()
            rmse = run_arima(data)
            elapsed = time.time() - t0
            if rmse is not None:
                results[ds]["ARIMA"] = rmse
                print(f"  ARIMA     : RMSE={rmse:.6f}  ({elapsed:.1f}s)")
            else:
                print(f"  ARIMA     : SKIP (insufficient test data)")

        # DL results summary
        if ds in vanilla_rmse:
            print(f"  Linear    : RMSE={vanilla_rmse[ds]:.6f}  (from 5-seed mean)")
        if ds in booster_rmse:
            print(f"  Linear+TSB: RMSE={booster_rmse[ds]:.6f}  (from 5-seed mean)")

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
    print("BENCHMARK BASELINES: All Methods × 16 Datasets")
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
        row = results[ds]

        line = f"  {ds:<12}"
        rmse_values = {}

        for m in methods:
            if m in row and row[m] is not None:
                line += f" | {row[m]:>12.4f}"
                rmse_values[m] = row[m]
            else:
                line += f" | {'N/A':>12}"

        # TSB vs best external baseline
        external = {k: v for k, v in rmse_values.items() if k not in ("Linear", "Linear+TSB")}
        if external and "Linear+TSB" in rmse_values:
            best_ext = min(external.values())
            gap = (best_ext - rmse_values["Linear+TSB"]) / best_ext * 100
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

    # --- Category breakdown ---
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    for cat_name, cat_datasets in CATEGORIES.items():
        print(f"\n  {cat_name}:")
        cat_means = {m: [] for m in methods}
        for ds in cat_datasets:
            if ds not in results:
                continue
            for m in methods:
                if m in results[ds] and results[ds][m] is not None:
                    cat_means[m].append(results[ds][m])

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
            if m in results[ds] and results[ds][m] is not None:
                available[m] = results[ds][m]
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
    print(r"\caption{RMSE comparison across baselines and TS-Booster on 16 datasets.}")
    print(r"\label{tab:baselines}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(methods) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(methods) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        row = results[ds]
        ds_latex = ds.replace("_", r"\_")

        # Find best (lowest) RMSE for bolding
        available_rmse = {}
        for m in methods:
            if m in row and row[m] is not None:
                available_rmse[m] = row[m]
        best_rmse = min(available_rmse.values()) if available_rmse else None

        cells = []
        for m in methods:
            if m in row and row[m] is not None:
                val = row[m]
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
    for m in methods:
        vals = [results[ds][m] for ds in ALL_DATASETS if ds in results and m in results[ds] and results[ds][m] is not None]
        if vals:
            avg = np.mean(vals)
            avg_cells.append(f"{avg:.4f}")
        else:
            avg_cells.append("--")
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
