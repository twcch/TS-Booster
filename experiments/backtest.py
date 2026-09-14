"""
Backtesting Experiment: Inverse-Volatility Position Sizing.

Uses predicted volatility to size positions via inverse-vol weighting
(Moreira & Muir, 2017), then compares Sharpe ratios across vol-based methods.

Core claim: better vol prediction (lower RMSE) → better position sizing → higher Sharpe.

Methods: Historical Vol, XGBoost, LightGBM, Linear, Linear+TSB

Usage:
  uv run python experiments/backtest.py              # Full run (single seed)
  uv run python experiments/backtest.py --analyze    # Tables from saved JSON only
"""
import sys
import json
import random
import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data

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

METHODS = ["Historical Vol", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]

RESULTS_PATH = Path("experiments/comprehensive_results/backtest/backtest_results.json")
DAILY_RETURNS_PATH = Path("experiments/comprehensive_results/backtest/backtest_daily_returns.json")
FIGURES_DIR = Path("experiments/comprehensive_results/backtest/figures")
SEED = 42
LOOKBACK = 30
HORIZON = 3
VOL_WINDOW = 7
TARGET_VOL = 15.0  # annualized %
WEIGHT_FLOOR = 0.2
WEIGHT_CAP = 3.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────
# Data Preparation
# ──────────────────────────────────────────────────────────────

def prepare_backtest_data(ds):
    """Load dataset and return loaders + scaler + raw test returns + tree-model arrays.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for DL models
        scaler: StandardScaler (fit on train)
        test_returns: [n_test, 3] raw log returns for backtesting
        tree_data: dict with X_train/y_train/X_val/y_val/X_test/y_test arrays for tree models
    """
    path = Path(f"dataset/{ds}.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # DL loaders (reuse existing prepare_data)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=LOOKBACK, horizon=HORIZON, batch_size=32
    )

    # Collect test returns from the loader
    test_returns = []
    for batch in test_loader:
        test_returns.append(batch["return"].numpy())
    test_returns = np.concatenate(test_returns, axis=0)  # [n_test, 3]

    # Build tree-model arrays (same splits as prepare_data)
    df2 = df.copy()
    df2["log_return"] = np.log(df2["Close"] / df2["Close"].shift(1))
    df2["log_return"] = df2["log_return"].fillna(0.0)
    df2["Volatility"] = df2["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    df2 = df2.dropna().reset_index(drop=True)

    features = df2[["Volatility", "log_return"]].values
    n_samples = len(features)
    split_1 = int(n_samples * 0.70)
    split_2 = int(n_samples * 0.80)

    tree_scaler = StandardScaler()
    tree_scaler.fit(features[:split_1])
    features_scaled = tree_scaler.transform(features)

    train_feat = features_scaled[:split_1]
    val_feat = features_scaled[split_1 - LOOKBACK: split_2]
    test_feat = features_scaled[split_2 - LOOKBACK:]

    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i: i + lookback])
            y.append(data[i + lookback: i + lookback + horizon, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_feat, LOOKBACK, HORIZON)
    X_val, y_val = create_sequences(val_feat, LOOKBACK, HORIZON)
    X_test, y_test = create_sequences(test_feat, LOOKBACK, HORIZON)

    tree_data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": tree_scaler,
    }

    return train_loader, val_loader, test_loader, scaler, test_returns, tree_data


# ──────────────────────────────────────────────────────────────
# Prediction Generation
# ──────────────────────────────────────────────────────────────

def generate_dl_predictions(train_loader, val_loader, test_loader, scaler, model_type="vanilla"):
    """Train a DL model and return [n_test, 3] predictions in real scale (annualized % vol)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = 100
    config['input_channels'] = 2
    seq_len, pred_len = LOOKBACK, HORIZON

    base_kwargs = dict(
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
    )

    set_seed(SEED)

    if model_type == "vanilla":
        model = FlexibleTSBooster(
            base_model=LinearBackbone(seq_len, 2, pred_len),
            use_decomp=False, use_experts=False, **base_kwargs
        )
    else:  # booster
        model = FlexibleTSBooster(
            base_model=LinearBackbone(seq_len, 2, pred_len),
            use_decomp=True, use_experts=True,
            expert_gate='snr', enrich_mode='trend',
            **base_kwargs
        )

    model = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=model
    )

    # Collect predictions
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds_list.append(pred.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)  # [n_test, 3] scaled

    # Inverse transform to real scale
    preds_real = preds * scaler.scale_[0] + scaler.mean_[0]
    return preds_real


def generate_tree_predictions(tree_data, method="xgboost"):
    """Train XGBoost or LightGBM and return [n_test, 3] predictions in real scale."""
    scaler = tree_data["scaler"]

    X_train = tree_data["X_train"].reshape(len(tree_data["X_train"]), -1)
    X_val = tree_data["X_val"].reshape(len(tree_data["X_val"]), -1)
    X_test = tree_data["X_test"].reshape(len(tree_data["X_test"]), -1)

    preds = np.zeros_like(tree_data["y_test"])  # [n_test, 3]

    for step in range(HORIZON):
        if method == "xgboost":
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=30, random_state=42, verbosity=0,
            )
            model.fit(
                X_train, tree_data["y_train"][:, step],
                eval_set=[(X_val, tree_data["y_val"][:, step])],
                verbose=False,
            )
        else:  # lightgbm
            from lightgbm import LGBMRegressor
            import lightgbm as lgb
            model = LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
            model.fit(
                X_train, tree_data["y_train"][:, step],
                eval_set=[(X_val, tree_data["y_val"][:, step])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )

        preds[:, step] = model.predict(X_test)

    # Inverse transform to real scale
    preds_real = preds * scaler.scale_[0] + scaler.mean_[0]
    return preds_real


def generate_historical_vol(tree_data):
    """Naive persistence: last known vol in lookback window as next-day prediction.

    Returns [n_test, 3] predictions in real scale (same value repeated for all 3 steps).
    """
    scaler = tree_data["scaler"]
    X_test = tree_data["X_test"]  # [n_test, 30, 2]

    # Last known vol (col 0) in each lookback window, inverse-transformed
    last_vol_scaled = X_test[:, -1, 0]  # [n_test]
    last_vol_real = last_vol_scaled * scaler.scale_[0] + scaler.mean_[0]

    # Repeat for 3 steps (persistence assumption)
    preds_real = np.column_stack([last_vol_real] * HORIZON)  # [n_test, 3]
    return preds_real


# ──────────────────────────────────────────────────────────────
# Strategy Logic
# ──────────────────────────────────────────────────────────────

def run_inv_vol_strategy(pred_vol, returns, target_vol=TARGET_VOL):
    """Inverse volatility position sizing.

    Args:
        pred_vol: [n_test] predicted vol (annualized %, real scale)
        returns: [n_test] actual daily log returns (raw, unscaled)
        target_vol: target annualized vol (%)

    Returns:
        strategy_returns: [n_test] daily strategy simple returns
    """
    # Ensure positive vol predictions (clip small/negative values)
    pred_vol = np.clip(pred_vol, 0.1, None)

    weight = np.clip(target_vol / pred_vol, WEIGHT_FLOOR, WEIGHT_CAP)
    # Convert log returns to simple returns before scaling by the position
    # weight: w * (e^r - 1) is the actual P&L of holding weight w for one day,
    # and is the quantity that compounds as prod(1 + r_s) in NAV/drawdown.
    strategy_returns = weight * np.expm1(returns)
    return strategy_returns


def compute_strategy_metrics(strategy_returns, annual_factor=252):
    """Compute financial metrics for a strategy.

    Returns dict with: ann_return, ann_vol, sharpe, max_drawdown, calmar
    """
    ann_return = np.mean(strategy_returns) * annual_factor
    ann_vol = np.std(strategy_returns) * np.sqrt(annual_factor)
    sharpe = ann_return / ann_vol if ann_vol > 1e-9 else 0.0

    # Max drawdown
    cumulative = np.cumprod(1 + strategy_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = np.min(drawdown)

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    vol_target_error = abs(ann_vol * 100 - TARGET_VOL) / TARGET_VOL  # as fraction

    return {
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "vol_target_error": float(vol_target_error),
    }


# ──────────────────────────────────────────────────────────────
# Main Experiment Loop
# ──────────────────────────────────────────────────────────────

def run_all_experiments():
    """Run backtest across 16 datasets x 5 methods. Saves incrementally."""
    # Load existing results (check format compatibility)
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        first_ds = next(iter(results.values()), {})
        first_method = next(iter(first_ds.values()), {})
        if "strategy" in first_method:
            print("Old nested format detected, starting fresh...")
            results = {}
        else:
            print(f"Loaded existing results from {RESULTS_PATH}")
    else:
        results = {}

    # Load existing daily returns
    if DAILY_RETURNS_PATH.exists():
        with open(DAILY_RETURNS_PATH) as f:
            daily_data = json.load(f)
        print(f"Loaded existing daily returns from {DAILY_RETURNS_PATH}")
    else:
        daily_data = {}

    for ds in ALL_DATASETS:
        # Skip only if BOTH summary and daily returns exist
        has_summary = ds in results and all(m in results[ds] for m in METHODS)
        has_daily = ds in daily_data and all(m in daily_data[ds] for m in METHODS)
        if has_summary and has_daily:
            print(f"  {ds:12s} | Already computed, skipping")
            continue

        print(f"\n{'=' * 70}")
        print(f"  Dataset: {ds}")
        print(f"{'=' * 70}")

        data = prepare_backtest_data(ds)
        if data is None:
            print(f"  SKIP (file not found)")
            continue

        train_loader, val_loader, test_loader, scaler, test_returns, tree_data = data
        n_test = len(test_returns)
        print(f"  Test samples: {n_test}")

        # Daily returns (step 0 of 3-step forecast)
        daily_returns = test_returns[:, 0]  # [n_test]

        if ds not in results:
            results[ds] = {}
        if ds not in daily_data:
            daily_data[ds] = {}
        daily_data[ds]["actual_returns"] = daily_returns.tolist()

        def compute_and_store(method_name, pred_vol):
            """Compute strategy metrics from predictions and store."""
            strat_ret = run_inv_vol_strategy(pred_vol, daily_returns)
            strat_metrics = compute_strategy_metrics(strat_ret)
            results[ds][method_name] = strat_metrics
            daily_data[ds][method_name] = strat_ret.tolist()
            return strat_metrics

        # 1. Historical Vol
        if "Historical Vol" not in daily_data[ds]:
            t0 = time.time()
            hist_preds = generate_historical_vol(tree_data)  # [n_test, 3]
            sm = compute_and_store("Historical Vol", hist_preds[:, 0])
            print(f"  Historical Vol| Sharpe={sm['sharpe']:>7.3f} ({time.time()-t0:.1f}s)")

        # 2. XGBoost
        if "XGBoost" not in daily_data[ds]:
            t0 = time.time()
            xgb_preds = generate_tree_predictions(tree_data, method="xgboost")
            sm = compute_and_store("XGBoost", xgb_preds[:, 0])
            print(f"  XGBoost       | Sharpe={sm['sharpe']:>7.3f} ({time.time()-t0:.1f}s)")

        # 3. LightGBM
        if "LightGBM" not in daily_data[ds]:
            t0 = time.time()
            lgb_preds = generate_tree_predictions(tree_data, method="lightgbm")
            sm = compute_and_store("LightGBM", lgb_preds[:, 0])
            print(f"  LightGBM      | Sharpe={sm['sharpe']:>7.3f} ({time.time()-t0:.1f}s)")

        # 4. Linear (vanilla)
        if "Linear" not in daily_data[ds]:
            t0 = time.time()
            vanilla_preds = generate_dl_predictions(
                train_loader, val_loader, test_loader, scaler, model_type="vanilla"
            )
            sm = compute_and_store("Linear", vanilla_preds[:, 0])
            print(f"  Linear        | Sharpe={sm['sharpe']:>7.3f} ({time.time()-t0:.1f}s)")

        # 5. Linear+TSB (TS-Booster)
        if "Linear+TSB" not in daily_data[ds]:
            t0 = time.time()
            booster_preds = generate_dl_predictions(
                train_loader, val_loader, test_loader, scaler, model_type="booster"
            )
            sm = compute_and_store("Linear+TSB", booster_preds[:, 0])
            print(f"  Linear+TSB    | Sharpe={sm['sharpe']:>7.3f} ({time.time()-t0:.1f}s)")

        # Incremental save
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        with open(DAILY_RETURNS_PATH, 'w') as f:
            json.dump(daily_data, f)

    print(f"\nAll results saved to {RESULTS_PATH}")
    print(f"Daily returns saved to {DAILY_RETURNS_PATH}")
    return results, daily_data


# ──────────────────────────────────────────────────────────────
# Analysis & Tables
# ──────────────────────────────────────────────────────────────

def analyze(results):
    """Print terminal table + LaTeX table for inverse-vol Sharpe comparison."""
    n_ds = len(ALL_DATASETS)

    # Helper: extract strategy metrics (flat format)
    def get_strat(ds_results, method):
        if method not in ds_results:
            return None
        return ds_results[method]

    short_labels = {
        "Historical Vol": "HistVol",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "Linear": "Linear",
        "Linear+TSB": "Lin+TSB",
    }

    latex_labels = {
        "Historical Vol": "Hist.\\ Vol",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "Linear": "Linear",
        "Linear+TSB": "Linear+TSB",
    }

    # ═══════════════════════════════════════════════════════════
    # SHARPE RATIO TABLE
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("INVERSE-VOLATILITY POSITION SIZING: SHARPE RATIO COMPARISON")
    print("  weight = target_vol / predicted_vol (capped [0.2, 3.0]), target_vol = 15%")
    print("=" * 120)

    col_w = 10
    header = f"  {'Dataset':<12}"
    for m in METHODS:
        header += f" | {short_labels[m]:>{col_w}}"
    header += f" | {'TSB>Lin':>7}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    sharpe_by_method = {m: [] for m in METHODS}
    tsb_best_count = 0
    tsb_vs_linear_wins = 0
    n_compared = 0

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        line = f"  {ds:<12}"

        sharpes = {}
        for m in METHODS:
            s_data = get_strat(results[ds], m)
            if s_data:
                s = s_data["sharpe"]
                sharpes[m] = s
                sharpe_by_method[m].append(s)
                line += f" | {s:>{col_w}.3f}"
            else:
                line += f" | {'N/A':>{col_w}}"

        # Global best?
        if sharpes:
            best_method = max(sharpes, key=lambda k: sharpes[k])
            if best_method == "Linear+TSB":
                tsb_best_count += 1

        # TSB vs Linear (paired)
        tsb_win = ""
        if "Linear+TSB" in sharpes and "Linear" in sharpes:
            n_compared += 1
            if sharpes["Linear+TSB"] > sharpes["Linear"]:
                tsb_vs_linear_wins += 1
                tsb_win = "+"
            else:
                tsb_win = "-"
        line += f" | {tsb_win:>7}"
        print(line)

    print("  " + "-" * (len(header) - 2))

    # Average row
    line = f"  {'Average':<12}"
    for m in METHODS:
        if sharpe_by_method[m]:
            avg = np.mean(sharpe_by_method[m])
            line += f" | {avg:>{col_w}.3f}"
        else:
            line += f" | {'N/A':>{col_w}}"
    line += f" | {tsb_vs_linear_wins:>4d}/{n_compared}"
    print(line)

    # Global best count
    best_counts = {m: 0 for m in METHODS}
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        sharpes = {}
        for m in METHODS:
            s_data = get_strat(results[ds], m)
            if s_data:
                sharpes[m] = s_data["sharpe"]
        if sharpes:
            best_val = max(sharpes.values())
            for m in METHODS:
                if m in sharpes and abs(sharpes[m] - best_val) < 1e-8:
                    best_counts[m] += 1

    line = f"  {'Best #':<12}"
    for m in METHODS:
        line += f" | {best_counts[m]:>{col_w - 3}d}/{n_ds}"
    line += f" |"
    print(line)

    # Pairwise wins
    print(f"\n  TSB vs each method (Sharpe wins):")
    for m in METHODS:
        if m == "Linear+TSB":
            continue
        wins, total = 0, 0
        for ds in ALL_DATASETS:
            s_tsb = get_strat(results.get(ds, {}), "Linear+TSB")
            s_other = get_strat(results.get(ds, {}), m)
            if s_tsb and s_other:
                total += 1
                if s_tsb["sharpe"] > s_other["sharpe"]:
                    wins += 1
        if total > 0:
            print(f"    vs {m:<15}: {wins}/{total} ({wins/total*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # CATEGORY BREAKDOWN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("CATEGORY BREAKDOWN (Average Sharpe)")
    print("=" * 100)

    for cat_name, cat_datasets in CATEGORIES.items():
        print(f"\n  {cat_name}:")
        line = "    "
        for m in METHODS:
            cat_sharpes = []
            for ds in cat_datasets:
                s_data = get_strat(results.get(ds, {}), m)
                if s_data:
                    cat_sharpes.append(s_data["sharpe"])
            if cat_sharpes:
                line += f"  {short_labels[m]}={np.mean(cat_sharpes):.3f}"
        print(line)

    # ═══════════════════════════════════════════════════════════
    # LATEX TABLE
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print("LATEX TABLE: Inverse-Volatility Strategy — Sharpe Ratio")
    print("=" * 100)
    print()

    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Sharpe ratio of inverse-volatility position sizing "
          r"(Moreira \& Muir, 2017) across vol-based methods. "
          r"Position weight $= \min(\max(\sigma_{\text{target}} / \hat{\sigma}, 0.2), 3.0)$ "
          r"with $\sigma_{\text{target}} = 15\%$. Bold indicates best per row.}")
    print(r"\label{tab:backtest-sharpe}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "r" * len(METHODS) + "}")
    print(r"\toprule")
    print("Dataset & " + " & ".join(latex_labels[m] for m in METHODS) + r" \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        ds_latex = ds.replace("_", r"\_")

        sharpes = {}
        for m in METHODS:
            s_data = get_strat(results[ds], m)
            if s_data:
                sharpes[m] = s_data["sharpe"]
        best_sharpe = max(sharpes.values()) if sharpes else None

        cells = []
        for m in METHODS:
            if m in sharpes:
                val = sharpes[m]
                if best_sharpe is not None and abs(val - best_sharpe) < 1e-8:
                    cells.append(r"\textbf{" + f"{val:.3f}" + "}")
                else:
                    cells.append(f"{val:.3f}")
            else:
                cells.append("--")

        print(f"{ds_latex} & " + " & ".join(cells) + r" \\")

    # Summary rows
    print(r"\midrule")

    # Average
    avg_cells = []
    avg_sharpes = {}
    for m in METHODS:
        if sharpe_by_method[m]:
            avg = np.mean(sharpe_by_method[m])
            avg_sharpes[m] = avg
            avg_cells.append(f"{avg:.3f}")
        else:
            avg_cells.append("--")
    if avg_sharpes:
        best_avg = max(avg_sharpes.values())
        avg_cells_final = []
        for m, c in zip(METHODS, avg_cells):
            if m in avg_sharpes and abs(avg_sharpes[m] - best_avg) < 1e-8:
                avg_cells_final.append(r"\textbf{" + c + "}")
            else:
                avg_cells_final.append(c)
        avg_cells = avg_cells_final
    print(r"\textit{Average} & " + " & ".join(avg_cells) + r" \\")

    # Best count
    best_s_cells = []
    best_s_max = max(best_counts.values()) if best_counts else 0
    for m in METHODS:
        c = f"{best_counts[m]}/{n_ds}"
        if best_counts[m] == best_s_max:
            best_s_cells.append(r"\textbf{" + c + "}")
        else:
            best_s_cells.append(c)
    print(r"\textit{Best} & " + " & ".join(best_s_cells) + r" \\")

    # TSB vs Linear wins
    tsb_vs_lin_cells = []
    for m in METHODS:
        if m == "Linear+TSB":
            tsb_vs_lin_cells.append(r"\textbf{" + f"{tsb_vs_linear_wins}/{n_compared}" + "}")
        elif m == "Linear":
            losses = n_compared - tsb_vs_linear_wins
            tsb_vs_lin_cells.append(f"{losses}/{n_compared}")
        else:
            tsb_vs_lin_cells.append("--")
    print(r"\textit{TSB vs Linear} & " + " & ".join(tsb_vs_lin_cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
               "EUR_USD", "GBP_USD", "USD_JPY"]
STOCK_DATASETS = ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]

METHOD_COLORS = {
    "Historical Vol": "#9E9E9E",
    "XGBoost": "#FF7043",
    "LightGBM": "#66BB6A",
    "Linear": "#42A5F5",
    "Linear+TSB": "#AB47BC",
}


def get_sharpe(results, ds, method):
    """Extract Sharpe from results dict (handles flat format)."""
    r = results.get(ds, {}).get(method)
    if r is None:
        return None
    return r["sharpe"]


def plot_tsb_vs_linear(results):
    """Paired comparison: TSB vs Linear Sharpe with difference bars."""
    datasets = [ds for ds in ALL_DATASETS if ds in results]

    diffs = []
    labels = []
    colors = []
    for ds in datasets:
        s_tsb = get_sharpe(results, ds, "Linear+TSB")
        s_lin = get_sharpe(results, ds, "Linear")
        if s_tsb is not None and s_lin is not None:
            diffs.append(s_tsb - s_lin)
            labels.append(ds)
            colors.append('#43A047' if s_tsb > s_lin else '#E53935')

    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(labels))
    bars = ax.barh(y, diffs, color=colors, edgecolor='white', linewidth=0.5, height=0.7)

    # Add value labels
    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        w = bar.get_width()
        ax.text(w + 0.002 * (1 if w >= 0 else -1), i,
                f'{diff:+.3f}', ha='left' if w >= 0 else 'right',
                va='center', fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='#424242', linewidth=1)

    # FX / Stocks separator
    fx_labels = [l for l in labels if l in FX_DATASETS]
    if 0 < len(fx_labels) < len(labels):
        ax.axhline(y=len(fx_labels) - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.5)

    wins = sum(1 for d in diffs if d > 0)
    total = len(diffs)

    ax.set_xlabel('Sharpe Difference (TSB - Linear)', fontsize=11)
    ax.set_title(f'TSB vs Linear: Paired Sharpe Comparison ({wins}/{total} wins)',
                 fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'tsb_vs_linear.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: tsb_vs_linear.png")


def plot_cumulative_return_diff(daily_data):
    """Bar chart: TSB vs Linear cumulative return advantage per dataset."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [ds for ds in ALL_DATASETS
                if ds in daily_data and "Linear+TSB" in daily_data[ds] and "Linear" in daily_data[ds]]

    diffs = []
    for ds in datasets:
        tsb_cum = np.cumprod(1 + np.array(daily_data[ds]["Linear+TSB"]))[-1]
        lin_cum = np.cumprod(1 + np.array(daily_data[ds]["Linear"]))[-1]
        diffs.append((tsb_cum / lin_cum - 1) * 100)

    # Separate FX and Stocks with a visual gap
    fx_ds = [ds for ds in datasets if ds in FX_DATASETS]
    stock_ds = [ds for ds in datasets if ds in STOCK_DATASETS]
    ordered = fx_ds + stock_ds
    ordered_diffs = []
    for ds in ordered:
        idx = datasets.index(ds)
        ordered_diffs.append(diffs[idx])

    colors = ['#43A047' if d >= 0 else '#E53935' for d in ordered_diffs]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(ordered))
    bars = ax.bar(x, ordered_diffs, color=colors, edgecolor='white', width=0.7)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, ordered_diffs)):
        y_pos = val + (0.8 if val >= 0 else -0.8)
        va = 'bottom' if val >= 0 else 'top'
        ax.text(i, y_pos, f'{val:+.1f}%', ha='center', va=va,
                fontsize=9, fontweight='bold')

    # Category separator
    sep = len(fx_ds) - 0.5
    ax.axvline(x=sep, color='#424242', linewidth=1, linestyle='--', alpha=0.5)
    ax.text(sep / 2, ax.get_ylim()[1] * 0.92, 'FX', ha='center',
            fontsize=12, fontweight='bold', color='#616161')
    ax.text((sep + len(ordered)) / 2, ax.get_ylim()[1] * 0.92, 'Stocks', ha='center',
            fontsize=12, fontweight='bold', color='#616161')

    ax.axhline(y=0, color='#424242', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Cumulative Return Advantage (%)', fontsize=12)
    ax.set_title(f'Linear+TSB vs Linear — Cumulative Return Difference '
                 f'({sum(1 for d in ordered_diffs if d > 0)}/{len(ordered_diffs)} wins)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tsb_vs_linear_cumreturn.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: tsb_vs_linear_cumreturn.png")


def generate_figures(results, daily_data=None):
    """Generate all backtest figures."""
    print("\nGenerating figures...")
    plot_tsb_vs_linear(results)
    if daily_data:
        plot_cumulative_return_diff(daily_data)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest: Inverse-Volatility Position Sizing")
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    args = parser.parse_args()

    daily_data = None
    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        if DAILY_RETURNS_PATH.exists():
            with open(DAILY_RETURNS_PATH) as f:
                daily_data = json.load(f)
    else:
        results, daily_data = run_all_experiments()

    analyze(results)
    generate_figures(results, daily_data)
