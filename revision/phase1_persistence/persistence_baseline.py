"""
Phase 1: Persistence Baseline Test

Question: Can vanilla Linear beat the naive carry-forward baseline?
If not, the "15% improvement over Linear" claim is built on a weak foundation.

Persistence prediction: σ_hat_{t+k} = σ_t for k=1,2,3
(carry the last known volatility forward for all 3 horizon steps)

Usage:
  uv run python revision/persistence_baseline.py
"""
import sys
import json
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
    "TWD FX":           ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD"],
    "International FX": ["EUR_USD", "GBP_USD", "USD_JPY"],
    "Taiwan Stocks":    ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"],
}

DL_RESULTS_PATH = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")

LOOKBACK   = 30
HORIZON    = 3
VOL_WINDOW = 7


def extract_test_data(ds: str) -> dict | None:
    """Reproduce the exact train/val/test split from data_loader.py."""
    path = Path(f"dataset/{ds}.csv")
    if not path.exists():
        print(f"  [WARN] {ds}.csv not found")
        return None

    df = pd.read_csv(path)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    df["Volatility"] = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    features = df[["Volatility", "log_return"]].values
    n = len(features)
    split_1 = int(n * 0.70)
    split_2 = int(n * 0.80)

    scaler = StandardScaler()
    scaler.fit(features[:split_1])
    fs = scaler.transform(features)

    test_feat = fs[split_2 - LOOKBACK:]

    X_test, y_test = [], []
    for i in range(len(test_feat) - LOOKBACK - HORIZON + 1):
        X_test.append(test_feat[i : i + LOOKBACK])
        y_test.append(test_feat[i + LOOKBACK : i + LOOKBACK + HORIZON, 0])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    last_obs = X_test[:, -1, 0]  # last known σ in scaled space, same as benchmark_baselines.py

    return {"X_test": X_test, "y_test": y_test, "last_obs": last_obs, "scaler": scaler}


def compute_metrics(preds_scaled, targets_scaled, scaler, last_obs_scaled):
    """Inverse-transform then compute RMSE / MAE / MDA — identical to benchmark_baselines.py."""
    scale, mean = scaler.scale_[0], scaler.mean_[0]
    p = preds_scaled   * scale + mean   # [N, H]
    t = targets_scaled * scale + mean
    l = last_obs_scaled * scale + mean  # [N]

    rmse = float(np.sqrt(mean_squared_error(t.flatten(), p.flatten())))
    mae  = float(np.mean(np.abs(t.flatten() - p.flatten())))
    mda  = float(np.mean(np.sign(p - l[:, None]) == np.sign(t - l[:, None])))
    return rmse, mae, mda


def persistence_predict(X_test):
    """Carry forward the last volatility in the lookback window."""
    last = X_test[:, -1, 0]                        # [N]  last σ (scaled)
    return np.tile(last[:, None], (1, HORIZON))     # [N, H]


def main():
    dl_results = json.loads(DL_RESULTS_PATH.read_text())

    rows = []
    for ds in ALL_DATASETS:
        data = extract_test_data(ds)
        if data is None:
            continue

        # ── Persistence ──────────────────────────────────────────────
        p_preds = persistence_predict(data["X_test"])
        p_rmse, p_mae, p_mda = compute_metrics(
            p_preds, data["y_test"], data["scaler"], data["last_obs"]
        )

        # ── Linear vanilla (mean of 5 seeds) ─────────────────────────
        if ds not in dl_results:
            print(f"  [WARN] {ds} not in DL results")
            continue
        v = dl_results[ds]["vanilla"]
        lin_rmse = float(np.mean(v["rmse"]))
        lin_mae  = float(np.mean(v["mae"]))
        lin_mda  = float(np.mean(v["mda"]))

        # ── TSB (mean of 5 seeds) ─────────────────────────────────────
        b = dl_results[ds]["booster"]
        tsb_rmse = float(np.mean(b["rmse"]))
        tsb_mae  = float(np.mean(b["mae"]))
        tsb_mda  = float(np.mean(b["mda"]))

        rows.append({
            "dataset":   ds,
            "per_rmse":  p_rmse,  "per_mae": p_mae,  "per_mda": p_mda,
            "lin_rmse":  lin_rmse, "lin_mae": lin_mae, "lin_mda": lin_mda,
            "tsb_rmse":  tsb_rmse, "tsb_mae": tsb_mae, "tsb_mda": tsb_mda,
        })

    # ── Print table ────────────────────────────────────────────────────
    header = (
        f"{'Dataset':<14} "
        f"{'Per RMSE':>9} {'Lin RMSE':>9} {'TSB RMSE':>9}  "
        f"{'Lin>Per?':>8}  "
        f"{'Per MAE':>8} {'Lin MAE':>8}  "
        f"{'Per MDA':>7} {'Lin MDA':>7} {'TSB MDA':>7}"
    )
    print("\n" + "=" * 100)
    print("PERSISTENCE vs LINEAR vs TSB  (test set, raw-scale metrics)")
    print("=" * 100)
    print(header)
    print("-" * 100)

    lin_beats_per = 0
    tsb_beats_per = 0
    rmse_gaps = []

    for r in rows:
        lin_wins = r["lin_rmse"] < r["per_rmse"]
        flag = "✓" if lin_wins else "✗ FAIL"
        if lin_wins:
            lin_beats_per += 1
        if r["tsb_rmse"] < r["per_rmse"]:
            tsb_beats_per += 1
        gap = (r["per_rmse"] - r["lin_rmse"]) / r["per_rmse"] * 100
        rmse_gaps.append(gap)

        print(
            f"{r['dataset']:<14} "
            f"{r['per_rmse']:>9.4f} {r['lin_rmse']:>9.4f} {r['tsb_rmse']:>9.4f}  "
            f"{flag:>8}  "
            f"{r['per_mae']:>8.4f} {r['lin_mae']:>8.4f}  "
            f"{r['per_mda']:>7.4f} {r['lin_mda']:>7.4f} {r['tsb_mda']:>7.4f}"
        )

    print("-" * 100)
    print(f"\nLinear beats Persistence (RMSE): {lin_beats_per}/{len(rows)}")
    print(f"TSB    beats Persistence (RMSE): {tsb_beats_per}/{len(rows)}")
    avg_gap = np.mean(rmse_gaps)
    print(f"Avg RMSE gap (Persistence→Linear): {avg_gap:+.2f}%  (positive = Linear better)")

    # ── Category breakdown ─────────────────────────────────────────────
    print("\n── Category Breakdown ──────────────────────────────────────────")
    ds_map = {r["dataset"]: r for r in rows}
    for cat, members in CATEGORIES.items():
        cat_rows = [ds_map[d] for d in members if d in ds_map]
        if not cat_rows:
            continue
        wins = sum(1 for r in cat_rows if r["lin_rmse"] < r["per_rmse"])
        avg  = np.mean([(r["per_rmse"] - r["lin_rmse"]) / r["per_rmse"] * 100 for r in cat_rows])
        print(f"  {cat:<20}: Linear beats Persistence {wins}/{len(cat_rows)}, avg gap {avg:+.2f}%")

    # ── Save results ───────────────────────────────────────────────────
    out_path = Path("revision/results/persistence_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nResults saved to {out_path}")

    # ── Verdict ────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    if lin_beats_per == len(rows):
        print("VERDICT: Linear beats Persistence on ALL datasets → baseline claim is solid.")
    elif lin_beats_per >= len(rows) * 0.75:
        print(f"VERDICT: Linear beats Persistence on {lin_beats_per}/{len(rows)} datasets → mostly solid, discuss exceptions.")
    else:
        print(f"VERDICT: Linear beats Persistence on only {lin_beats_per}/{len(rows)} datasets → SERIOUS ISSUE.")
        print("         The '15% improvement over Linear' is relative to a baseline weaker than Persistence.")
        print("         Consider reframing results relative to Persistence.")
    print("=" * 100)


if __name__ == "__main__":
    main()
