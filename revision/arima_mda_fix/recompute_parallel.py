"""Parallel version: one process per dataset, BLAS threads pinned to 1."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import json
import time
import warnings
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).parent
OUT = HERE / "arima_mda_recompute.json"
STORED = Path(
    "/Users/twcch/My Drive/Dev/Research/time-series-research/TS-Booster/"
    "experiments/comprehensive_results/benchmark/benchmark_baselines_results.json"
)

DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]
VOL_WINDOW = 7
HORIZON = 3


def run_one(ds):
    import numpy as np
    import pandas as pd
    from pmdarima import auto_arima

    t0 = time.time()
    df = pd.read_csv(HERE / "dataset" / f"{ds}.csv")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return"] = df["log_return"].fillna(0.0)
    df["Volatility"] = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    vol = df["Volatility"].values
    n = len(df)
    split_2 = int(n * 0.80)
    train_val_vol = vol[:split_2]
    test_vol = vol[split_2:]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = auto_arima(
            train_val_vol, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore",
            max_order=10, max_p=5, max_q=5,
        )
        n_windows = len(test_vol) - HORIZON + 1
        preds = np.zeros((n_windows, HORIZON))
        for i in range(n_windows):
            preds[i] = model.predict(n_periods=HORIZON)
            model.update(test_vol[i : i + 1])

    targets = np.array([test_vol[i : i + HORIZON] for i in range(n_windows)])
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae = float(np.mean(np.abs(preds - targets)))
    lo_buggy = test_vol[:n_windows]
    lo_fixed = vol[split_2 - 1 : split_2 - 1 + n_windows]

    def mda(lo):
        return float(np.mean(np.sign(preds - lo[:, None]) == np.sign(targets - lo[:, None])))

    return ds, {
        "order": str(model.order), "rmse": rmse, "mae": mae,
        "mda_buggy": mda(lo_buggy), "mda_fixed": mda(lo_fixed),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    import numpy as np
    stored = json.load(open(STORED))
    results = {}
    if OUT.exists():
        results = {k: v for k, v in json.load(open(OUT)).items() if not k.startswith("_")}
    todo = [ds for ds in DATASETS if ds not in results]
    print(f"cached: {sorted(results)}; todo: {todo}", flush=True)

    with Pool(5) as pool:
        for ds, r in pool.imap_unordered(run_one, todo):
            s = stored[ds]["ARIMA"]
            s_rmse = s["rmse"][0] if isinstance(s["rmse"], list) else s["rmse"]
            s_mda = s["mda"][0] if isinstance(s["mda"], list) else s["mda"]
            r["stored_rmse"] = s_rmse
            r["stored_mda"] = s_mda
            r["rmse_match"] = bool(abs(r["rmse"] - s_rmse) < 5e-5)
            r["mda_match"] = bool(abs(r["mda_buggy"] - s_mda) < 5e-5)
            results[ds] = r
            print(f"{ds}: order={r['order']} rmse={r['rmse']:.4f} "
                  f"(stored {s_rmse:.4f}, match={r['rmse_match']}) "
                  f"mda_buggy={r['mda_buggy']:.4f} (stored {s_mda:.4f}, match={r['mda_match']}) "
                  f"mda_fixed={r['mda_fixed']:.4f} [{r['seconds']}s]", flush=True)
            json.dump(results, open(OUT, "w"), indent=2)

    fixed_avg = float(np.mean([results[d]["mda_fixed"] for d in DATASETS]))
    buggy_avg = float(np.mean([results[d]["mda_buggy"] for d in DATASETS]))
    all_match = all(results[d]["rmse_match"] and results[d]["mda_match"] for d in DATASETS)
    print(f"\nAvg MDA buggy={buggy_avg:.4f} fixed={fixed_avg:.4f} all_match={all_match}", flush=True)
    json.dump({**results, "_summary": {"avg_mda_buggy": buggy_avg,
                                        "avg_mda_fixed": fixed_avg,
                                        "all_match": all_match}},
              open(OUT, "w"), indent=2)
