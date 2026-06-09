"""
Phase 2: Volatility-Specific Baselines

Implements and evaluates four classical volatility forecasting baselines:
  1. EWMA       (λ=0.94, RiskMetrics)
  2. HAR-RV     (Corsi, 2009) — direct multi-step OLS
  3. GARCH(1,1)
  4. GJR-GARCH  (asymmetric GARCH)

All baselines are fit on the training set only (no look-ahead).
Metrics (RMSE/MAE/MDA) are computed in raw (unscaled) volatility space,
matching the evaluation protocol in benchmark_baselines.py.

Usage:
  uv run python revision/phase2_vol_baselines/vol_baselines.py
"""
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent.parent))

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
EWMA_LAMBDA = 0.94


# ── Data loading ────────────────────────────────────────────────────────────

def load_dataset(ds: str) -> dict | None:
    path = Path(f"dataset/{ds}.csv")
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    df["Volatility"] = df["log_return"].rolling(VOL_WINDOW).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    split_1 = int(n * 0.70)
    split_2 = int(n * 0.80)

    vol = df["Volatility"].values
    ret = df["log_return"].values

    # Build test targets and last_obs — same alignment as benchmark_baselines.py
    # test sample i: predict vol[split_2+i : split_2+i+HORIZON]
    n_test = n - split_2 - HORIZON + 1
    y_test    = np.stack([vol[split_2 + i : split_2 + i + HORIZON] for i in range(n_test)])
    last_obs  = np.array([vol[split_2 + i - 1] for i in range(n_test)])  # vol at t (last known)

    return {
        "vol": vol, "ret": ret,
        "split_1": split_1, "split_2": split_2,
        "y_test": y_test, "last_obs": last_obs,
        "n_test": n_test,
    }


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray, last_obs: np.ndarray) -> dict:
    """RMSE / MAE / MDA in raw vol space."""
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae  = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))
    return {"rmse": rmse, "mae": mae, "mda": mda}


# ── EWMA (λ=0.94) ────────────────────────────────────────────────────────────

def run_ewma(data: dict) -> dict:
    """
    RiskMetrics EWMA on log returns.
    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t
    h-step forecast = flat (random walk in variance), convert to annualized vol.
    """
    ret = data["ret"]
    split_2 = data["split_2"]
    n_test  = data["n_test"]
    lam = EWMA_LAMBDA

    # Initialise on first return, then run through entire series
    var = ret[0] ** 2
    ewma_vars = np.zeros(len(ret))
    ewma_vars[0] = var
    for t in range(1, len(ret)):
        var = lam * var + (1 - lam) * ret[t] ** 2
        ewma_vars[t] = var

    # For test sample i: last known state is at index split_2 + i - 1
    preds = np.zeros((n_test, HORIZON))
    for i in range(n_test):
        t_idx = split_2 + i - 1
        sigma_ann = np.sqrt(ewma_vars[t_idx] * 252) * 100
        preds[i] = sigma_ann  # flat forecast for all h

    return compute_metrics(preds, data["y_test"], data["last_obs"])


# ── HAR-RV (Corsi, 2009) ─────────────────────────────────────────────────────

def run_har_rv(data: dict) -> dict:
    """
    HAR model on rolling volatility series.
    Features: σ_d (daily), σ_w (5-day avg), σ_m (22-day avg).
    Direct multi-step: fit separate OLS for each horizon h=1,2,3.
    Fit on train, predict on test.
    """
    vol = data["vol"]
    split_1 = data["split_1"]
    split_2 = data["split_2"]
    n_test  = data["n_test"]

    def build_features(vol_arr, start, end):
        X, y_h = {h: [] for h in range(1, HORIZON + 1)}, {h: [] for h in range(1, HORIZON + 1)}
        for t in range(22, end - start):           # need 22 days of history
            abs_t = start + t
            if abs_t + HORIZON > len(vol_arr):
                break
            sigma_d = vol_arr[abs_t]
            sigma_w = np.mean(vol_arr[abs_t - 4 : abs_t + 1])   # 5-day incl. today
            sigma_m = np.mean(vol_arr[abs_t - 21 : abs_t + 1])  # 22-day incl. today
            feat = [1.0, sigma_d, sigma_w, sigma_m]
            for h in range(1, HORIZON + 1):
                X[h].append(feat)
                y_h[h].append(vol_arr[abs_t + h])
        return {h: np.array(X[h]) for h in range(1, HORIZON + 1)}, \
               {h: np.array(y_h[h]) for h in range(1, HORIZON + 1)}

    X_tr, y_tr = build_features(vol, 0, split_1)

    # Fit one OLS per horizon
    models = {}
    for h in range(1, HORIZON + 1):
        reg = LinearRegression(fit_intercept=False)  # intercept in features
        reg.fit(X_tr[h], y_tr[h])
        models[h] = reg

    # Predict test — use split_2+i-1 as last known (matches DL model's X_test[:,-1,:])
    preds = np.zeros((n_test, HORIZON))
    for i in range(n_test):
        t_idx = split_2 + i - 1
        if t_idx < 22:
            continue
        sigma_d = vol[t_idx]
        sigma_w = np.mean(vol[t_idx - 4 : t_idx + 1])
        sigma_m = np.mean(vol[t_idx - 21 : t_idx + 1])
        feat = np.array([[1.0, sigma_d, sigma_w, sigma_m]])
        for h in range(1, HORIZON + 1):
            preds[i, h - 1] = float(models[h].predict(feat)[0])

    return compute_metrics(preds, data["y_test"], data["last_obs"])


# ── GARCH(1,1) ───────────────────────────────────────────────────────────────

def run_garch(data: dict, p: int = 1, q: int = 1, o: int = 0, label: str = "GARCH") -> dict:
    """
    Fit GARCH(p,q) [or GJR-GARCH with o>0] on training log returns.
    For each test sample, compute h-step ahead conditional variance forecast,
    convert to annualized vol: sqrt(σ²_{t+h|t} * 252) * 100.

    Uses arch's analytic h-step forecast (no simulation).
    """
    from arch import arch_model

    ret = data["ret"]
    split_1 = data["split_1"]
    split_2 = data["split_2"]
    n_test  = data["n_test"]

    train_ret = ret[:split_1] * 100  # arch expects returns in % scale

    am = arch_model(train_ret, vol="Garch", p=p, o=o, q=q, dist="normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    omega = res.params["omega"]
    alpha = res.params.get("alpha[1]", 0.0)
    gamma = res.params.get("gamma[1]", 0.0)  # GJR asymmetry term
    beta  = res.params.get("beta[1]", 0.0)

    # Compute conditional variance for the entire series using fitted params
    # We need to track conditional variance through val and test periods
    full_ret = ret * 100
    n_total  = len(full_ret)

    cond_var = np.zeros(n_total)
    cond_var[0] = res.conditional_volatility[0] ** 2

    # Use fitted initial variance; replay recursion from train_end onward
    init_cond_var = res.conditional_volatility ** 2  # shape (split_1,)

    # Seed with last train value
    var_t = float(init_cond_var[-1])
    eps_t = float(full_ret[split_1 - 1])

    for t in range(split_1, n_total):
        indicator = 1.0 if eps_t < 0 else 0.0
        var_t = omega + (alpha + gamma * indicator) * eps_t ** 2 + beta * var_t
        var_t = max(var_t, 1e-8)
        cond_var[t] = var_t
        eps_t = full_ret[t]

    # h-step analytic forecast from state at time t:
    # σ²_{t+h} = ω*(1 + Σ_{k=1}^{h-1} (α+β)^k) + (α+β)^h * σ²_t
    ab = alpha + 0.5 * gamma + beta  # persistence (GJR uses 0.5γ unconditionally)
    long_run_var = omega / max(1 - ab, 1e-8)

    preds = np.zeros((n_test, HORIZON))
    for i in range(n_test):
        t_idx = split_2 + i - 1
        var_now = cond_var[t_idx]
        for h in range(1, HORIZON + 1):
            # Analytic h-step forecast
            var_h = long_run_var + (ab ** h) * (var_now - long_run_var)
            var_h = max(var_h, 1e-8)
            # Convert from % returns scale back to decimal, then annualize
            preds[i, h - 1] = np.sqrt(var_h / 10000 * 252) * 100  # /10000 because ret was *100

    return compute_metrics(preds, data["y_test"], data["last_obs"])


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    dl_results = json.loads(DL_RESULTS_PATH.read_text())

    METHODS = ["ewma", "har_rv", "garch", "gjr_garch", "persistence", "linear", "tsb"]
    all_results = {}

    print("\n" + "=" * 110)
    print("PHASE 2 — Volatility Baselines vs Linear vs TSB  (test set, raw-scale RMSE)")
    print("=" * 110)
    header = (f"{'Dataset':<14} {'EWMA':>7} {'HAR-RV':>7} {'GARCH':>7} {'GJR':>7} "
              f"{'Persist':>7} {'Linear':>7} {'TSB':>7}  {'Best':>10}")
    print(header)
    print("-" * 110)

    for ds in ALL_DATASETS:
        data = load_dataset(ds)
        if data is None:
            continue

        # Statistical baselines
        ewma_m   = run_ewma(data)
        har_m    = run_har_rv(data)
        garch_m  = run_garch(data, p=1, q=1, o=0, label="GARCH")
        gjr_m    = run_garch(data, p=1, q=1, o=1, label="GJR-GARCH")

        # DL baselines (5-seed mean)
        if ds not in dl_results:
            continue
        per_rmse = float(np.mean([ewma_m["rmse"]]))  # placeholder, replaced below

        # Load persistence from phase1 if available
        p1_path = Path("revision/phase1_persistence/persistence_results.json")
        per_rmse = None
        if p1_path.exists():
            p1 = json.loads(p1_path.read_text())
            for row in p1:
                if row["dataset"] == ds:
                    per_rmse = row["per_rmse"]
                    break

        lin_rmse = float(np.mean(dl_results[ds]["vanilla"]["rmse"]))
        tsb_rmse = float(np.mean(dl_results[ds]["booster"]["rmse"]))

        scores = {
            "EWMA": ewma_m["rmse"], "HAR-RV": har_m["rmse"],
            "GARCH": garch_m["rmse"], "GJR": gjr_m["rmse"],
            "Persist": per_rmse, "Linear": lin_rmse, "TSB": tsb_rmse,
        }
        best = min(scores, key=scores.get)

        all_results[ds] = {
            "ewma":     ewma_m,
            "har_rv":   har_m,
            "garch":    garch_m,
            "gjr_garch": gjr_m,
            "persistence_rmse": per_rmse,
            "linear_rmse": lin_rmse,
            "tsb_rmse":  tsb_rmse,
        }

        per_str = f"{per_rmse:7.4f}" if per_rmse else "   N/A "
        print(f"{ds:<14} {ewma_m['rmse']:7.4f} {har_m['rmse']:7.4f} "
              f"{garch_m['rmse']:7.4f} {gjr_m['rmse']:7.4f} "
              f"{per_str} {lin_rmse:7.4f} {tsb_rmse:7.4f}  {best:>10}")

    print("-" * 110)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n── Win counts vs TSB (RMSE, lower=better) ──────────────────────")
    methods = ["ewma", "har_rv", "garch", "gjr_garch"]
    labels  = ["EWMA", "HAR-RV", "GARCH(1,1)", "GJR-GARCH"]
    tsb_wins = {m: 0 for m in methods}
    tsb_avg_imp = {m: [] for m in methods}

    for ds, res in all_results.items():
        tsb_r = res["tsb_rmse"]
        for m in methods:
            base_r = res[m]["rmse"]
            if tsb_r < base_r:
                tsb_wins[m] += 1
            imp = (base_r - tsb_r) / base_r * 100
            tsb_avg_imp[m].append(imp)

    n = len(all_results)
    for m, label in zip(methods, labels):
        avg = np.mean(tsb_avg_imp[m])
        print(f"  TSB vs {label:<12}: {tsb_wins[m]:2d}/{n}  avg improvement {avg:+.2f}%")

    # ── Category breakdown for HAR-RV (most important baseline) ──────────
    print("\n── HAR-RV vs TSB by category ────────────────────────────────────")
    for cat, members in CATEGORIES.items():
        rows = [(ds, all_results[ds]) for ds in members if ds in all_results]
        if not rows:
            continue
        wins = sum(1 for _, r in rows if r["tsb_rmse"] < r["har_rv"]["rmse"])
        avg  = np.mean([(r["har_rv"]["rmse"] - r["tsb_rmse"]) / r["har_rv"]["rmse"] * 100
                        for _, r in rows])
        print(f"  {cat:<20}: TSB wins {wins}/{len(rows)}, avg {avg:+.2f}%")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = Path("revision/phase2_vol_baselines/vol_baselines_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
