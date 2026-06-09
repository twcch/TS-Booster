"""
Phase 5b: Diebold-Mariano Test

Why DM test?
  - 5-seed paired t-test tests seed-level stability (initialization robustness)
  - DM test tests whether two methods produce significantly different forecast accuracy
    on the SAME test samples (proper evaluation of generalization)
  - Gold standard in financial forecasting comparison (Diebold & Mariano, 1995)

Method:
  - Re-run with seed=42: TSB + Vanilla Linear + DL baselines + HAR-RV
  - Save per-sample squared errors for each method
  - For each dataset, compute DM statistic:
      d_t = L(e1_t) - L(e2_t),  where L = mean squared error over h={1,2,3}
      DM  = d_bar / sqrt(LRV(d) / T)
  - HLN (Harvey-Leybourne-Newbold, 1997) small-sample correction
  - H0: E[d_t] = 0 (equal predictive accuracy), H1: E[d_t] > 0 (TSB is better)

Usage:
  uv run python revision/phase5_stat_upgrade/dm_test_runner.py
  uv run python revision/phase5_stat_upgrade/dm_test_runner.py --analyze  # skip re-run
"""
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats as scipy_stats

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.models.sota_baselines import DLinear, PatchTST, iTransformer
from src.engine.trainer import train_v11, train_baseline
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEED     = 42
SEQ_LEN  = 30
PRED_LEN = 3
OUT_DIR  = Path("revision/phase5_stat_upgrade")
ERRORS_PATH = OUT_DIR / "dm_per_sample_errors.json"


# ── Seed / Evaluate ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate_with_errors(model, test_loader, scaler, device):
    """Return per-sample MSE (averaged over h=1,2,3) and scalar metrics."""
    model.eval()
    preds_list, targets_list, last_obs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            out = model(x)
            pred = out[0] if isinstance(out, tuple) else out
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())

    preds   = np.concatenate(preds_list)      # [T, 3]
    targets = np.concatenate(targets_list)    # [T, 3]
    last_obs = np.concatenate(last_obs_list)  # [T]

    preds    = preds    * scaler.scale_[0] + scaler.mean_[0]
    targets  = targets  * scaler.scale_[0] + scaler.mean_[0]
    last_obs = last_obs * scaler.scale_[0] + scaler.mean_[0]

    sq_errors = (preds - targets) ** 2              # [T, 3]
    mse_per_t = sq_errors.mean(axis=1)              # [T] — avg over h

    rmse = float(np.sqrt(sq_errors.mean()))
    mae  = float(np.abs(preds - targets).mean())
    mda  = float(np.mean(np.sign(preds - last_obs[:, None]) == np.sign(targets - last_obs[:, None])))

    return {"rmse": rmse, "mae": mae, "mda": mda, "mse_per_t": mse_per_t.tolist()}


# ── HAR-RV re-run (deterministic) ────────────────────────────────────────────

def run_har_rv(df: pd.DataFrame, seq_len: int, pred_len: int) -> list[float]:
    """Re-run HAR-RV, return per-sample MSE (mean over h=1..pred_len)."""
    from sklearn.linear_model import LinearRegression

    # Compute volatility from Close (same as data_loader)
    close = df["Close"].values.astype(float)
    log_ret = np.diff(np.log(close), prepend=np.nan)
    log_ret[0] = 0.0
    vol_series = pd.Series(log_ret).rolling(7).std().values * np.sqrt(252) * 100
    vols = vol_series
    n    = len(vols)
    n1   = int(n * 0.7)
    n2   = int(n * 0.8)

    def make_features(idx):
        d  = vols[idx]
        w  = vols[max(0, idx - 4): idx + 1].mean()
        m  = vols[max(0, idx - 21): idx + 1].mean()
        return [d, w, m]

    X_train = np.array([make_features(i) for i in range(seq_len - 1, n1)])
    har_models = {}
    for h in range(1, pred_len + 1):
        y_train = np.array([vols[i + h] if i + h < n else vols[-1]
                            for i in range(seq_len - 1, n1)])
        m = LinearRegression().fit(X_train, y_train)
        har_models[h] = m

    mse_per_t = []
    for i in range(n2 - seq_len + 1, n - pred_len):
        t_idx = seq_len - 1 + (i - (n2 - seq_len + 1)) + n2 - seq_len
        # Use last known index = split2 + sample - 1
        t_idx = n2 + (i - (n2 - seq_len + 1)) - 1
        if t_idx < 0 or t_idx + pred_len >= n:
            continue
        feats = make_features(t_idx)
        sq_errs = []
        for h in range(1, pred_len + 1):
            pred_v  = har_models[h].predict([feats])[0]
            true_v  = vols[t_idx + h]
            sq_errs.append((pred_v - true_v) ** 2)
        mse_per_t.append(np.mean(sq_errs))

    return mse_per_t


# ── DM Test ───────────────────────────────────────────────────────────────────

def dm_test(e1: list[float], e2: list[float], h: int = 1) -> tuple[float, float]:
    """
    DM test with HLN (1997) small-sample correction.
    H0: E[d_t] = 0  (equal accuracy)
    H1: E[d_t] > 0  (e1 < e2, i.e., method1 is better)

    Returns (dm_stat, p_value_onesided)
    """
    n = min(len(e1), len(e2))
    d = np.array(e1[:n]) - np.array(e2[:n])  # positive = method1 worse

    d_bar = d.mean()
    # Newey-West HAC variance (lags = h-1 for h-step forecast)
    T     = len(d)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_l = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        gamma_sum += (1 - lag / h) * gamma_l
    lrv = (gamma0 + 2 * gamma_sum) / T

    dm_stat = d_bar / np.sqrt(max(lrv, 1e-12))

    # HLN correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm_stat * correction

    # Two-sided t distribution, df = T-1; one-sided: H1: method2 better than method1
    # (we want p(TSB < baseline) so we want d_bar < 0, i.e. dm_stat < 0)
    p_two = float(2 * scipy_stats.t.cdf(-abs(dm_hln), df=T - 1))
    # One-sided p-value that e1 < e2 (TSB better than baseline)
    p_one = float(scipy_stats.t.cdf(-dm_hln, df=T - 1))

    return float(dm_hln), p_one


# ── Run all models ─────────────────────────────────────────────────────────────

def run_all(device):
    all_errors = {}

    for ds in ALL_DATASETS:
        print(f"\n── {ds} ──────────────────────────────────────────────")
        set_seed(SEED)
        df = pd.read_csv(f"dataset/{ds}.csv")
        train_loader, val_loader, test_loader, scaler = prepare_data(
            df, lookback=SEQ_LEN, horizon=PRED_LEN, batch_size=32
        )

        ds_errors = {}

        config = BEST_STAGE3_CONFIG.copy()
        base_kwargs = dict(
            seq_len=SEQ_LEN, pred_len=PRED_LEN, input_channels=2,
            decomp_kernels=config.get("decomp_kernels", [7, 13]),
            expert_kernels=config.get("expert_kernels", [3, 5, 7]),
            expert_hidden_dim=config.get("expert_hidden_dim", 32),
            dropout_p=config.get("dropout", 0.3),
            n_boost_rounds=config.get("n_boost_rounds", 2),
            expert_type=config.get("expert_type", "cnn"),
        )

        # 1. Vanilla Linear
        vanilla = FlexibleTSBooster(
            base_model=LinearBackbone(SEQ_LEN, 2, PRED_LEN),
            use_decomp=False, use_experts=False, **base_kwargs
        ).to(device)
        vanilla = train_v11(
            train_loader, val_loader, test_loader, device=device,
            horizon=PRED_LEN, config=config, model=vanilla, verbose=False
        )
        r = evaluate_with_errors(vanilla, test_loader, scaler, device)
        ds_errors["vanilla"] = r["mse_per_t"]
        print(f"  Vanilla  RMSE={r['rmse']:.4f}")
        del vanilla
        torch.cuda.empty_cache()

        # 2. TSB
        set_seed(SEED)
        tsb = FlexibleTSBooster(
            base_model=LinearBackbone(SEQ_LEN, 2, PRED_LEN),
            use_decomp=True, use_experts=True,
            expert_gate="snr", enrich_mode="trend", **base_kwargs
        ).to(device)
        tsb = train_v11(
            train_loader, val_loader, test_loader, device=device,
            horizon=PRED_LEN, config=config, model=tsb, verbose=False
        )
        r = evaluate_with_errors(tsb, test_loader, scaler, device)
        ds_errors["tsb"] = r["mse_per_t"]
        print(f"  TSB      RMSE={r['rmse']:.4f}")
        del tsb
        torch.cuda.empty_cache()

        # 3. DLinear
        set_seed(SEED)
        dlinear = DLinear(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=2)
        dlinear = train_baseline(
            train_loader, val_loader, test_loader,
            device=device, model=dlinear, num_epochs=100, lr=1e-4, verbose=False
        )
        r = evaluate_with_errors(dlinear, test_loader, scaler, device)
        ds_errors["dlinear"] = r["mse_per_t"]
        print(f"  DLinear  RMSE={r['rmse']:.4f}")
        del dlinear
        torch.cuda.empty_cache()

        # 4. PatchTST (with RevIN, as in original)
        set_seed(SEED)
        ptst = PatchTST(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=2, use_revin=True)
        ptst = train_baseline(
            train_loader, val_loader, test_loader,
            device=device, model=ptst, num_epochs=100, lr=1e-4, verbose=False
        )
        r = evaluate_with_errors(ptst, test_loader, scaler, device)
        ds_errors["patchtst"] = r["mse_per_t"]
        print(f"  PatchTST RMSE={r['rmse']:.4f}")
        del ptst
        torch.cuda.empty_cache()

        # 5. iTransformer
        set_seed(SEED)
        itr = iTransformer(seq_len=SEQ_LEN, pred_len=PRED_LEN, n_features=2)
        itr = train_baseline(
            train_loader, val_loader, test_loader,
            device=device, model=itr, num_epochs=100, lr=1e-4, verbose=False
        )
        r = evaluate_with_errors(itr, test_loader, scaler, device)
        ds_errors["itransformer"] = r["mse_per_t"]
        print(f"  iTransf  RMSE={r['rmse']:.4f}")
        del itr
        torch.cuda.empty_cache()

        # 6. HAR-RV (deterministic)
        ds_errors["har_rv"] = run_har_rv(df, SEQ_LEN, PRED_LEN)
        print(f"  HAR-RV   T={len(ds_errors['har_rv'])} samples")

        all_errors[ds] = ds_errors

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_PATH.write_text(json.dumps(all_errors, indent=2))
    print(f"\nPer-sample errors saved → {ERRORS_PATH}")
    return all_errors


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze(all_errors: dict):
    BASELINES = ["vanilla", "dlinear", "patchtst", "itransformer", "har_rv"]
    LABELS    = ["Vanilla", "DLinear", "PatchTST", "iTransformer", "HAR-RV"]

    print("\n" + "=" * 90)
    print("Diebold-Mariano Test: TSB vs each baseline (H1: TSB has lower MSE)")
    print("HLN (1997) small-sample correction, one-sided p-value")
    print("=" * 90)

    dm_results = {}

    for bl, lb in zip(BASELINES, LABELS):
        print(f"\n── TSB vs {lb} ──────────────────────────────────────────")
        print(f"{'Dataset':<14} {'DM_stat':>10} {'p-value':>10} {'sig':>6} {'T':>6}")
        print("-" * 60)

        p_vals  = []
        dm_list = []

        for ds in ALL_DATASETS:
            if ds not in all_errors or bl not in all_errors[ds]:
                continue
            tsb_e  = all_errors[ds]["tsb"]
            base_e = all_errors[ds][bl]
            # DM test: H1 tsb < baseline → d_t = baseline_e - tsb_e > 0 means TSB better
            # We'll call dm_test(baseline, tsb): d = e_base - e_tsb, H1: d > 0
            dm_stat, p_one = dm_test(base_e, tsb_e, h=PRED_LEN)
            p_vals.append(p_one)
            dm_list.append(dm_stat)
            T = min(len(tsb_e), len(base_e))
            sig = "**" if p_one < 0.01 else ("*" if p_one < 0.05 else "n.s.")
            print(f"{ds:<14} {dm_stat:>10.3f} {p_one:>10.4f} {sig:>6} {T:>6}")

        print(f"\n  Sig (p<0.05): {sum(p < 0.05 for p in p_vals)}/{len(p_vals)}")
        print(f"  Sig (p<0.01): {sum(p < 0.01 for p in p_vals)}/{len(p_vals)}")

        dm_results[lb] = {"dm_stats": dm_list, "p_values": p_vals,
                          "datasets": [ds for ds in ALL_DATASETS if ds in all_errors and bl in all_errors[ds]]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dm_test_results.json").write_text(json.dumps(dm_results, indent=2))
    print(f"\nDM results saved → {OUT_DIR}/dm_test_results.json")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Skip re-run, load saved errors")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.analyze:
        if not ERRORS_PATH.exists():
            print(f"ERROR: {ERRORS_PATH} not found. Run without --analyze first.")
            return
        all_errors = json.loads(ERRORS_PATH.read_text())
        print(f"Loaded per-sample errors from {ERRORS_PATH}")
    else:
        all_errors = run_all(device)

    analyze(all_errors)


if __name__ == "__main__":
    main()
