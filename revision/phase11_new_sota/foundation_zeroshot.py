"""
Phase 11b: Zero-shot Foundation Model Baselines (TimesFM, Chronos, Moirai)

Compares Linear+TS-Booster against 3 pretrained time-series foundation
models in a ZERO-SHOT setting (no training/fine-tuning on our datasets) --
this is the "Foundation Model" extension from the reviewer's SOTA list,
kept separate from the trained baselines in run_new_sota.py.

Each model only sees the univariate Volatility series (its own context
window of `SEQ_LEN`=30 raw values) and forecasts `PRED_LEN`=3 steps ahead.
Evaluation uses the SAME test split as prepare_data() (last 20% of each
dataset), in the ORIGINAL (un-scaled) Volatility units.

REQUIRED DEPENDENCIES (none of these are in pyproject.toml -- install only
the ones you want to run; each has its own heavy deps and may need a
separate venv if there are version conflicts with torch==2.7.1+cu118):

  Chronos (Amazon, T5-based):
    uv pip install chronos-forecasting

  TimesFM (Google):
    uv pip install timesfm
    # NOTE: TimesFM v1 requires context_len to be a multiple of 32 (its
    # patch size). We left-pad the 30-step context with the first value
    # to 32 steps (PAD_LEN below).

  Moirai (Salesforce, uni2ts):
    uv pip install uni2ts gluonts

NOTE: This script was written against each library's documented zero-shot
API as of mid-2025. These are fast-moving research repos -- if an API has
changed since, you may need to adjust the `run_*` function for that model
(they are independent, so fixing one does not affect the others).

Usage:
  uv run python revision/phase11_new_sota/foundation_zeroshot.py --model chronos
  uv run python revision/phase11_new_sota/foundation_zeroshot.py --model timesfm
  uv run python revision/phase11_new_sota/foundation_zeroshot.py --model moirai
  uv run python revision/phase11_new_sota/foundation_zeroshot.py --model all
  uv run python revision/phase11_new_sota/foundation_zeroshot.py --analyze
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import mean_squared_error

sys.path.append(str(Path(__file__).parent.parent.parent))

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEQ_LEN = 30
PRED_LEN = 3
PAD_LEN = 32  # TimesFM v1 patch size requires context_len % 32 == 0

EXISTING_SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
RESULTS_PATH = Path("revision/phase11_new_sota/foundation_zeroshot_results.json")


def get_test_windows(ds, lookback=SEQ_LEN, horizon=PRED_LEN):
    """Raw (un-scaled) Volatility test windows, using the same 80% split
    point as prepare_data() for a fair comparison."""
    df = pd.read_csv(f"dataset/{ds}.csv")
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1)).fillna(0.0)
    df["Volatility"] = df["log_return"].rolling(7).std() * np.sqrt(252) * 100
    df = df.dropna().reset_index(drop=True)

    vol = df["Volatility"].values
    split_2 = int(len(vol) * 0.80)
    test_vol = vol[split_2 - lookback:]

    X, y = [], []
    for i in range(len(test_vol) - lookback - horizon + 1):
        X.append(test_vol[i:i + lookback])
        y.append(test_vol[i + lookback:i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def compute_metrics(preds, targets, last_obs):
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    pred_dir = np.sign(preds - last_obs[:, None])
    actual_dir = np.sign(targets - last_obs[:, None])
    mda = float(np.mean(pred_dir == actual_dir))
    return {"rmse": rmse, "mae": mae, "mda": mda}


# ──────────────────────────────────────────────────────────────
# Chronos (amazon/chronos-forecasting)
# ──────────────────────────────────────────────────────────────

def run_chronos(model_name="amazon/chronos-t5-small", batch_size=64, num_samples=20):
    from chronos import ChronosPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = ChronosPipeline.from_pretrained(
        model_name, device_map=device, torch_dtype=torch.float32
    )

    results = {}
    for ds in ALL_DATASETS:
        X, y = get_test_windows(ds)
        last_obs = X[:, -1]
        preds_all = []
        for i in range(0, len(X), batch_size):
            context = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            forecast = pipeline.predict(
                context=context, prediction_length=PRED_LEN, num_samples=num_samples
            )  # [batch, num_samples, PRED_LEN]
            preds_all.append(np.median(forecast.numpy(), axis=1))
        preds = np.concatenate(preds_all, axis=0)
        results[ds] = compute_metrics(preds, y, last_obs)
        print(f"  Chronos {ds:<12}: RMSE={results[ds]['rmse']:.4f} "
              f"MAE={results[ds]['mae']:.4f} MDA={results[ds]['mda']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# TimesFM (google/timesfm)
# ──────────────────────────────────────────────────────────────

def run_timesfm(repo_id="google/timesfm-1.0-200m-pytorch"):
    import timesfm

    backend = "gpu" if torch.cuda.is_available() else "cpu"
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            per_core_batch_size=32,
            horizon_len=PRED_LEN,
            context_len=PAD_LEN,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=repo_id),
    )

    results = {}
    for ds in ALL_DATASETS:
        X, y = get_test_windows(ds)
        last_obs = X[:, -1]
        # left-pad 30 -> 32 steps with the first observed value (edge padding)
        pad_width = PAD_LEN - SEQ_LEN
        X_padded = np.pad(X, ((0, 0), (pad_width, 0)), mode="edge")
        inputs = [row for row in X_padded]
        point_forecast, _ = tfm.forecast(inputs, freq=[0] * len(inputs))
        preds = np.array(point_forecast)[:, :PRED_LEN]
        results[ds] = compute_metrics(preds, y, last_obs)
        print(f"  TimesFM {ds:<12}: RMSE={results[ds]['rmse']:.4f} "
              f"MAE={results[ds]['mae']:.4f} MDA={results[ds]['mda']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# Moirai (Salesforce/uni2ts)
# ──────────────────────────────────────────────────────────────

def run_moirai(model_size="small", batch_size=32, num_samples=20, patch_size=8):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
        prediction_length=PRED_LEN,
        context_length=SEQ_LEN,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    ).to(device)
    predictor = model.create_predictor(batch_size=batch_size)

    results = {}
    for ds in ALL_DATASETS:
        X, y = get_test_windows(ds)
        last_obs = X[:, -1]
        entries = [
            {FieldName.START: pd.Timestamp("2000-01-01"), FieldName.TARGET: row}
            for row in X
        ]
        dataset = ListDataset(entries, freq="D")
        forecasts = list(predictor.predict(dataset))
        preds = np.array([f.median for f in forecasts])[:, :PRED_LEN]
        results[ds] = compute_metrics(preds, y, last_obs)
        print(f"  Moirai  {ds:<12}: RMSE={results[ds]['rmse']:.4f} "
              f"MAE={results[ds]['mae']:.4f} MDA={results[ds]['mda']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────

def analyze(results):
    existing = {}
    if EXISTING_SOTA_PATH.exists():
        with open(EXISTING_SOTA_PATH) as f:
            raw = json.load(f)
        for ds in ALL_DATASETS:
            if ds in raw:
                existing[ds] = raw[ds]

    fm_models = list(next(iter(results.values())).keys()) if results else []
    methods = ["Linear", "Linear+TSB"] + fm_models

    print("\n" + "=" * 120)
    print("PHASE 11b: ZERO-SHOT FOUNDATION MODELS vs Linear / Linear+TSB (RMSE)")
    print("=" * 120)
    header = f"{'Dataset':<12}"
    for m in methods:
        header += f" | {m:>14}"
    print(header)
    print("-" * len(header))

    for ds in ALL_DATASETS:
        line = f"  {ds:<12}"
        for m in methods:
            if m in ("Linear", "Linear+TSB"):
                val = existing.get(ds, {}).get(m, {})
                val = float(np.mean(val["rmse"])) if isinstance(val, dict) and "rmse" in val else None
            else:
                val = results.get(ds, {}).get(m, {}).get("rmse")
            line += f" | {val:>14.4f}" if val is not None else f" | {'N/A':>14}"
        print(line)

    print("-" * len(header))
    avg_line = "  Average     "
    for m in methods:
        if m in ("Linear", "Linear+TSB"):
            vals = [float(np.mean(existing.get(ds, {}).get(m, {}).get("rmse", []) or [np.nan])) for ds in ALL_DATASETS]
        else:
            vals = [results.get(ds, {}).get(m, {}).get("rmse") for ds in ALL_DATASETS]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        avg_line += f" | {np.mean(vals):>14.4f}" if vals else f" | {'N/A':>14}"
    print(avg_line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "timesfm", "moirai", "all"], default="all")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {ds: {} for ds in ALL_DATASETS}

    if args.analyze:
        analyze(results)
        return

    runners = {"chronos": run_chronos, "timesfm": run_timesfm, "moirai": run_moirai}
    models_to_run = list(runners.keys()) if args.model == "all" else [args.model]

    for name in models_to_run:
        print(f"\n{'#'*60}\n  {name}\n{'#'*60}")
        try:
            model_results = runners[name]()
        except ImportError as e:
            print(f"  SKIP {name}: missing dependency ({e}). See module docstring for install instructions.")
            continue

        for ds in ALL_DATASETS:
            results.setdefault(ds, {})[name] = model_results[ds]

        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    analyze(results)


if __name__ == "__main__":
    main()
