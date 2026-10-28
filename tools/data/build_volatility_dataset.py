"""Build TS-Booster's volatility CSVs (date, log_return, Volatility) from close prices.

Same features as docs/TS-Booster_ori/src/utils/data_loader.py::prepare_data:
log_return = ln(C_t / C_{t-1}) with the first value set to 0, and
Volatility = rolling(7).std(ddof=1) * sqrt(252) * 100; rows without a full
window are dropped. Train with --data custom --features MS --target Volatility
--enc_in 2: the loader puts Volatility last and fits the scaler on TRAIN only.

Daily moves beyond --jump_threshold are reported (not altered): unadjusted
splits such as 0050_TW on 2014-01-02 distort the volatility target.

--observed_window_features H adds obs_vol_h1..obs_vol_hH: for step h, the
annualized std of the returns of the target window y_{t+h} already known at the
origin t, i.e. std(r_{t+h-6}, ..., r_t) (the paper's Eq. 19 rule). Their shorter
windows drop no extra rows, so splits match the two-feature data.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def observed_window_columns(horizon, window=7):
    """Feature names for steps whose target window still has at least two known returns."""
    if horizon < 0 or (horizon and window - horizon < 2):
        raise ValueError(f"observed-window features need window - h >= 2 for every h <= {horizon}")
    return [f"obs_vol_h{h}" for h in range(1, horizon + 1)]


def volatility_frame(prices, window=7, annualization=252, scale=100.0, observed_window_features=0):
    frame = prices.rename(columns={"Date": "date"})
    if not {"date", "Close"} <= set(frame):
        raise ValueError("price CSV needs date (or Date) and Close columns")
    frame = frame[["date", "Close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    close = frame["Close"].astype(float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Close must be finite and positive")
    frame["log_return"] = np.log(close / close.shift(1)).fillna(0.0)
    observed = observed_window_columns(observed_window_features, window)
    for h, name in enumerate(observed, start=1):
        frame[name] = frame["log_return"].rolling(window - h).std() * np.sqrt(annualization) * scale
    frame["Volatility"] = frame["log_return"].rolling(window).std() * np.sqrt(annualization) * scale
    frame = frame.dropna().reset_index(drop=True)
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame[["date", "log_return", *observed, "Volatility"]]


def large_moves(frame, threshold):
    moves = frame.loc[frame["log_return"].abs() > threshold, ["date", "log_return"]]
    return [{"date": row.date, "log_return": float(row.log_return)} for row in moves.itertuples()]


def build(source, output_dir, *, window=7, annualization=252, scale=100.0, jump_threshold=0.2, overwrite=False,
          observed_window_features=0):
    target = Path(output_dir) / Path(source).name
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite deliberately")
    frame = volatility_frame(pd.read_csv(source), window, annualization, scale, observed_window_features)
    observed = observed_window_columns(observed_window_features, window)
    csv = frame.to_csv(index=False).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(csv)
    report = {"source": str(source), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(csv).hexdigest(), "rows": len(frame),
              "start": frame["date"].iloc[0], "end": frame["date"].iloc[-1],
              "features": {"log_return": "ln(C_t / C_{t-1}), first value 0",
                           "Volatility": f"rolling({window}).std(ddof=1) * sqrt({annualization}) * {scale}",
                           **{name: f"rolling({window - h}).std(ddof=1) * sqrt({annualization}) * {scale}: "
                                    f"known part of the target window of step {h} (Eq. 19)"
                              for h, name in enumerate(observed, start=1)}},
              "enc_in": 2 + len(observed),
              "dropped_warmup_rows": window - 1, "jump_threshold": jump_threshold,
              "large_moves": large_moves(frame, jump_threshold)}
    target.with_suffix(".csv.manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("dataset/raw"), help="CSV file or folder of CSVs")
    parser.add_argument("--output", type=Path, default=Path("dataset/volatility"))
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--annualization", type=int, default=252)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--jump_threshold", type=float, default=0.2,
                        help="report |log return| above this (0.2 is about a 18%% to 22%% daily move)")
    parser.add_argument("--observed_window_features", type=int, default=0, metavar="H",
                        help="add obs_vol_h1..obs_vol_hH (Eq. 19 known-return features); train with --enc_in 2+H")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.window < 2 or args.annualization <= 0 or args.scale <= 0 or args.jump_threshold <= 0:
        parser.error("window >= 2 and positive annualization, scale and jump_threshold are required")
    try:
        observed_window_columns(args.observed_window_features, args.window)
    except ValueError as error:
        parser.error(str(error))
    sources = [args.input] if args.input.is_file() else sorted(args.input.glob("*.csv"))
    if not sources:
        parser.error(f"No CSV found in {args.input}")
    for source in sources:
        target, report = build(source, args.output, window=args.window, annualization=args.annualization,
                               scale=args.scale, jump_threshold=args.jump_threshold, overwrite=args.overwrite,
                               observed_window_features=args.observed_window_features)
        print(f"{source.stem:10s} {report['rows']:6d} rows {report['start']} .. {report['end']} -> {target}")
        for move in report["large_moves"]:
            print(f"  WARNING {move['date']}: log return {move['log_return']:+.3f} (check splits/adjustment)",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
