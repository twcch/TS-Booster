#!/usr/bin/env python3
"""Recompute backtest metrics under a consistent simple-return convention.

Background
----------
The original backtest computed daily strategy returns as ``w * log_return``
and then compounded them as ``prod(1 + r_s)``.  Mixing a log return into a
simple-return compounding formula is internally inconsistent: the resulting
NAV/cumulative-return figures do not correspond to any implementable strategy.

Fix (mirrored in experiments/backtest.py): the daily strategy return is
``w * expm1(log_return)`` — the actual one-day P&L of holding weight ``w`` —
which compounds correctly as ``prod(1 + r_s)``.

This script re-derives all per-dataset Sharpe ratios and cumulative returns
from the archived daily series in
``experiments/comprehensive_results/backtest/backtest_daily_returns.json``
(whose stored strategy series are still ``w * log_return``; the weight is
recovered elementwise as ``strat / actual`` and re-applied to the simple
return).  No model retraining is involved.

Result summary (Linear vs Linear+TSB), corrected convention:
  * Sharpe improves on 13/16 datasets (unchanged); declines remain
    2330 (-0.080), 2881 (-0.027), CAD/TWD (-0.019).
  * Top-3 Sharpe gains: 2603 +0.291, EUR/USD +0.102, EUR/TWD +0.090
    (0050 moves to 4th at +0.085).
  * 2603 cumulative return: +62.1% -> +125.6% (relative NAV lift +39.2%).
  * 0050 cumulative return: +181.2% -> +243.9% (relative NAV lift +22.3%).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "experiments", "comprehensive_results",
                   "backtest", "backtest_daily_returns.json")

ORDER = ["USD_TWD", "EUR_TWD", "JPY_TWD", "CNY_TWD", "THB_TWD", "KRW_TWD",
         "CAD_TWD", "EUR_USD", "GBP_USD", "USD_JPY", "0050_TW", "2317_TW",
         "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
STRATS = ["Historical Vol", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]


def sharpe(sr):
    sr = np.asarray(sr)
    return float(np.sqrt(252) * sr.mean() / sr.std())


def cum(sr):
    return float(np.prod(1 + np.asarray(sr)) - 1.0)


def main():
    d = json.load(open(SRC))
    out = {}
    for ds in ORDER:
        rec = d[ds]
        act = np.asarray(rec["actual_returns"], dtype=float)
        simple = np.expm1(act)
        row = {}
        for strat in STRATS:
            sr_old = np.asarray(rec[strat], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(act != 0, sr_old / act, 0.0)
            wa = w[act != 0]
            assert wa.min() > 0.19 and wa.max() < 3.01, (ds, strat)
            sr_new = w * simple
            row[strat] = {
                "sharpe_old": sharpe(sr_old), "sharpe_new": sharpe(sr_new),
                "cum_return_old": cum(sr_old), "cum_return_new": cum(sr_new),
            }
        lin, tsb = row["Linear"], row["Linear+TSB"]
        row["tsb_minus_linear_sharpe_new"] = tsb["sharpe_new"] - lin["sharpe_new"]
        row["tsb_vs_linear_nav_lift_new"] = (
            (1 + tsb["cum_return_new"]) / (1 + lin["cum_return_new"]) - 1)
        out[ds] = row

    dst = os.path.join(HERE, "backtest_corrected_results.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)

    ups = [ds for ds in ORDER if out[ds]["tsb_minus_linear_sharpe_new"] > 0]
    print(f"Sharpe up: {len(ups)}/16")
    for ds in sorted(ORDER, key=lambda x: -out[x]["tsb_minus_linear_sharpe_new"]):
        r = out[ds]
        print(f"  {ds:10s} dSR {r['tsb_minus_linear_sharpe_new']:+.3f}  "
              f"cum {r['Linear']['cum_return_new']*100:+7.1f}% -> "
              f"{r['Linear+TSB']['cum_return_new']*100:+7.1f}%  "
              f"lift {r['tsb_vs_linear_nav_lift_new']*100:+6.1f}%")


if __name__ == "__main__":
    main()
