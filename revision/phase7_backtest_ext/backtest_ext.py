"""
Phase 7: Backtest Extension

Extends the existing backtest with:
  1. Sortino ratio (downside deviation)
  2. Sharpe significance test (bootstrap, Ledoit-Wolf inspired)
  3. Buy-and-hold baseline (annualized reference)
  4. Summary table with all metrics

Loads from existing backtest_results.json + backtest_daily_returns.json
No re-run needed.

Usage:
  uv run python revision/phase7_backtest_ext/backtest_ext.py
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))

RESULTS_PATH      = Path("experiments/comprehensive_results/backtest/backtest_results.json")
DAILY_RETURNS_PATH = Path("experiments/comprehensive_results/backtest/backtest_daily_returns.json")
OUT_DIR           = Path("revision/phase7_backtest_ext")

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

METHODS = ["Historical Vol", "XGBoost", "LightGBM", "Linear", "Linear+TSB"]


# ── Metrics ────────────────────────────────────────────────────────────────────

def sortino_ratio(daily_returns: list[float], rf: float = 0.0, ann: int = 252) -> float:
    r = np.array(daily_returns)
    mean_r = r.mean() * ann - rf
    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan
    downside_std = np.std(downside, ddof=1) * np.sqrt(ann)
    return float(mean_r / downside_std) if downside_std > 0 else np.nan


def sharpe_ratio(daily_returns: list[float], rf: float = 0.0, ann: int = 252) -> float:
    r = np.array(daily_returns)
    mean_r = r.mean() * ann - rf
    std_r  = r.std(ddof=1) * np.sqrt(ann)
    return float(mean_r / std_r) if std_r > 0 else np.nan


def max_drawdown(daily_returns: list[float]) -> float:
    r = np.array(daily_returns)
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar_ratio(daily_returns: list[float], ann: int = 252) -> float:
    r = np.array(daily_returns)
    ann_return = r.mean() * ann
    mdd = max_drawdown(daily_returns)
    return float(ann_return / abs(mdd)) if mdd != 0 else np.nan


def bootstrap_sharpe_diff_pvalue(r1: list[float], r2: list[float],
                                  n_boot: int = 5000, ann: int = 252) -> float:
    """
    Bootstrap p-value for H1: Sharpe(r1) > Sharpe(r2).
    Uses circular block bootstrap to preserve autocorrelation.
    Block length ≈ T^(1/3).
    """
    r1, r2 = np.array(r1), np.array(r2)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    obs_diff = sharpe_ratio(r1, ann=ann) - sharpe_ratio(r2, ann=ann)

    block = max(1, int(n ** (1 / 3)))
    boot_diffs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        s1 = sharpe_ratio(r1[idx], ann=ann)
        s2 = sharpe_ratio(r2[idx], ann=ann)
        boot_diffs.append(s1 - s2)

    # one-sided p-value for H1: Sharpe(r1) > Sharpe(r2)
    # = fraction of bootstrap samples where boot_diff <= 0 (consistent with H0)
    p = float(np.mean(np.array(boot_diffs) <= 0))
    return p


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze():
    results       = json.loads(RESULTS_PATH.read_text())
    daily_returns = json.loads(DAILY_RETURNS_PATH.read_text())

    ext_results = {}

    for ds in ALL_DATASETS:
        if ds not in results or ds not in daily_returns:
            continue

        dr = daily_returns[ds]
        actual_returns = dr.get("actual_returns", [])

        ds_ext = {}

        # Buy-and-hold
        bh_sharpe  = sharpe_ratio(actual_returns)
        bh_sortino = sortino_ratio(actual_returns)
        bh_mdd     = max_drawdown(actual_returns)
        bh_calmar  = calmar_ratio(actual_returns)
        ds_ext["Buy&Hold"] = {
            "sharpe": bh_sharpe, "sortino": bh_sortino,
            "mdd": bh_mdd, "calmar": bh_calmar,
        }

        tsb_returns = dr.get("Linear+TSB", [])

        for m in METHODS:
            if m not in dr:
                continue
            ret = dr[m]
            s    = sharpe_ratio(ret)
            sort = sortino_ratio(ret)
            mdd  = max_drawdown(ret)
            cal  = calmar_ratio(ret)

            # Bootstrap Sharpe significance vs buy-and-hold
            p_vs_bh  = bootstrap_sharpe_diff_pvalue(ret, actual_returns) if actual_returns else None
            # Bootstrap Sharpe: TSB vs this method
            p_tsb_vs = bootstrap_sharpe_diff_pvalue(tsb_returns, ret) if (m != "Linear+TSB" and tsb_returns) else None

            ds_ext[m] = {
                "sharpe": s, "sortino": sort, "mdd": mdd, "calmar": cal,
                "p_vs_buyhold": p_vs_bh,
                "p_tsb_beats": p_tsb_vs,
            }

        ext_results[ds] = ds_ext

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("Extended Backtest: Sharpe / Sortino / MDD / Calmar")
    print("(** p<0.05 that TSB Sharpe > this method via bootstrap; B&H = buy-and-hold)")
    print("=" * 110)

    for ds in ALL_DATASETS:
        if ds not in ext_results:
            continue
        print(f"\n── {ds} ──────────────────────────────")
        print(f"{'Method':<16} {'Sharpe':>8} {'Sortino':>9} {'MDD':>8} {'Calmar':>8} {'p(TSB>)':>10}")
        print("-" * 75)

        # Buy&hold
        bh = ext_results[ds]["Buy&Hold"]
        print(f"{'Buy&Hold':<16} {bh['sharpe']:>8.3f} {bh['sortino']:>9.3f} {bh['mdd']:>8.3f} {bh['calmar']:>8.3f}   (baseline)")

        for m in METHODS:
            if m not in ext_results[ds]:
                continue
            e = ext_results[ds][m]
            p_str = f"{e['p_tsb_beats']:.3f}" if e.get("p_tsb_beats") is not None else "  —  "
            sig   = "**" if e.get("p_tsb_beats") is not None and e["p_tsb_beats"] < 0.05 else "  "
            print(f"{m:<16} {e['sharpe']:>8.3f} {e['sortino']:>9.3f} {e['mdd']:>8.3f} {e['calmar']:>8.3f}  {p_str}{sig}")

    # ── Cross-dataset summary ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Cross-dataset Averages (16 datasets)")
    print("=" * 80)
    print(f"{'Method':<16} {'Avg Sharpe':>11} {'Avg Sortino':>12} {'Avg MDD':>9} {'TSB sig wins':>13}")
    print("-" * 65)
    all_methods_ext = ["Buy&Hold"] + METHODS
    for m in all_methods_ext:
        sharpes  = [ext_results[ds][m]["sharpe"]  for ds in ALL_DATASETS if ds in ext_results and m in ext_results[ds]]
        sortinos = [ext_results[ds][m]["sortino"] for ds in ALL_DATASETS if ds in ext_results and m in ext_results[ds]]
        mdds     = [ext_results[ds][m]["mdd"]     for ds in ALL_DATASETS if ds in ext_results and m in ext_results[ds]]
        avg_s  = np.nanmean(sharpes)
        avg_so = np.nanmean(sortinos)
        avg_mdd = np.nanmean(mdds)

        if m not in ("Buy&Hold", "Linear+TSB"):
            sig_wins = sum(
                1 for ds in ALL_DATASETS
                if ds in ext_results and m in ext_results[ds]
                and ext_results[ds][m].get("p_tsb_beats") is not None
                and ext_results[ds][m]["p_tsb_beats"] < 0.05
            )
            n = sum(1 for ds in ALL_DATASETS if ds in ext_results and m in ext_results[ds])
            sig_str = f"{sig_wins}/{n}"
        else:
            sig_str = "—"

        print(f"{m:<16} {avg_s:>11.3f} {avg_so:>12.3f} {avg_mdd:>9.3f} {sig_str:>13}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "backtest_ext_results.json"
    out_path.write_text(json.dumps(ext_results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    analyze()
