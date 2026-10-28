"""The 0050 case (R1.5/R1.11), KRW/TWD's measurement limitation (§4.3) and the dataset summary (spec §5.9).

    python analysis/case_studies.py [--results results] [--output analysis/output/case_studies]
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (BACKBONE_LABEL, BACKBONES, DATASETS, GROUPS, SEEDS, SEEDS_3, choose_learning_rate,  # noqa: E402
                             classical, deep, display, load_runs, load_volatility, require, test_origins, tsb,
                             write_json, write_table)
from analysis.stats import compare  # noqa: E402

FOCUS, KRW = "0050_TW", "KRW_TWD"
WARMUP = 6  # volatility rows start at the seventh price (rolling window of 7 returns, the first return being 0)
# article_v2 §4.1.2, §4.2.1-4.3 and Tables 4-6.
PAPER = {"krw": {"distinct_closes": 49, "zero_return_share_pct": 41.3, "unchanged_volatility_share_pct": 9.8, "min_increment_pct": 0.43},
         "test_mean_volatility_range": {"Equities": (19.7, 45.2), "FX": (4.7, 8.6)},
         "focus": {"linear_reduction_pct": 27.25, "transformer_reduction_pct": 5.9, "tsb_rmse": 7.5071, "har_rv_rmse": 6.5019,
                   "persistence_rmse": 6.8347, "arima_rmse": 6.8347, "dlinear_rmse": 6.4404}}


def _errors(rows, data_root):
    """One record per test sample of every run: dataset, seed, origin date, anchor and the horizon-mean error."""
    parts = []
    for row in rows.itertuples():
        pred, true, last = (np.load(Path(row.path) / f"{name}.npy") for name in ("pred", "true", "last"))
        _, dates = test_origins(row.dataset, len(pred), data_root)
        parts.append(pd.DataFrame({"dataset": row.dataset, "seed": int(row.seed), "date": pd.to_datetime(dates),
                                   "anchor": last[:, 0], "error": (pred - true)[:, :, 0].mean(axis=1),
                                   "squared": ((pred - true)[:, :, 0] ** 2).mean(axis=1)}))
    return pd.concat(parts, ignore_index=True)


def unchanged_share(values, rows, horizon):
    """% of (origin, step) pairs whose target equals the origin's value, on the exact float64 data."""
    rows = np.asarray(rows)
    return float(100 * np.mean([values[rows + h] == values[rows] for h in range(1, horizon + 1)]))


def _bias(errors, by):
    grouped = errors.groupby(by, observed=True)
    return pd.DataFrame({"N": grouped.size(), "Mean error": grouped["error"].mean(),
                         "RMSE": np.sqrt(grouped["squared"].mean())}).reset_index()


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    focus = FOCUS if FOCUS in datasets else datasets[0]
    vanilla = require(frame, tsb("TSBooster-vanilla"), datasets, seeds, "Vanilla Linear", allow_missing)
    booster = require(frame, tsb(), datasets, seeds, "TS-Booster+Linear", allow_missing)
    present = [d for d in datasets if d in set(vanilla["dataset"])]

    # Dataset summary: periods, splits and the test-period volatility level.
    rows, test_level = [], {}
    for dataset in present:
        volatility = load_volatility(dataset, data_root)
        raw = pd.read_csv(Path(data_root) / "raw" / f"{dataset}.csv")
        n = len(volatility)
        num_test, num_train = int(0.2 * n), int(0.7 * n)
        num_val = n - num_train - num_test
        reference = vanilla[(vanilla["dataset"] == dataset) & (vanilla["seed"] == seeds[0])].iloc[0]
        true = np.load(Path(reference.path) / "true.npy")
        test_level[dataset] = float(true.mean())
        rows.append({"Dataset": display(dataset), "Start": volatility["date"].iloc[0], "End": volatility["date"].iloc[-1],
                     "Prices": len(raw), "Rows": n, "Train end": volatility["date"].iloc[num_train - 1],
                     "Validation end": volatility["date"].iloc[num_train + num_val - 1],
                     "First test origin": volatility["date"].iloc[n - num_test - 1], "Test samples": len(true),
                     "Test mean volatility": test_level[dataset]})
    write_table(pd.DataFrame(rows), out / "table_datasets", decimals=2,
                caption="Sample periods, chronological 70/10/20 splits and the mean annualized volatility of the test targets.")
    level_range = {}
    for group, members in (("FX", GROUPS["TWD FX"] + GROUPS["International FX"]), ("Equities", GROUPS["Equities"])):
        values = [test_level[d] for d in members if d in test_level]
        if values:
            level_range[group] = (min(values), max(values))

    # KRW/TWD: coarse quotes.
    krw = None
    if KRW in present:
        volatility = load_volatility(KRW, data_root)
        raw = pd.read_csv(Path(data_root) / "raw" / f"{KRW}.csv")
        n = len(volatility)
        test_rows = np.arange(n - int(0.2 * n), n)
        closes = raw["Close"].to_numpy(dtype=float)[test_rows + WARMUP]
        reference = vanilla[(vanilla["dataset"] == KRW) & (vanilla["seed"] == seeds[0])].iloc[0]
        origins, _ = test_origins(KRW, len(np.load(Path(reference.path) / "last.npy")), data_root)
        distinct = np.unique(closes)
        increments = np.diff(distinct)
        krw = {"distinct_closes": int(len(distinct)),
               "zero_return_share_pct": float(100 * np.mean(volatility["log_return"].to_numpy()[test_rows] == 0)),
               "unchanged_volatility_share_pct": unchanged_share(volatility["Volatility"].to_numpy(dtype=float), origins,
                                                                 int(reference.pred_len)),
               "min_increment_pct": float(100 * increments[increments > 0].min() / np.median(closes)) if (increments > 0).any() else math.nan}
        write_table(pd.DataFrame([{"Quantity": "Distinct closing prices in the test period", "Value": krw["distinct_closes"]},
                                  {"Quantity": "Zero daily log returns in the test period (%)", "Value": krw["zero_return_share_pct"]},
                                  {"Quantity": "Origin–horizon pairs with unchanged volatility (%)", "Value": krw["unchanged_volatility_share_pct"]},
                                  {"Quantity": "Smallest price increment relative to the median close (%)", "Value": krw["min_increment_pct"]}]),
                    out / "table_krw", decimals=2, caption="KRW/TWD quotes to four decimals: the measurement limitation behind its MDA.")

    # The focus dataset across backbones (R1.11) and against the comparators (R1.5).
    backbone_rows, backbone_rank = [], {}
    for backbone in BACKBONES:
        table = compare(frame, tsb("TSBooster-vanilla", backbone), tsb("TSBooster", backbone), present, seeds,
                        allow_missing=allow_missing, labels=(f"Vanilla {backbone}", f"TS-Booster {backbone}")).set_index("dataset")
        ranks = table["reduction_pct"].rank(ascending=False)
        backbone_rank[backbone] = (float(table.loc[focus, "reduction_pct"]), int(ranks.loc[focus]))
        backbone_rows.append({"Backbone": BACKBONE_LABEL[backbone], "Vanilla RMSE": table.loc[focus, "baseline_mean"],
                              "+TS-Booster RMSE": table.loc[focus, "candidate_mean"], "Reduction %": table.loc[focus, "reduction_pct"],
                              "Rank among datasets": f"{int(ranks.loc[focus])}/{len(table)}", "Adjusted p": table.loc[focus, "p_holm"]})
    write_table(pd.DataFrame(backbone_rows), out / "table_0050_backbones", pct_columns=["Reduction %"], p_columns=["Adjusted p"],
                caption=f"{display(focus)}: RMSE reduction by backbone and its rank among the datasets (1 = largest reduction).")
    search_seeds = [seed for seed in SEEDS_3 if seed in seeds] or list(seeds)
    chosen = choose_learning_rate(frame, deep("DLinear-paper"), [focus], search_seeds)
    comparators = {"TS-Booster+Linear": (tsb(), seeds), "Vanilla Linear": (tsb("TSBooster-vanilla"), seeds),
                   "HAR-RV": (classical("HAR-RV"), [0]), "Persistence": (classical("Persistence"), [0]),
                   "ARIMA": (classical("ARIMA"), [0]), "XGBoost": (classical("XGBoost"), seeds),
                   "DLinear": (deep("DLinear-paper", float(chosen["chosen"].iloc[0])), seeds)}
    comparator_rows, focus_rmse = [], {}
    for label, (filters, method_seeds) in comparators.items():
        subset = require(frame, filters, [focus], method_seeds, label, allow_missing)
        means = subset[["rmse", "mae", "mda"]].mean()
        focus_rmse[label] = float(means["rmse"])
        p = math.nan
        if label != "TS-Booster+Linear":
            p = float(compare(frame, filters, tsb(), [focus], seeds, base_seeds=method_seeds if method_seeds == [0] else None,
                              allow_missing=allow_missing, labels=(label, "TS-Booster+Linear"))["p"].iloc[0])
        comparator_rows.append({"Method": display(label), "RMSE": float(means["rmse"]), "MAE": float(means["mae"]),
                                "MDA": float(means["mda"]), "p (TS-Booster better)": p})
    write_table(pd.DataFrame(comparator_rows), out / "table_0050_comparators", bold=["RMSE"], p_columns=["p (TS-Booster better)"],
                caption=f"{display(focus)}: TS-Booster+Linear against the closest comparators (one-sided seed-based p, uncorrected).")

    # Bias by volatility regime and by year (R1.5), on the focus dataset and pooled over all datasets.
    frames = []
    for label, rows_ in (("TS-Booster+Linear", booster), ("Vanilla Linear", vanilla)):
        errors = _errors(rows_, data_root)
        errors["Model"] = label
        errors["Tercile"] = errors.groupby("dataset")["anchor"].transform(
            lambda a: pd.qcut(a.rank(method="first"), 3, labels=["low", "mid", "high"]))
        errors["Year"] = errors["date"].dt.year
        frames.append(errors)
    errors = pd.concat(frames, ignore_index=True)
    tercile = pd.concat([_bias(errors[errors["dataset"] == focus], ["Model", "Tercile"]).assign(Scope=display(focus)),
                         _bias(errors, ["Model", "Tercile"]).assign(Scope="All datasets")], ignore_index=True)
    year = pd.concat([_bias(errors[errors["dataset"] == focus], ["Model", "Year"]).assign(Scope=display(focus)),
                      _bias(errors, ["Model", "Year"]).assign(Scope="All datasets")], ignore_index=True)
    write_table(tercile[["Scope", "Model", "Tercile", "N", "Mean error", "RMSE"]], out / "table_0050_bias_tercile",
                caption="Mean forecast error (prediction − target, averaged over the three steps) and RMSE by tercile of the "
                        "last observed volatility within each dataset; all seeds pooled.")
    write_table(year[["Scope", "Model", "Year", "N", "Mean error", "RMSE"]], out / "table_0050_bias_year",
                caption="Mean forecast error and RMSE by calendar year of the forecast origin; all seeds pooled.")

    summary = {"datasets_table_rows": len(rows), "test_mean_volatility_range": level_range, "krw": krw,
               "focus_dataset": focus, "focus_backbone_rank": backbone_rank, "focus_vs_comparators": focus_rmse,
               "focus_loses_to_har_rv": bool(focus_rmse["TS-Booster+Linear"] > focus_rmse["HAR-RV"]),
               "focus_best_comparator": min(focus_rmse, key=focus_rmse.get)}
    write_json({**summary, "paper": PAPER, "datasets": rows, "comparators": comparator_rows,
                "bias_tercile": tercile.to_dict("records"), "bias_year": year.to_dict("records")}, out / "case_studies.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "case_studies")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print(f"{summary['focus_dataset']}: {summary['focus_backbone_rank']}; loses to HAR-RV: {summary['focus_loses_to_har_rv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
