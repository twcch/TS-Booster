"""Fig. 11 and the extended inverse-volatility backtest of R1.7 (spec §5.8): every seed, transaction
costs, several volatility targets, more comparators, and Ledoit–Wolf tests of the Sharpe ratios.

    python analysis/backtest.py [--results results] [--output analysis/output/backtest] [--bootstrap 5000]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (DATASETS, PALETTE, SEEDS, classical, display, load_runs, load_volatility,  # noqa: E402
                             next_day_return, require, test_origins, tsb, write_figure, write_json, write_table)
from analysis.stats import holm_adjusted, inverse_volatility, ledoit_wolf_pvalue  # noqa: E402

VANILLA, TSB = "Vanilla Linear", "TS-Booster+Linear"
METHODS = {VANILLA: tsb("TSBooster-vanilla"), TSB: tsb(), "Persistence": classical("Persistence"),
           "HAR-RV": classical("HAR-RV"), "GJR-GARCH-aligned": classical("GJR-GARCH-aligned"),
           "Observed-window rule": classical("ObservedWindowRule")}
DETERMINISTIC = {"Persistence", "HAR-RV", "GJR-GARCH-aligned", "Observed-window rule"}
TARGETS = (10.0, 15.0, 20.0)   # annualized volatility target σ* (%)
COSTS_BPS = (0.0, 5.0, 10.0)   # proportional cost per unit of position change
BASE_TARGET, BASE_COST, ALPHA = 15.0, 0.0, 0.05
# article_v2 §4.8 and Fig. 11 (one seed, σ* = 15, no costs).
PAPER = {"improved": "13/16", "nominal_significant": 3, "holm_significant": 0, "min_adjusted_p": 0.1024,
         "top3": [("2603_TW", 0.291), ("EUR_USD", 0.102), ("EUR_TWD", 0.090)],
         "declines": {"2330_TW": -0.080, "2881_TW": -0.027, "CAD_TWD": -0.019},
         "nominal_p": {"EUR_TWD": 0.0074, "EUR_USD": 0.0064, "2603_TW": 0.0102},
         "cumulative_return_pct": {"2603_TW": (62.1, 125.6), "0050_TW": (181.2, 243.9)}}


def _predictions(frame, filters, datasets, seeds, label, allow_missing):
    """{dataset: {seed: (run path, ŷ_{t+1} on the original scale)}}"""
    out = {}
    for row in require(frame, filters, datasets, seeds, label, allow_missing).itertuples():
        out.setdefault(row.dataset, {})[int(row.seed)] = (Path(row.path), np.load(Path(row.path) / "pred.npy")[:, 0, 0].astype(float))
    return out


def check_windows(path, volatility, rows, rtol=1e-4):
    """A run's test windows must sit on the origin rows: last.npy on the origins, true.npy one step on.

    Float32 storage leaves relative gaps of about 1e-6; anything larger means the windows are shifted.
    """
    path = Path(path)
    last, true = np.load(path / "last.npy")[:, 0], np.load(path / "true.npy")[:, 0, 0]
    rows = np.asarray(rows)
    if rows[-1] + 1 >= len(volatility) or not np.allclose(last, volatility[rows], rtol=rtol, atol=0) \
            or not np.allclose(true, volatility[rows + 1], rtol=rtol, atol=0):
        raise ValueError(f"{path.name}: the test windows do not sit on the expected origin rows")


def _delta_figure(table, stem, title, errors=None):
    import matplotlib.pyplot as plt
    order = table.sort_values("ΔSR").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(order) + 1.4))
    colors = [PALETTE[0] if value > 0 else PALETTE[3] for value in order["ΔSR"]]
    xerr = None if errors is None else np.vstack([order["ΔSR"] - order[errors[0]], order[errors[1]] - order["ΔSR"]])
    ax.barh(range(len(order)), order["ΔSR"], color=colors, xerr=xerr, capsize=2)
    for position, row in enumerate(order.to_dict("records")):
        star = " *" if row.get("p", 1.0) < ALPHA else ""
        positive = row["ΔSR"] >= 0
        end = row["ΔSR"] if errors is None else (max(row["ΔSR"], row[errors[1]]) if positive else min(row["ΔSR"], row[errors[0]]))
        ax.text(end + (0.004 if positive else -0.004), position, f"{row['ΔSR']:+.3f}{star}", va="center",
                ha="left" if row["ΔSR"] >= 0 else "right", fontsize=8)
    ax.set_yticks(range(len(order)), order["Dataset"])
    ax.axvline(0, color="black", linewidth=0.6)
    low = float(min(0.0, (order[errors[0]] if errors else order["ΔSR"]).min()))
    high = float(max(0.0, (order[errors[1]] if errors else order["ΔSR"]).max()))
    span = (high - low) or 0.1
    ax.set_xlim(low - 0.28 * span, high + 0.28 * span)  # room for the value labels beside the bars
    ax.set_xlabel("Sharpe-ratio difference (TS-Booster+Linear − Vanilla Linear)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, stem)


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_seed = 42 if 42 in seeds else seeds[0]
    predictions = {label: _predictions(frame, filters, datasets, [0] if label in DETERMINISTIC else seeds, label, allow_missing)
                   for label, filters in METHODS.items()}
    present = [d for d in datasets if all(d in predictions[label] for label in METHODS)]
    returns = {}
    for dataset in present:
        count = len(predictions[VANILLA][dataset][seeds[0]][1])
        rows, _ = test_origins(dataset, count, data_root)
        volatility = load_volatility(dataset, data_root)["Volatility"].to_numpy(dtype=float)
        for label in METHODS:
            for seed, (path, values) in predictions[label][dataset].items():
                if len(values) != count:
                    raise ValueError(f"{label} {dataset} seed {seed}: {len(values)} predictions, expected {count}")
                check_windows(path, volatility, rows)
        returns[dataset] = next_day_return(dataset, rows, data_root)

    def strategy(label, dataset, seed, target=BASE_TARGET, cost=BASE_COST):
        values = predictions[label][dataset][0 if label in DETERMINISTIC else seed][1]
        return inverse_volatility(values, returns[dataset], target=target, cost_bps=cost)

    def lw(candidate, baseline):
        return ledoit_wolf_pvalue(candidate["returns"], baseline["returns"], n_boot=bootstrap)

    # The paper's setting: one seed, σ* = 15, no costs, every seed's p values for the seed table.
    lw_results = {seed: {} for seed in seeds}
    p_values = {seed: {} for seed in seeds}
    deltas = {seed: {} for seed in seeds}
    sharpe_by_seed = {seed: {} for seed in seeds}
    base_rows = []
    for dataset in present:
        for seed in seeds:
            vanilla, booster = strategy(VANILLA, dataset, seed), strategy(TSB, dataset, seed)
            lw_results[seed][dataset] = lw(booster, vanilla)
            p_values[seed][dataset] = lw_results[seed][dataset]["p"]
            deltas[seed][dataset] = booster["sharpe"] - vanilla["sharpe"]
            sharpe_by_seed[seed][dataset] = {VANILLA: vanilla["sharpe"], TSB: booster["sharpe"]}
            if seed == main_seed:
                base_rows.append({"Dataset": display(dataset), "dataset": dataset, "SR Vanilla": vanilla["sharpe"],
                                  "SR TS-Booster": booster["sharpe"], "ΔSR": booster["sharpe"] - vanilla["sharpe"],
                                  "Cum. return Vanilla %": 100 * vanilla["cumulative_return"],
                                  "Cum. return TS-Booster %": 100 * booster["cumulative_return"],
                                  "NAV lift %": 100 * (booster["nav"] / vanilla["nav"] - 1),
                                  "MDD Vanilla": vanilla["max_drawdown"], "MDD TS-Booster": booster["max_drawdown"],
                                  "p": p_values[seed][dataset]})
    base = pd.DataFrame(base_rows)
    base["p_holm"] = holm_adjusted(base["p"].to_numpy())
    base["significant"] = base["p_holm"] <= ALPHA
    write_table(base.drop(columns="dataset"), out / "table_backtest",
                pct_columns=["Cum. return Vanilla %", "Cum. return TS-Booster %", "NAV lift %"], p_columns=["p", "p_holm"],
                caption=f"Inverse-volatility strategies (σ* = {BASE_TARGET:g}, no costs) driven by Vanilla Linear and "
                        f"TS-Booster+Linear, seed {main_seed}; Ledoit–Wolf one-sided p with Holm correction across datasets.")
    _delta_figure(base, out / "fig11", f"Sharpe-ratio differences, seed {main_seed} (* nominal p < {ALPHA})")

    holm_by_seed = {seed: dict(zip(present, holm_adjusted([p_values[seed][d] for d in present]))) for seed in seeds}
    seed_rows = []
    for dataset in present:
        values = np.array([deltas[seed][dataset] for seed in seeds])
        seed_rows.append({"Dataset": display(dataset), "ΔSR": float(values.mean()), "Std ΔSR": float(values.std()),
                          "Min ΔSR": float(values.min()), "Max ΔSR": float(values.max()),
                          "Seeds positive": f"{int((values > 0).sum())}/{len(seeds)}",
                          "Seeds nominal": f"{sum(p_values[seed][dataset] < ALPHA for seed in seeds)}/{len(seeds)}",
                          "Seeds Holm": f"{sum(holm_by_seed[seed][dataset] <= ALPHA for seed in seeds)}/{len(seeds)}"})
    seed_table = pd.DataFrame(seed_rows)
    write_table(seed_table, out / "table_backtest_seeds",
                caption=f"The same strategies over {len(seeds)} training seeds: mean ΔSR, the seeds on which TS-Booster+Linear "
                        "has the higher Sharpe ratio, and the seeds with nominal and Holm-corrected Ledoit–Wolf significance.")
    _delta_figure(seed_table, out / "fig11_seeds", f"Mean Sharpe-ratio difference over {len(seeds)} seeds (whiskers: min–max)",
                  errors=("Min ΔSR", "Max ΔSR"))

    # Every method: seed-mean Sharpe ratios and TS-Booster against each deterministic method.
    sharpe = pd.DataFrame({label: {dataset: float(np.mean([strategy(label, dataset, seed)["sharpe"]
                                                           for seed in ([0] if label in DETERMINISTIC else seeds)]))
                                   for dataset in present} for label in METHODS})
    cumulative = {label: float(np.mean([strategy(label, d, seed)["cumulative_return"] for d in present for seed in ([0] if label in DETERMINISTIC else seeds)]))
                  for label in METHODS}
    drawdown = {label: float(np.mean([strategy(label, d, seed)["max_drawdown"] for d in present for seed in ([0] if label in DETERMINISTIC else seeds)]))
                for label in METHODS}
    best = sharpe.idxmax(axis=1)
    method_rows, method_tests = [], {}
    for label in METHODS:
        row = {"Method": label, "Mean Sharpe": float(sharpe[label].mean()), "Mean cum. return %": 100 * cumulative[label],
               "Mean MDD": drawdown[label], "Best Sharpe on": f"{int((best == label).sum())}/{len(present)}",
               "TS-Booster better on": "—", "LW nominal": "—", "LW Holm": "—"}
        if label in DETERMINISTIC:
            tests = {d: lw(strategy(TSB, d, main_seed), strategy(label, d, main_seed)) for d in present}
            p = {d: tests[d]["p"] for d in present}
            adjusted = dict(zip(present, holm_adjusted([p[d] for d in present])))
            method_tests[label] = {"p": p, "p_holm": adjusted, "ledoit_wolf": tests}
            row.update({"TS-Booster better on": f"{int((sharpe[TSB] > sharpe[label]).sum())}/{len(present)}",
                        "LW nominal": f"{sum(v < ALPHA for v in p.values())}/{len(present)}",
                        "LW Holm": f"{sum(v <= ALPHA for v in adjusted.values())}/{len(present)}"})
        method_rows.append(row)
    write_table(pd.DataFrame(method_rows), out / "table_backtest_methods", pct_columns=["Mean cum. return %"],
                caption=f"Inverse-volatility strategies driven by each method (σ* = {BASE_TARGET:g}, no costs; neural methods as "
                        f"seed means); Ledoit–Wolf tests of TS-Booster+Linear (seed {main_seed}) against the deterministic methods.")

    # Sensitivity to the volatility target and to transaction costs (main seed).
    grid_rows, sensitivity = [], {}
    for target in TARGETS:
        for cost in COSTS_BPS:
            delta, p = [], []
            for dataset in present:
                vanilla, booster = strategy(VANILLA, dataset, main_seed, target, cost), strategy(TSB, dataset, main_seed, target, cost)
                delta.append(booster["sharpe"] - vanilla["sharpe"])
                p.append(base.set_index("dataset").loc[dataset, "p"] if (target, cost) == (BASE_TARGET, BASE_COST) else lw(booster, vanilla)["p"])
            adjusted = holm_adjusted(p)
            cell = {"improved": int(np.sum(np.array(delta) > 0)), "nominal": int(np.sum(np.array(p) < ALPHA)),
                    "holm_significant": int(np.sum(adjusted <= ALPHA)), "mean_delta": float(np.mean(delta)), "n": len(present)}
            sensitivity[f"{target:g}_{cost:g}"] = cell
            grid_rows.append({"σ* (%)": target, "Cost (bps)": cost, "Improved": f"{cell['improved']}/{cell['n']}",
                              "Nominal p < 0.05": f"{cell['nominal']}/{cell['n']}", "Holm significant": f"{cell['holm_significant']}/{cell['n']}",
                              "Mean ΔSR": cell["mean_delta"]})
    write_table(pd.DataFrame(grid_rows), out / "table_backtest_sensitivity",
                caption=f"TS-Booster+Linear against Vanilla Linear (seed {main_seed}) under three volatility targets and three "
                        "proportional transaction costs.")

    ranked = base.sort_values("ΔSR", ascending=False)
    summary = {"improved": f"{int((base['ΔSR'] > 0).sum())}/{len(base)}", "nominal_significant": int((base["p"] < ALPHA).sum()),
               "holm_significant": int(base["significant"].sum()), "min_adjusted_p": float(base["p_holm"].min()),
               "top3": [(row.dataset, round(float(row.ΔSR), 3)) for row in ranked.head(3).itertuples()],
               "declines": {row.dataset: round(float(row.ΔSR), 3) for row in base[base["ΔSR"] < 0].itertuples()},
               "cumulative_return_pct": {row["dataset"]: (round(row["Cum. return Vanilla %"], 1), round(row["Cum. return TS-Booster %"], 1))
                                         for row in base.to_dict("records")},
               "seeds": {"improved_all_seeds": f"{sum(all(deltas[seed][d] > 0 for seed in seeds) for d in present)}/{len(present)}",
                         "mean_delta_sharpe": float(seed_table["ΔSR"].mean())},
               "methods_mean_sharpe": {label: float(sharpe[label].mean()) for label in METHODS},
               "best_sharpe_method": str(sharpe.mean().idxmax()), "sensitivity": sensitivity}
    write_json({**summary, "paper": PAPER, "table": base.to_dict("records"), "seed_table": seed_table.to_dict("records"),
                "methods": method_rows, "method_tests": method_tests,
                "ledoit_wolf": {str(s): v for s, v in lw_results.items()}, "deltas": {str(s): v for s, v in deltas.items()},
                "sharpe_by_seed": {str(s): v for s, v in sharpe_by_seed.items()}, "sharpe_by_dataset": sharpe.to_dict()},
               out / "backtest.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "backtest")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing, bootstrap=args.bootstrap)
    print(f"improved {summary['improved']}, nominal {summary['nominal_significant']}, Holm {summary['holm_significant']}, "
          f"top3 {summary['top3']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
