"""Fig. 10, the §4.7 numbers and the gate variance decomposition for R1.2/R2.2 (spec §5.6).

    python analysis/gate.py [--results results] [--output analysis/output/gate]
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (DATASETS, DEFAULT, PALETTE, SEEDS, display, load_runs, ordered, require, tsb,  # noqa: E402
                             write_figure, write_json, write_table)
from analysis.stats import compare  # noqa: E402

SHORT, LONG = DEFAULT["decomp_kernels"]
# article_v2 §4.7 and Fig. 10 (seed 42).
PAPER = {"snr_w_range": (0.26, 0.74), "snr_b_range": (-0.75, -0.25), "gate_mean_range": (0.37, 0.60),
         "positive_slopes": "16/16", "within_std_range": (0.004, 0.010), "short_kernel_weight_range": (0.52, 0.66),
         "short_kernel_preferred": "12/16",
         "decomp_short_weight_seed42": {"EUR_TWD": 0.47, "GBP_USD": 0.49, "USD_JPY": 0.50, "2454_TW": 0.50, "2881_TW": 0.52,
                                        "KRW_TWD": 0.53, "CAD_TWD": 0.55, "JPY_TWD": 0.55, "2330_TW": 0.55, "0050_TW": 0.56,
                                        "EUR_USD": 0.56, "2603_TW": 0.57, "THB_TWD": 0.62, "USD_TWD": 0.62, "CNY_TWD": 0.64,
                                        "2317_TW": 0.66},
         "gate_mean_seed42": {"USD_TWD": 0.37, "EUR_USD": 0.38, "2330_TW": 0.39, "USD_JPY": 0.39, "EUR_TWD": 0.41,
                              "CAD_TWD": 0.41, "THB_TWD": 0.42, "2881_TW": 0.42, "2603_TW": 0.44, "JPY_TWD": 0.44,
                              "2454_TW": 0.45, "GBP_USD": 0.45, "CNY_TWD": 0.46, "2317_TW": 0.47, "KRW_TWD": 0.49,
                              "0050_TW": 0.60}}


def _horizontal_bars(values, errors, xlabel, title, stem, stacked_complement=None):
    """Horizontal bars sorted ascending; `errors` adds error bars, `stacked_complement` a second segment to 1."""
    import matplotlib.pyplot as plt
    order = values.sort_values().index
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(order) + 1.2))
    y = np.arange(len(order))
    ax.barh(y, values.loc[order], color=PALETTE[0], xerr=None if errors is None else errors.loc[order], capsize=2,
            label=f"k = {SHORT}" if stacked_complement is not None else None)
    if stacked_complement is not None:
        ax.barh(y, stacked_complement.loc[order], left=values.loc[order], color=PALETTE[1], label=f"k = {LONG}")
        ax.legend(frameon=False, loc="lower right")
    reach = values if errors is None else values + errors  # labels sit beyond the error bars
    for position, name in enumerate(order):
        x = reach.loc[name] + 0.01 if stacked_complement is None else values.loc[name] - 0.02
        ax.text(x, position, f"{values.loc[name]:.2f}",
                va="center", ha="left" if stacked_complement is None else "right", fontsize=8,
                color="black" if stacked_complement is None else "white")
    ax.set_yticks(y, [display(name) for name in order])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 1 if stacked_complement is not None else max(0.7, float(reach.max()) * 1.2))
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, stem)


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_seed = 42 if 42 in seeds else seeds[0]
    rows = require(frame, tsb(), datasets, seeds, "TS-Booster+Linear", allow_missing)

    # Per-fit parameters and gate statistics from gate.npy.
    records, gates = [], {}
    for row in rows.itertuples():
        path = Path(row.path) / "gate.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{Path(row.path).name}: gate.npy is missing")
        values = np.load(path).astype(float)
        gates[(row.dataset, int(row.seed))] = values
        weights = row.decomp_w or {}
        records.append({"Dataset": display(row.dataset), "dataset": row.dataset, "Seed": int(row.seed), "w_g": row.snr_w,
                        "b_g": row.snr_b, "Gate mean": float(values.mean()), "Gate std": float(values.std()),
                        "Gate min": float(values.min()), "Gate max": float(values.max()),
                        f"Weight k={SHORT}": weights.get(SHORT), "Slope bound |w_g|/4": abs(row.snr_w) / 4 if row.snr_w is not None else math.nan})
    params = pd.DataFrame(records)
    write_table(params.drop(columns="dataset"), out / "table_gate_parameters",
                caption="Fitted gate slope and bias, test-set gate activation statistics (from gate.npy) and the shorter "
                        "decomposition kernel's weight, for every TS-Booster+Linear fit.")

    # Fig. 10: the paper's single fit (seed 42) and the seed mean ± std.
    main = params[params["Seed"] == main_seed].set_index("dataset")
    short = main[f"Weight k={SHORT}"].astype(float)
    _horizontal_bars(short, None, "Learned decomposition weight", f"(a) Multi-scale decomposition weights (seed {main_seed})",
                     out / "fig10_decomp_weights", stacked_complement=1 - short)
    _horizontal_bars(main["Gate mean"], None, "Mean gate activation g (test set)", f"(b) Reliability-gate activation (seed {main_seed})",
                     out / "fig10_gate")
    by_dataset = params.groupby("dataset")
    _horizontal_bars(by_dataset[f"Weight k={SHORT}"].mean(), by_dataset[f"Weight k={SHORT}"].std(ddof=0),
                     "Learned decomposition weight", f"(a) Decomposition weights, mean ± std over {len(seeds)} seeds",
                     out / "fig10_decomp_weights_seeds")
    _horizontal_bars(by_dataset["Gate mean"].mean(), by_dataset["Gate mean"].std(ddof=0), "Mean gate activation g (test set)",
                     f"(b) Gate activation, mean ± std over {len(seeds)} seeds", out / "fig10_gate_seeds")

    # Three-way variance decomposition of the pooled gate values (law of total variance, weighted by
    # sample counts): over time within one fitted run, between seeds of a dataset, between datasets.
    pooled = {dataset: np.concatenate([gates[(dataset, seed)] for seed in seeds if (dataset, seed) in gates])
              for dataset in datasets if any((dataset, seed) in gates for seed in seeds)}
    runs = [(dataset, values) for (dataset, _), values in gates.items()]
    counts = np.array([len(values) for _, values in runs], dtype=float)
    run_means = np.array([values.mean() for _, values in runs])
    run_variances = np.array([values.var() for _, values in runs])
    dataset_means_ = {dataset: float(values.mean()) for dataset, values in pooled.items()}
    grand = float(np.average(run_means, weights=counts))
    within_run = float(np.average(run_variances, weights=counts))
    between_seeds = float(np.average((run_means - np.array([dataset_means_[d] for d, _ in runs])) ** 2, weights=counts))
    between_datasets = float(np.average((np.array([dataset_means_[d] for d, _ in runs]) - grand) ** 2, weights=counts))
    total = within_run + between_seeds + between_datasets
    shares = {key: (value / total if total > 0 else math.nan)
              for key, value in (("within_run", within_run), ("between_seeds", between_seeds), ("between_datasets", between_datasets))}
    per_fit_std = params["Gate std"].astype(float)
    pooled_std = np.array([values.std() for values in pooled.values()])
    variance_rows = [
        {"Quantity": "Pooled gate values", "Value": f"{int(counts.sum())} samples, {len(pooled)} datasets, {len(seeds)} seeds"},
        {"Quantity": "Total variance", "Value": f"{total:.6f}"},
        {"Quantity": "Share within one fitted run (over the test period)", "Value": f"{100 * shares['within_run']:.1f}%"},
        {"Quantity": "Share between seeds of the same dataset", "Value": f"{100 * shares['between_seeds']:.1f}%"},
        {"Quantity": "Share between datasets", "Value": f"{100 * shares['between_datasets']:.1f}%"},
        {"Quantity": "Gate std within a fit (min–max over fits)", "Value": f"{per_fit_std.min():.4f}–{per_fit_std.max():.4f}"},
        {"Quantity": "Gate std within a dataset, seeds pooled (min–max)", "Value": f"{pooled_std.min():.4f}–{pooled_std.max():.4f}"},
        {"Quantity": "Gate range within a fit, max − min (median)", "Value": f"{(params['Gate max'] - params['Gate min']).median():.4f}"},
        {"Quantity": "Slope bound |w_g|/4 (min–max)", "Value": f"{params['Slope bound |w_g|/4'].min():.4f}–{params['Slope bound |w_g|/4'].max():.4f}"},
    ]
    write_table(pd.DataFrame(variance_rows), out / "table_gate_variance",
                caption="Where the gate varies: the pooled gate values' variance split into variation over time within "
                        "one fitted run, between training seeds of a dataset, and between datasets (law of total variance), "
                        "against the sigmoid's slope bound.")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.55 * len(pooled) + 2, 3.4))
    ax.boxplot([pooled[d] for d in pooled], tick_labels=[display(d) for d in pooled], showfliers=False)
    ax.set_ylabel("Gate activation g")
    ax.set_title(f"Gate activation per dataset ({len(seeds)} seeds pooled)")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, out / "fig_gate_distribution")

    # Dataset mean gate against the Linear RMSE reduction (descriptive).
    gain = compare(frame, tsb("TSBooster-vanilla"), tsb(), datasets, seeds, allow_missing=allow_missing,
                   labels=("Vanilla Linear", "TS-Booster+Linear")).set_index("dataset")["reduction_pct"]
    joined = pd.DataFrame({"Mean gate": by_dataset["Gate mean"].mean(), "RMSE reduction %": gain}).dropna()
    joined = joined.reindex(ordered(joined.index, datasets))
    if len(joined) >= 3:
        rho, p_value = sp.spearmanr(joined["Mean gate"], joined["RMSE reduction %"])
        spearman = {"rho": float(rho), "p": float(p_value), "n": len(joined)}
    else:
        spearman = {"rho": math.nan, "p": math.nan, "n": len(joined)}
    gain_table = joined.reset_index().rename(columns={"dataset": "Dataset"})
    gain_table["Dataset"] = gain_table["Dataset"].map(display)
    write_table(gain_table, out / "table_gate_vs_gain", pct_columns=["RMSE reduction %"],
                caption=f"Dataset mean gate activation against the RMSE reduction of TS-Booster+Linear; Spearman rho = "
                        f"{spearman['rho']:.3f} (p = {spearman['p']:.3g}, n = {spearman['n']}).")

    slopes = params["w_g"].astype(float)
    summary = {"snr_w_range": (float(slopes.min()), float(slopes.max())),
               "snr_b_range": (float(params["b_g"].min()), float(params["b_g"].max())),
               "gate_mean_range": (float(main["Gate mean"].min()), float(main["Gate mean"].max())),
               "gate_mean_range_all_fits": (float(params["Gate mean"].min()), float(params["Gate mean"].max())),
               "positive_slopes": f"{int((slopes > 0).sum())}/{len(slopes)}",
               "within_std_range": (float(per_fit_std.min()), float(per_fit_std.max())),
               "variance_share": shares, "spearman_gate_vs_gain": spearman,
               "short_kernel_weight_range": (float(short.min()), float(short.max())),
               "short_kernel_preferred": f"{int((short > 0.5).sum())}/{len(short)}"}
    write_json({**summary, "paper": PAPER, "fits": params.drop(columns="dataset").to_dict("records"),
                "dataset_means": {d: {"gate_mean": float(by_dataset["Gate mean"].mean()[d]), "short_weight": float(by_dataset[f"Weight k={SHORT}"].mean()[d])}
                                  for d in by_dataset.groups}}, out / "gate.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "gate")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print(f"w_g {summary['snr_w_range']}, gate mean {summary['gate_mean_range']}, within share "
          f"{100 * summary['variance_share']['within']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
