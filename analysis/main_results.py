"""Table 3, Table 4 and Fig. 2 (spec §5.1): the four backbones alone and with TS-Booster.

    python analysis/main_results.py [--results results] [--output analysis/output/main_results]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (BACKBONE_LABEL, BACKBONES, DATASETS, GROUPS, PALETTE, SEEDS, display, load_runs,  # noqa: E402
                             tsb, write_figure, write_json, write_table)
from analysis.stats import compare, overall, reductions  # noqa: E402

METRICS = ("rmse", "mae", "mda")
METRIC_LABEL = {"rmse": "RMSE", "mae": "MAE", "mda": "MDA"}
# article_v2: Table 3, Table 4 and the numbers quoted in §4.2.1.
PAPER = {
    "table3": {"linear": (4.5699, 3.7622, 15.1, "16/16"), "lstm": (4.6331, 4.1797, 7.6, "6/16"),
               "gru": (4.6730, 4.1510, 8.3, "9/16"), "transformer": (5.0603, 4.2077, 16.9, "15/16")},
    "mae_improvement_pct": {"linear": 15.5, "lstm": 10.2, "gru": 10.7, "transformer": 19.0},
    "mda_improvement_pct": {"linear": 14.1, "lstm": 2.8, "gru": 2.4, "transformer": 16.7},
    "significant_combinations": 46, "positive_combinations": 64, "mean_reduction_all_backbones_pct": 12.0,
    "linear_group_reductions_pct": {"TWD FX": 12.3, "International FX": 13.6, "Equities": 19.0},
    "table4": {"USD_TWD": (1.8102, 1.5282), "EUR_TWD": (2.0276, 1.8049), "JPY_TWD": (2.4169, 2.0504),
               "CNY_TWD": (1.8653, 1.5986), "THB_TWD": (1.9364, 1.6950), "KRW_TWD": (1.8220, 1.6839),
               "CAD_TWD": (1.9763, 1.7736), "EUR_USD": (1.5952, 1.3999), "GBP_USD": (1.9204, 1.6570),
               "USD_JPY": (2.2813, 1.9459), "0050_TW": (10.3195, 7.5071), "2317_TW": (7.8338, 6.6660),
               "2330_TW": (7.1627, 5.8629), "2454_TW": (8.8069, 7.5864), "2603_TW": (12.8879, 10.2181),
               "2881_TW": (6.4564, 5.2176)},
    "table4_mean_reduction_pct": 15.07,
}
# The exchange-rate series whose data equal the paper's: CNY/TWD and THB/TWD were repaired (tools/data/repair_fx.py).
UNCHANGED_FX = [dataset for dataset in DATASETS[:10] if dataset not in ("CNY_TWD", "THB_TWD")]


def _heatmap(matrix, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(0.75 * matrix.shape[1] + 2, 0.55 * matrix.shape[0] + 1.6))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), [display(name) for name in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]), [BACKBONE_LABEL[name] for name in matrix.index])
    limit = np.nanmax(matrix.to_numpy(dtype=float))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.6 * limit else "black")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="RMSE improvement over vanilla (%)")
    ax.set_title("(a) RMSE improvement over the vanilla backbone (%), seed mean")
    write_figure(fig, out / "fig2_heatmap")


def _bars(improvement, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x, width = np.arange(len(BACKBONES)), 0.26
    for k, metric in enumerate(METRICS):
        values = [improvement[metric][backbone] for backbone in BACKBONES]
        bars = ax.bar(x + (k - 1) * width, values, width, label=METRIC_LABEL[metric], color=PALETTE[k])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(x, [BACKBONE_LABEL[backbone] for backbone in BACKBONES])
    ax.set_ylabel("Average improvement (%)")
    ax.set_title("(b) Mean relative improvement across datasets by metric")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, out / "fig2_bars")


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = {(backbone, metric): compare(frame, tsb("TSBooster-vanilla", backbone), tsb("TSBooster", backbone), datasets, seeds,
                                         metric=metric, allow_missing=allow_missing,
                                         labels=(f"Vanilla {BACKBONE_LABEL[backbone]}", f"TS-Booster+{BACKBONE_LABEL[backbone]}"))
              for backbone in BACKBONES for metric in METRICS}

    rows, improvement = [], {metric: {} for metric in METRICS}
    for backbone in BACKBONES:
        table = tables[(backbone, "rmse")].set_index("dataset")
        summary = reductions(table["baseline_mean"], table["candidate_mean"])
        for metric in METRICS:
            improvement[metric][backbone] = float(tables[(backbone, metric)]["reduction_pct"].mean())
        rows.append({"Backbone": BACKBONE_LABEL[backbone], "Vanilla": summary["base_mean"],
                     "+TS-Booster": summary["cand_mean"], "Reduction %": summary["mean_of_reductions_pct"],
                     "Reduction of mean %": summary["reduction_of_mean_pct"],
                     "Significant": f"{int(table['significant'].sum())}/{len(table)}",
                     "MAE improvement %": improvement["mae"][backbone], "MDA improvement %": improvement["mda"][backbone]})
    write_table(pd.DataFrame(rows), out / "table3",
                caption="Cross-dataset mean RMSE by backbone (seed means averaged over datasets); Reduction is the "
                        "mean of the dataset-specific percentage reductions; Significant counts Holm-corrected "
                        "one-sided paired t tests within the backbone.",
                pct_columns=["Reduction %", "Reduction of mean %", "MAE improvement %", "MDA improvement %"])

    linear = tables[("linear", "rmse")]
    rows = [{"Dataset": display(row.dataset),
             "Vanilla Linear": f"{row.baseline_mean:.4f} ± {row.baseline_std:.4f}",
             "TS-Booster+Linear": f"{row.candidate_mean:.4f} ± {row.candidate_std:.4f}",
             "Reduction %": row.reduction_pct, "Adjusted p": row.p_holm} for row in linear.itertuples()]
    rows.append({"Dataset": f"Mean ({int(linear['significant'].sum())}/{len(linear)} significant)",
                 "Vanilla Linear": "", "TS-Booster+Linear": "", "Reduction %": float(linear["reduction_pct"].mean()),
                 "Adjusted p": float("nan")})
    write_table(pd.DataFrame(rows), out / "table4", bold=["Vanilla Linear", "TS-Booster+Linear"],
                pct_columns=["Reduction %"], p_columns=["Adjusted p"],
                caption="RMSE of Vanilla Linear and TS-Booster+Linear by dataset (mean ± std over seeds, divisor N); "
                        "one-sided paired t tests with Holm correction across datasets.")

    long = pd.concat([tables[(backbone, metric)].assign(backbone=backbone, metric=metric)
                      for backbone in BACKBONES for metric in METRICS], ignore_index=True)
    long = long[["backbone", "metric", "dataset", "baseline_mean", "candidate_mean", "reduction_pct", "p", "p_holm",
                 "significant"]].rename(columns={"reduction_pct": "improvement_pct"})
    write_table(long, out / "cross_backbone_reductions", pct_columns=["improvement_pct"], p_columns=["p", "p_holm"],
                caption="Relative improvement of TS-Booster over each vanilla backbone, per dataset and metric.")

    matrix = pd.DataFrame({backbone: tables[(backbone, "rmse")].set_index("dataset")["reduction_pct"]
                           for backbone in BACKBONES}).T.reindex(columns=[d for d in datasets if d in linear["dataset"].values])
    _heatmap(matrix, out)
    _bars(improvement, out)

    rmse_tables = [tables[(backbone, "rmse")] for backbone in BACKBONES]
    linear_indexed = linear.set_index("dataset")
    summary = {
        "positive_combinations": int(sum((table["reduction_pct"] > 0).sum() for table in rmse_tables)),
        "significant_combinations": int(sum(table["significant"].sum() for table in rmse_tables)),
        "combinations": int(sum(len(table) for table in rmse_tables)),
        "mean_reduction_all_backbones_pct": float(np.mean([improvement["rmse"][b] for b in BACKBONES])),
        "reduction_by_backbone_pct": improvement["rmse"],
        "improvement_by_backbone": improvement,
        "table3": {backbone: reductions(tables[(backbone, "rmse")].set_index("dataset")["baseline_mean"],
                                        tables[(backbone, "rmse")].set_index("dataset")["candidate_mean"])
                   for backbone in BACKBONES},
        "linear_group_reductions_pct": {group: float(linear_indexed.loc[[d for d in members if d in linear_indexed.index],
                                                                        "reduction_pct"].mean())
                                        for group, members in GROUPS.items()
                                        if any(d in linear_indexed.index for d in members)},
        "table4_mean_reduction_pct": float(linear["reduction_pct"].mean()),
        "table4_significant": f"{int(linear['significant'].sum())}/{len(linear)}",
        "overall_tests_by_backbone": {backbone: overall(tables[(backbone, "rmse")]["reduction_pct"])
                                      for backbone in BACKBONES},
        "table4_vs_paper_pct": {dataset: (float((linear_indexed.loc[dataset, "baseline_mean"] / PAPER["table4"][dataset][0] - 1) * 100),
                                          float((linear_indexed.loc[dataset, "candidate_mean"] / PAPER["table4"][dataset][1] - 1) * 100))
                                for dataset in UNCHANGED_FX if dataset in linear_indexed.index},
    }
    write_json({**summary, "paper": PAPER,
                "comparisons": {f"{backbone}_{metric}": table.to_dict("records") for (backbone, metric), table in tables.items()}},
               out / "main_results.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "main_results")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print(f"{summary['significant_combinations']}/{summary['combinations']} combinations significant; "
          f"mean reduction {summary['mean_reduction_all_backbones_pct']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
