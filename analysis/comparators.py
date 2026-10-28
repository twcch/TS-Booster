"""Tables 5, 6 and 7, the CD diagrams of Figs. 3–5, the Eq. 19 reference and the ARIMA orders (spec §5.2).

    python analysis/comparators.py [--results results] [--output analysis/output/comparators]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (ALIGNED, DAGGER, DATASETS, GROUPS, PAPER_DEEP, SEEDS, SEEDS_3, STATISTICAL,  # noqa: E402
                             TREES, TSLIB_DEEP, choose_learning_rate, classical, deep, display, load_runs,
                             ordered, require, tsb, write_json, write_table)
from analysis.stats import compare, dm_compare, plot_cd, rank_test, reductions  # noqa: E402

TSB = "TS-Booster+Linear"
DEEP_LABEL = {"DLinear-paper": "DLinear", "PatchTST-paper": "PatchTST", "iTransformer-paper": "iTransformer"}
METRIC_LABEL = {"rmse": "RMSE", "mae": "MAE", "mda": "MDA"}
# article_v2: Tables 5-7, §4.2.2-4.3 and Figs. 3-5.
PAPER = {
    "table5_means": {TSB: (3.7622, 2.4639, 0.7080), "Persistence": (4.7332, 2.8479, 0.0071), "EWMA": (5.9501, 4.1501, 0.5877),
                     "HAR-RV": (4.4039, 2.7961, 0.5871), "GARCH": (7.5801, 5.9223, 0.5724), "GJR-GARCH": (7.6285, 5.9895, 0.5733),
                     "ARIMA": (4.5254, 2.8047, 0.4727), "XGBoost": (4.9023, 2.7135, 0.6744), "LightGBM": (5.0253, 2.7092, 0.6735)},
    "table6_means": {"DLinear": (4.2594, 2.7015, 0.6260), "PatchTST": (4.4070, 2.8556, 0.6054),
                     "iTransformer": (4.4466, 2.7730, 0.6447)},
    "table7": {"Vanilla Linear": ("16/16", 15.07), "Persistence": ("15/16", 20.92), "EWMA": ("16/16", 37.06),
               "HAR-RV": ("15/16", 13.97), "GARCH": ("16/16", 44.18), "GJR-GARCH": ("16/16", 44.16),
               "ARIMA": ("15/16", 17.15), "XGBoost": ("15/16", 18.54), "LightGBM": ("16/16", 17.27),
               "DLinear": ("15/16", 10.75), "PatchTST": ("15/16", 13.78), "iTransformer": ("14/16", 14.52)},
    "vs_har_rv_reduction_of_mean_pct": 14.6, "subset_reduction_vs_har_rv_pct": {"FX": 14.7, "Equities": 14.5},
    "vs_deep_reduction_of_mean_pct": {"DLinear": 11.7, "PatchTST": 14.6, "iTransformer": 15.4},
    "rank_tests": {"statistical": {"friedman_chi2": 36.25, "cd": 1.17, "average_ranks": {TSB: 1.19, "HAR-RV": 2.19, "ARIMA": 2.78, "Persistence": 3.84}},
                   "ml": {"friedman_chi2": 21.88, "cd": 0.83, "average_ranks": {TSB: 1.06, "LightGBM": 2.31, "XGBoost": 2.62}},
                   "deep": {"friedman_chi2": 27.52, "cd": 1.17, "average_ranks": {TSB: 1.25, "DLinear": 2.31, "iTransformer": 2.88, "PatchTST": 3.56}}},
    "reference_rule": {"mda": 0.683, "rmse": 4.1352},
}


def _means(rows, datasets):
    """Seed means of RMSE, MAE and MDA per dataset, in the paper's dataset order."""
    means = rows.groupby("dataset")[["rmse", "mae", "mda"]].mean()
    return means.reindex(ordered(means.index, datasets))


def _deep_rows(frame, model, chosen, datasets, seeds, allow_missing, **overrides):
    """The runs of a deep comparator at each dataset's chosen learning rate."""
    parts = [require(frame, deep(model, float(row.chosen), **overrides), [row.dataset], seeds,
                     f"{model} {row.dataset} lr {row.chosen:g}", allow_missing)
             for row in chosen.itertuples() if row.dataset in datasets]
    return pd.concat(parts) if parts else frame.iloc[0:0]


def _methods_table(means, order, stem, caption, note=""):
    """Dataset rows plus the three average rows, best value per row in bold (MDA: highest)."""
    datasets = means[order[0]].index
    rows = [{"Dataset": display(dataset), **{display(method): means[method].loc[dataset, "rmse"] for method in order}}
            for dataset in datasets]
    for metric in ("rmse", "mae", "mda"):
        rows.append({"Dataset": f"Avg. {METRIC_LABEL[metric]}",
                     **{display(method): float(means[method][metric].mean()) for method in order}})
    frame = pd.DataFrame(rows)
    write_table(frame, stem, caption=caption, note=note, bold=[display(method) for method in order],
                higher_rows=[len(rows) - 1])
    return frame


def _significance_row(table, family, label):
    weak = table[~table["significant"]]
    return {"Family": family, "Comparator": label, "Significant": f"{int(table['significant'].sum())}/{len(table)}",
            "Mean reduction %": float(table["reduction_pct"].mean()),
            "No significant advantage": "; ".join(f"{display(row.dataset)} ({row.reduction_pct:+.1f}%)"
                                                  for row in weak.itertuples()) or "—"}


def _cd(means, order, stem, title):
    matrix = pd.DataFrame({display(method): means[method]["rmse"] for method in order})
    result = rank_test(matrix)
    for suffix in (".png", ".pdf"):
        plot_cd(result["average_ranks"], result["cd"], stem.with_suffix(suffix),
                f"{title} (N={result['n']}, CD={result['cd']:.2f})")
    return result


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    search_seeds = [seed for seed in SEEDS_3 if seed in seeds] or list(seeds)
    main_seed = 42 if 42 in seeds else seeds[0]

    rows = {TSB: require(frame, tsb(), datasets, seeds, TSB, allow_missing),
            "Vanilla Linear": require(frame, tsb("TSBooster-vanilla"), datasets, seeds, "Vanilla Linear", allow_missing)}
    for model in STATISTICAL + ALIGNED + ["ObservedWindowRule"]:
        rows[model] = require(frame, classical(model), datasets, [0], model, allow_missing)
    for model in TREES:
        rows[model] = require(frame, classical(model), datasets, seeds, model, allow_missing)
    chosen = {model: choose_learning_rate(frame, deep(model), datasets, search_seeds) for model in PAPER_DEEP + TSLIB_DEEP}
    for model in PAPER_DEEP:
        rows[DEEP_LABEL[model]] = _deep_rows(frame, model, chosen[model], datasets, seeds, allow_missing)
        rows[f"{DEEP_LABEL[model]} (matched)"] = require(frame, deep(model, 1e-3, loss="HYBRID", train_mse_weight=0.1),
                                                         datasets, seeds, f"{model} matched", allow_missing)
    for model in TSLIB_DEEP:
        rows[f"{model} (TSLib)"] = _deep_rows(frame, model, chosen[model], datasets, seeds, allow_missing)
    means = {label: _means(table, datasets) for label, table in rows.items()}

    # Tables 5 and 6 (and the TSLib and matched-objective variants of Table 6).
    table5 = _methods_table(means, [TSB] + STATISTICAL + TREES + ALIGNED, out / "table5",
                            "Dataset-level RMSE and cross-dataset mean metrics for the statistical, machine-learning and "
                            "target-aligned comparators; dagger-marked models forecast single-day conditional volatility.")
    for metric in ("mae", "mda"):
        aligned = pd.DataFrame({display(m): means[m][metric] for m in [TSB] + STATISTICAL + TREES + ALIGNED})
        aligned = aligned.reindex(means[TSB].index)  # align on the dataset index, never by position
        per_dataset = aligned.reset_index().rename(columns={"dataset": "Dataset"})
        per_dataset["Dataset"] = per_dataset["Dataset"].map(display)
        write_table(per_dataset, out / f"table5_{metric}", bold=[display(m) for m in [TSB] + STATISTICAL + TREES + ALIGNED],
                    lower_is_better=metric != "mda", caption=f"Dataset-level {METRIC_LABEL[metric]} of the Table 5 methods.")
    deep_labels = [DEEP_LABEL[model] for model in PAPER_DEEP]
    table6 = _methods_table(means, [TSB] + deep_labels, out / "table6",
                            "Deep comparators (the paper's reimplementations) at the learning rate chosen per dataset by "
                            "validation; MSE + Adam versus TS-Booster's hybrid objective.")
    _methods_table(means, [TSB] + [f"{m} (TSLib)" for m in TSLIB_DEEP], out / "table6_tslib",
                   "TSLib's DLinear, PatchTST and iTransformer at the learning rate chosen per dataset by validation.")
    _methods_table(means, [TSB] + [f"{label} (matched)" for label in deep_labels], out / "table6_matched",
                   "The paper's deep comparators trained with TS-Booster's objective and protocol (R2.5).")
    selected = pd.concat([table.assign(model=model) for model, table in chosen.items()], ignore_index=True)
    write_table(selected[["model", *[c for c in selected.columns if c != "model"]]], out / "table6_selected_lr",
                caption="Learning rate chosen per dataset: lowest mean validation score over the search seeds.", decimals=6)

    # Table 7 and the DM tests: TS-Booster+Linear against every comparator.
    comparisons = [("Backbone", "Vanilla Linear", tsb("TSBooster-vanilla"), seeds, frame)]
    comparisons += [("Statistical", m, classical(m), [0], frame) for m in STATISTICAL]
    comparisons += [("Target-aligned", m, classical(m), [0], frame) for m in ALIGNED]
    comparisons += [("Machine learning", m, classical(m), seeds, frame) for m in TREES]
    tsb_rows = rows[TSB]
    for model in PAPER_DEEP:
        comparisons.append(("Deep learning", DEEP_LABEL[model], deep(model), seeds, pd.concat([tsb_rows, rows[DEEP_LABEL[model]]])))
        comparisons.append(("Deep learning (matched objective)", f"{DEEP_LABEL[model]} (matched)",
                            deep(model, 1e-3, loss="HYBRID", train_mse_weight=0.1), seeds, frame))
    for model in TSLIB_DEEP:
        comparisons.append(("Deep learning (TSLib)", f"{model} (TSLib)", deep(model), seeds, pd.concat([tsb_rows, rows[f"{model} (TSLib)"]])))
    table7_rows, dm_rows, tests = [], [], {}
    for family, label, filters, base_seeds, source in comparisons:
        table = compare(source, filters, tsb(), datasets, seeds, base_seeds=base_seeds, allow_missing=allow_missing,
                        labels=(label, TSB))
        tests[label] = table
        table7_rows.append(_significance_row(table, family, display(label)))
        dm = dm_compare(source, filters, tsb(), datasets, seeds, base_seeds=base_seeds, allow_missing=allow_missing)
        counts = dm.groupby("dataset")["significant"].agg(["sum", "size"])
        for row in dm[dm["seed"] == main_seed].itertuples():
            dm_rows.append({"Comparator": display(label), "Dataset": display(row.dataset), "DM": row.dm, "p": row.p,
                            "p_holm": row.p_holm, "significant": bool(row.significant),
                            "seeds_significant": f"{int(counts.loc[row.dataset, 'sum'])}/{int(counts.loc[row.dataset, 'size'])}"})
    table7 = pd.DataFrame(table7_rows)
    write_table(table7, out / "table7", pct_columns=["Mean reduction %"],
                caption="Significance of the TS-Booster+Linear RMSE advantage: one-sided seed-based tests with Holm "
                        "correction across datasets per comparator; deterministic comparators use one-sample tests.")
    write_table(pd.DataFrame(dm_rows), out / "table_dm", p_columns=["p", "p_holm"],
                caption=f"Diebold–Mariano (HLN) tests on per-window MSE, seed {main_seed}; seeds_significant counts the "
                        "candidate seeds with a Holm-significant DM test.")

    # Figs. 3-5: rank tests.
    rank_tests = {"statistical": _cd(means, [TSB, "Persistence", "HAR-RV", "ARIMA"] + ALIGNED, out / "fig3_cd_statistical",
                                     "RMSE ranks: common-target statistical comparators"),
                  "ml": _cd(means, [TSB] + TREES, out / "fig4_cd_ml", "RMSE ranks: machine-learning comparators"),
                  "deep": _cd(means, [TSB] + deep_labels, out / "fig5_cd_deep", "RMSE ranks: deep comparators")}
    ranks_md = ["| Group | Friedman χ² | p | CD | Average ranks |", "|---|---|---|---|---|"]
    for group, result in rank_tests.items():
        ranks = ", ".join(f"{m} {r:.2f}" for m, r in sorted(result["average_ranks"].items(), key=lambda item: item[1]))
        ranks_md.append(f"| {group} | {result['friedman_chi2']:.2f} | {result['friedman_p']:.2e} | {result['cd']:.2f} | {ranks} |")
    (out / "rank_tests.md").write_text("\n".join(ranks_md) + "\n", encoding="utf-8")

    # ARIMA orders and the observed-window reference.
    arima = []
    for row in rows["ARIMA"].itertuples():
        selected_ = (row.baseline or {}).get("selected", {})
        order = tuple(selected_.get("order", ()))
        arima.append({"Dataset": display(row.dataset), "Order (p,d,q)": str(order),
                      "Intercept": bool(selected_.get("with_intercept", False)), "AIC": selected_.get("aic"),
                      "Random walk": order == (0, 1, 0) and not selected_.get("with_intercept", False)})
    write_table(pd.DataFrame(arima), out / "arima_orders", caption="ARIMA orders selected by auto_arima on training and validation.")
    reference = means["ObservedWindowRule"].mean()
    write_table(pd.DataFrame([{"Reference": "Observed-window rule (Eq. 19)", "Mean RMSE": reference["rmse"],
                               "Mean MAE": reference["mae"], "Mean MDA": reference["mda"]}]), out / "reference_rule",
                caption="The observed-window reference rule: known returns of the target window only.")

    # Summary.
    har = means["HAR-RV"]["rmse"]
    tsb_means = means[TSB]["rmse"]
    fx = [d for group in ("TWD FX", "International FX") for d in GROUPS[group] if d in tsb_means.index]
    equities = [d for d in GROUPS["Equities"] if d in tsb_means.index]
    common_target = [TSB, "Persistence", "HAR-RV", "ARIMA"] + TREES + ALIGNED
    rmse_matrix = pd.DataFrame({m: means[m]["rmse"] for m in common_target})
    best = rmse_matrix.idxmin(axis=1)
    summary = {
        "vs_har_rv": reductions(har, tsb_means),
        "subset_reduction_vs_har_rv_pct": {name: reductions(har.loc[members], tsb_means.loc[members])["reduction_of_mean_pct"]
                                           for name, members in (("FX", fx), ("Equities", equities)) if members},
        "vs_deep_reduction_of_mean_pct": {label: reductions(means[label]["rmse"], tsb_means)["reduction_of_mean_pct"]
                                          for label in deep_labels},
        "vs_tslib_reduction_of_mean_pct": {m: reductions(means[f"{m} (TSLib)"]["rmse"], tsb_means)["reduction_of_mean_pct"]
                                           for m in TSLIB_DEEP},
        "best_rmse_count": {m: int((best == m).sum()) for m in common_target},
        "rank_tests": rank_tests,
        "aligned_best_datasets": [d for d in tsb_means.index if rmse_matrix.loc[d, ALIGNED].min() < tsb_means.loc[d]],
        "selected_learning_rates": {model: dict(zip(table["dataset"], table["chosen"].astype(float))) for model, table in chosen.items()},
        "table5_means": {m: tuple(float(means[m][k].mean()) for k in ("rmse", "mae", "mda")) for m in [TSB] + STATISTICAL + TREES + ALIGNED},
        "table6_means": {label: tuple(float(means[label][k].mean()) for k in ("rmse", "mae", "mda")) for label in deep_labels},
        "table7": {row["Comparator"]: (row["Significant"], row["Mean reduction %"]) for row in table7_rows},
        "reference_rule": {k: float(reference[k]) for k in ("rmse", "mae", "mda")},
        "arima_random_walk": [row["Dataset"] for row in arima if row["Random walk"]],
    }
    write_json({**summary, "paper": PAPER, "tests": {label: table.to_dict("records") for label, table in tests.items()},
                "dm": dm_rows, "table5": table5.to_dict("records"), "table6": table6.to_dict("records")},
               out / "comparators.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "comparators")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print(f"vs HAR-RV: {summary['vs_har_rv']['reduction_of_mean_pct']:+.1f}% of the mean RMSE; "
          f"best RMSE counts {summary['best_rmse_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
