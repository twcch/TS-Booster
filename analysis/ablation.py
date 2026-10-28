"""Fig. 8, Fig. 9, the architecture × objective factorial (R1.1/R2.1) and the incremental tests (R2.6)
of spec §5.3.

    python analysis/ablation.py [--results results] [--output analysis/output/ablation]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (DATASETS, PALETTE, SEEDS, dataset_means, display, load_runs, tsb, write_figure,  # noqa: E402
                             write_json, write_table)
from analysis.stats import compare, dm_compare, overall, reductions  # noqa: E402

LATER = {"train_epochs": 120}  # the paper's later batch (gate ablation)
ARMS = {"Vanilla": tsb("TSBooster-vanilla"), "+Decomp": tsb("TSBooster-woexpert"),
        "+Expert": tsb("TSBooster-wodecomp"), "Full": tsb()}
OBJECTIVES = ["L_pred only", "L_pred + training terms", "MSE only"]
FACTORIAL = {
    ("Vanilla", OBJECTIVES[0]): tsb("TSBooster-vanilla"),
    ("Vanilla", OBJECTIVES[1]): tsb("TSBooster-vanilla", train_mse_weight=0.1),  # the anchoring term alone
    ("Vanilla", OBJECTIVES[2]): tsb("TSBooster-vanilla", loss="MSE"),
    ("+Expert", OBJECTIVES[0]): tsb("TSBooster-wodecomp", aux=0.0, base=0.0),
    ("+Expert", OBJECTIVES[1]): tsb("TSBooster-wodecomp"),
    ("+Expert", OBJECTIVES[2]): tsb("TSBooster-wodecomp", loss="MSE", aux=0.0, base=0.0),
    ("Full", OBJECTIVES[0]): tsb(aux=0.0, base=0.0),
    ("Full", OBJECTIVES[1]): tsb(),
    ("Full", OBJECTIVES[2]): tsb(loss="MSE", aux=0.0, base=0.0),
}
CONTRASTS = {  # key: (name, baseline, candidate)
    "a": ("Architecture under MSE: Full vs Vanilla", tsb("TSBooster-vanilla", loss="MSE"), tsb(loss="MSE", aux=0.0, base=0.0)),
    "b": ("Auxiliary losses: Full (paper) vs Full with L_pred only", tsb(aux=0.0, base=0.0), tsb()),
    "c": ("Hybrid loss: Vanilla (paper) vs Vanilla with MSE", tsb("TSBooster-vanilla", loss="MSE"), tsb("TSBooster-vanilla")),
    "d": ("Experts under MSE: +Expert vs Vanilla", tsb("TSBooster-vanilla", loss="MSE"), tsb("TSBooster-wodecomp", loss="MSE", aux=0.0, base=0.0)),
}
GATES = {"No gate": tsb("TSBooster-nogate", **LATER), "Constant (0.5)": tsb("TSBooster-constgate", **LATER),
         "Learned scalar": tsb("TSBooster-learnedgate", **LATER), "RAR gate": tsb(**LATER)}
GATE_KEY = {"No gate": "no_gate", "Constant (0.5)": "constant", "Learned scalar": "learned", "RAR gate": "rar"}
INCREMENTAL = [("Full vs +Expert", ARMS["+Expert"], ARMS["Full"]),
               ("RAR gate vs Constant (0.5)", GATES["Constant (0.5)"], GATES["RAR gate"]),
               ("RAR gate vs No gate", GATES["No gate"], GATES["RAR gate"]),
               ("RAR gate vs Learned scalar", GATES["Learned scalar"], GATES["RAR gate"])]
# article_v2 §4.6 and Fig. 8-9.
PAPER = {"fig8_means": {"Vanilla": 4.5699, "+Decomp": 4.5872, "+Expert": 3.8436, "Full": 3.7622},
         "fig8_reductions_pct": {"expert_only_vs_vanilla": 15.9, "full_vs_expert": 2.1, "full_vs_vanilla": 17.7},
         "fig9_means": {"no_gate": 3.8263, "constant": 3.7792, "learned": 3.7849, "rar": 3.7761},
         "fig9_mda": {"no_gate": 0.7041, "constant": 0.7069, "learned": 0.7069, "rar": 0.7078},
         "rar_vs_constant_reduction_of_mean_pct": 0.08, "rar_vs_no_gate_reduction_of_mean_pct": 1.3,
         "rar_best_on_rmse": "10/16", "constant_better_seed_level": "3 of 5"}


def _means(frame, arms, datasets, seeds, allow_missing, metrics=("rmse", "mae", "mda")):
    """{arm: {metric: Series of dataset seed means}}"""
    return {arm: {metric: dataset_means(frame, filters, datasets, seeds, metric, arm, allow_missing) for metric in metrics}
            for arm, filters in arms.items()}


def _bar_figure(labels, values, annotations, ylabel, title, stem, color):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 1.5, 3.4))
    bars = ax.bar(range(len(labels)), values, color=color)
    for bar, value, note in zip(bars, values, annotations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}\n{note}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, stem)


def _pair_summary(frame, name, base, cand, datasets, seeds, allow_missing, main_seed):
    table = compare(frame, base, cand, datasets, seeds, allow_missing=allow_missing, labels=(name, name))
    summary = reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])
    dm = dm_compare(frame, base, cand, datasets, seeds, allow_missing=allow_missing)
    tests = overall(table["reduction_pct"])
    wins = 0  # seeds on which the candidate's cross-dataset mean RMSE is lower
    for seed in seeds:
        b = dataset_means(frame, base, datasets, [seed], "rmse", name, allow_missing).mean()
        c = dataset_means(frame, cand, datasets, [seed], "rmse", name, allow_missing).mean()
        wins += int(c < b)
    return {"Comparison": name, "Baseline mean": summary["base_mean"], "Candidate mean": summary["cand_mean"],
            "Reduction of mean %": summary["reduction_of_mean_pct"], "Mean reduction %": summary["mean_of_reductions_pct"],
            "Significant (paired t)": f"{int(table['significant'].sum())}/{len(table)}",
            f"DM significant (seed {main_seed})": f"{int(dm[dm['seed'] == main_seed]['significant'].sum())}/{len(table)}",
            "DM significant (all seeds)": f"{int(dm['significant'].sum())}/{len(dm)}",
            "Wilcoxon p": tests["wilcoxon_p"], "Sign-test p": tests["sign_p"],
            "Seed-level wins": f"{wins}/{len(seeds)}"}, table, dm


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    main_seed = 42 if 42 in seeds else seeds[0]

    # Fig. 8: component ablation in the main batch.
    means = _means(frame, ARMS, datasets, seeds, allow_missing)
    vanilla = means["Vanilla"]["rmse"]
    rows, annotations = [], []
    for arm in ARMS:
        summary = reductions(vanilla, means[arm]["rmse"])
        test = compare(frame, ARMS["Vanilla"], ARMS[arm], datasets, seeds, allow_missing=allow_missing, labels=("Vanilla", arm)) \
            if arm != "Vanilla" else None
        rows.append({"Variant": arm, "Mean RMSE": summary["cand_mean"], "Mean MAE": float(means[arm]["mae"].mean()),
                     "Mean MDA": float(means[arm]["mda"].mean()), "Reduction of mean %": summary["reduction_of_mean_pct"],
                     "Mean reduction %": summary["mean_of_reductions_pct"],
                     "Significant vs Vanilla": f"{int(test['significant'].sum())}/{len(test)}" if test is not None else "—"})
        annotations.append("" if arm == "Vanilla" else f"{summary['reduction_of_mean_pct']:+.1f}%")
    table8 = pd.DataFrame(rows)
    write_table(table8, out / "table_ablation", pct_columns=["Reduction of mean %", "Mean reduction %"],
                caption="Component ablation on the Linear host (main batch): cross-dataset mean metrics and reductions "
                        "relative to Vanilla Linear.")
    _bar_figure(list(ARMS), table8["Mean RMSE"].tolist(), annotations, "Mean RMSE (seed means over datasets)",
                "Component ablation", out / "fig8", PALETTE[0])
    fig8_means = dict(zip(ARMS, table8["Mean RMSE"]))
    fig8_reductions = {"expert_only_vs_vanilla": reductions(vanilla, means["+Expert"]["rmse"])["reduction_of_mean_pct"],
                       "full_vs_expert": reductions(means["+Expert"]["rmse"], means["Full"]["rmse"])["reduction_of_mean_pct"],
                       "full_vs_vanilla": reductions(vanilla, means["Full"]["rmse"])["reduction_of_mean_pct"]}

    # Architecture × objective factorial (R1.1/R2.1).
    factorial = _means(frame, {f"{a} | {o}": f for (a, o), f in FACTORIAL.items()}, datasets, seeds, allow_missing)
    rows = []
    for architecture in ("Vanilla", "+Expert", "Full"):
        for metric in ("rmse", "mae", "mda"):
            rows.append({"Architecture": architecture, "Metric": metric.upper(),
                         **{objective: float(factorial[f"{architecture} | {objective}"][metric].mean()) for objective in OBJECTIVES}})
    write_table(pd.DataFrame(rows), out / "table_objective_factorial",
                caption="Cross-dataset mean metrics of three architectures under three objectives: the prediction loss "
                        "alone, with its training terms (anchoring MSE for Vanilla, the auxiliary losses otherwise), and MSE only.")
    contrasts, contrast_rows = {}, []
    for key, (name, base, cand) in CONTRASTS.items():
        table = compare(frame, base, cand, datasets, seeds, allow_missing=allow_missing, labels=(name, name))
        summary = reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])
        tests = overall(table["reduction_pct"])
        contrasts[key] = {"name": name, **summary, **tests, "significant": int(table["significant"].sum()), "n": len(table)}
        contrast_rows.append({"Contrast": f"({key}) {name}", "Baseline mean": summary["base_mean"],
                              "Candidate mean": summary["cand_mean"], "Reduction of mean %": summary["reduction_of_mean_pct"],
                              "Mean reduction %": summary["mean_of_reductions_pct"],
                              "Significant": f"{int(table['significant'].sum())}/{len(table)}",
                              "Wilcoxon p": tests["wilcoxon_p"], "Sign-test p": tests["sign_p"]})
    write_table(pd.DataFrame(contrast_rows), out / "table_objective_contrasts", pct_columns=["Reduction of mean %", "Mean reduction %"],
                p_columns=["Wilcoxon p", "Sign-test p"],
                caption="Contrasts separating architecture from objective; positive reductions favour the candidate.")

    # Fig. 9: gate ablation in the later batch.
    gates = _means(frame, GATES, datasets, seeds, allow_missing)
    rmse_matrix = pd.DataFrame({gate: gates[gate]["rmse"] for gate in GATES})
    mda_matrix = pd.DataFrame({gate: gates[gate]["mda"] for gate in GATES})
    best_rmse, best_mda = rmse_matrix.idxmin(axis=1), mda_matrix.idxmax(axis=1)
    rows = [{"Variant": gate, "Mean RMSE": float(rmse_matrix[gate].mean()), "Best RMSE on": f"{int((best_rmse == gate).sum())}/{len(rmse_matrix)}",
             "Mean MDA": float(mda_matrix[gate].mean()), "Best MDA on": f"{int((best_mda == gate).sum())}/{len(mda_matrix)}"}
            for gate in GATES]
    gate_table = pd.DataFrame(rows)
    write_table(gate_table, out / "table_gate_ablation", bold=["Mean RMSE"],
                caption="Gate ablation (later batch, 120 epochs): no gate, a fixed 0.5 multiplier, an unconditional learned "
                        "scalar and the RAR gate.")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, column, best_column, title in ((axes[0], "Mean RMSE", "Best RMSE on", "(a) RMSE (lower is better)"),
                                           (axes[1], "Mean MDA", "Best MDA on", "(b) MDA (higher is better)")):
        values = gate_table[column].tolist()
        bars = ax.bar(range(len(GATES)), values, color=PALETTE[0])
        for bar, value, best in zip(bars, values, gate_table[best_column]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}\n{best}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(GATES)), list(GATES), rotation=15)
        ax.set_ylim(min(values) * 0.98, max(values) * 1.02)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, out / "fig9")

    # Incremental gains (R2.6): seed-level, DM and cross-dataset tests.
    incremental_rows, incremental_tables = [], {}
    for name, base, cand in INCREMENTAL:
        row, table, dm = _pair_summary(frame, name, base, cand, datasets, seeds, allow_missing, main_seed)
        incremental_rows.append(row)
        incremental_tables[name] = {"paired": table.to_dict("records"), "dm": dm.to_dict("records")}
    incremental = pd.DataFrame(incremental_rows)
    write_table(incremental, out / "table_incremental", pct_columns=["Reduction of mean %", "Mean reduction %"],
                p_columns=["Wilcoxon p", "Sign-test p"],
                caption="Significance of the incremental gains: Holm-corrected paired t tests and DM tests per dataset, "
                        "cross-dataset Wilcoxon and sign tests, and the seeds on which the candidate's mean RMSE is lower.")

    rar_constant = incremental.set_index("Comparison").loc["RAR gate vs Constant (0.5)"]
    summary = {"fig8_means": fig8_means, "fig8_reductions_pct": fig8_reductions,
               "fig8_significant_vs_vanilla": dict(zip(ARMS, table8["Significant vs Vanilla"])),
               "fig9_means": {GATE_KEY[gate]: float(rmse_matrix[gate].mean()) for gate in GATES},
               "fig9_mda": {GATE_KEY[gate]: float(mda_matrix[gate].mean()) for gate in GATES},
               "fig9_best_on_rmse": dict(zip(gate_table["Variant"], gate_table["Best RMSE on"])),
               "rar_vs_constant": {"reduction_of_mean_pct": float(rar_constant["Reduction of mean %"]),
                                   "p_overall": float(rar_constant["Wilcoxon p"]), "seed_level_wins": rar_constant["Seed-level wins"],
                                   "significant": rar_constant["Significant (paired t)"]},
               "rar_vs_no_gate_reduction_of_mean_pct": float(incremental.set_index("Comparison").loc["RAR gate vs No gate", "Reduction of mean %"]),
               "incremental": incremental.to_dict("records"), "factorial_contrasts": contrasts}
    write_json({**summary, "paper": PAPER, "incremental_tests": incremental_tables}, out / "ablation.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "ablation")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print("Fig. 8 means:", {k: round(v, 4) for k, v in summary["fig8_means"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
