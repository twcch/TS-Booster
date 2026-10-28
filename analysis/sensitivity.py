"""Fig. 7 (horizon and look-back), the §4.5 sensitivity checks, the loss components, the expert kernels
and the decomposition controls of 05 (spec §5.4).

    python analysis/sensitivity.py [--results results] [--output analysis/output/sensitivity]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (BACKBONE_LABEL, BACKBONES, DATASETS, PALETTE, SEEDS, dataset_means, display,  # noqa: E402
                             load_runs, tsb, write_figure, write_json, write_table)
from analysis.stats import compare, overall, reductions  # noqa: E402

LATER = {"train_epochs": 120}
HORIZONS, LOOKBACKS = (1, 3, 5, 10), (30, 60, 90)
KNOWN_SHARE = {1: 86, 3: 71, 5: 57, 10: 30}  # % of the targets' constituent returns known at the origin (paper §4.5)
SENSITIVITY = {"decomp_kernels": [(5, 11), (11, 21), (7, 13, 21)], "n_experts": [1, 3], "aux": [0.1, 0.5], "base": [0.0, 0.3]}
LOSS_COMPONENTS = {"no_direction": ("Without the direction term (λ_dir = 0)", tsb(dir_weight=0.0)),
                   "no_residual": ("Without the residual term (α = 0)", tsb(aux=0.0)),
                   "no_anchoring": ("Without the anchoring term (β = 0)", tsb(base=0.0))}
EXPERT_KERNELS = [(3,), (5,), (7,), (3, 7), (3, 5, 7, 9)]
DECOMPOSITION_CONTROLS = {"[x ‖ x] (raw twice)": tsb(enrich="raw"), "[x ‖ 0] (zeros)": tsb(enrich="zeros"),
                          "Fixed equal decomposition weights": tsb(decomp_fixed=True),
                          "+Expert (no decomposition)": tsb("TSBooster-wodecomp"),
                          "One expert of the two experts' width": tsb(n_experts=1, expert_hidden=60)}
# article_v2 §4.5 and Fig. 7.
PAPER = {"horizon_mean_reduction_pct": {1: 17.1, 3: 15.9, 5: 12.2, 10: 4.0},
         "horizon_improved": {1: "16/16", 3: "16/16", 5: "16/16", 10: "14/16"},
         "lookback_mean_reduction_pct": {30: 15.9, 60: 14.5, 90: 13.7},
         "sensitivity_change_pct": {"decomp kernels": "−0.11 to +0.09", "n_experts 1 or 3": "≤ 0.15 in magnitude",
                                    "aux=0.1": 3.0, "aux=0.5": -1.0, "base=0.0": 2.1}}


def _key(name, value):
    return str(list(value)) if isinstance(value, (list, tuple)) else f"{name}={value}"


def _arms_table(frame, default, arms, datasets, seeds, allow_missing):
    """Each arm against the default: cross-dataset means, change in mean RMSE (%), per-dataset tests."""
    base = {metric: dataset_means(frame, default, datasets, seeds, metric, "default", allow_missing) for metric in ("rmse", "mae", "mda")}
    rows, changes = [{"Variant": "Default", "Mean RMSE": float(base["rmse"].mean()), "Change %": 0.0,
                      "Mean MAE": float(base["mae"].mean()), "Mean MDA": float(base["mda"].mean()),
                      "Datasets better": "—", "Significant (better)": "—", "Wilcoxon p": float("nan")}], {}
    for key, (label, filters) in arms.items():
        table = compare(frame, default, filters, datasets, seeds, allow_missing=allow_missing, labels=("default", label))
        mae = dataset_means(frame, filters, datasets, seeds, "mae", label, allow_missing)
        mda = dataset_means(frame, filters, datasets, seeds, "mda", label, allow_missing)
        change = float(-reductions(base["rmse"], table.set_index("dataset")["candidate_mean"])["reduction_of_mean_pct"])
        changes[key] = change
        tests = overall(table["reduction_pct"])
        rows.append({"Variant": label, "Mean RMSE": float(table["candidate_mean"].mean()), "Change %": change,
                     "Mean MAE": float(mae.mean()), "Mean MDA": float(mda.mean()),
                     "Datasets better": f"{int((table['reduction_pct'] > 0).sum())}/{len(table)}",
                     "Significant (better)": f"{int(table['significant'].sum())}/{len(table)}", "Wilcoxon p": tests["wilcoxon_p"]})
    return pd.DataFrame(rows), changes


def _sweep(frame, values, make_filters, datasets, seeds, allow_missing, label):
    """Vanilla vs TS-Booster for each value of a sweep (horizon or look-back)."""
    rows, per_dataset = [], {}
    for value in values:
        table = compare(frame, make_filters("TSBooster-vanilla", value), make_filters("TSBooster", value), datasets, seeds,
                        allow_missing=allow_missing, labels=(f"Vanilla {label} {value}", f"TS-Booster {label} {value}"))
        summary = reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])
        per_dataset[value] = table
        rows.append({label: value, "Vanilla mean RMSE": summary["base_mean"], "TS-Booster mean RMSE": summary["cand_mean"],
                     "Mean reduction %": summary["mean_of_reductions_pct"], "Reduction of mean %": summary["reduction_of_mean_pct"],
                     "Improved": f"{int((table['reduction_pct'] > 0).sum())}/{len(table)}",
                     "Significant": f"{int(table['significant'].sum())}/{len(table)}"})
    return pd.DataFrame(rows), per_dataset


def _sweep_figure(table, per_dataset, label, xlabel, title, stem):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.5 * len(table) + 2, 3.4))
    x = np.arange(len(table))
    bars = ax.bar(x, table["Mean reduction %"], color=PALETTE[0], alpha=0.85, width=0.6)
    for k, value in enumerate(table[label]):
        points = per_dataset[value]["reduction_pct"].to_numpy()
        ax.scatter(np.full(len(points), x[k]) + np.linspace(-0.15, 0.15, len(points)), points, color=PALETTE[3], s=12, zorder=3)
    for bar, mean, improved in zip(bars, table["Mean reduction %"], table["Improved"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.1f}%\n{improved}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x, [str(v) for v in table[label]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("RMSE improvement over Vanilla Linear (%)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, stem)


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    three = list(seeds[:3])

    # Fig. 7: horizon and look-back (later batch, three seeds).
    horizon, horizon_sets = _sweep(frame, HORIZONS, lambda variant, h: tsb(variant, pred_len=h, **LATER), datasets, three,
                                   allow_missing, "H")
    horizon.insert(1, "Known returns %", [KNOWN_SHARE[h] for h in horizon["H"]])
    write_table(horizon, out / "table_horizon", pct_columns=["Mean reduction %", "Reduction of mean %"],
                caption="Forecast horizon at L = 30 (later batch, three seeds): TS-Booster+Linear against Vanilla Linear.")
    _sweep_figure(horizon, horizon_sets, "H", "Forecast horizon H (L = 30)", "(a) Sensitivity to the forecast horizon", out / "fig7_horizon")
    lookback, lookback_sets = _sweep(frame, LOOKBACKS, lambda variant, l: tsb(variant, seq_len=l, **LATER), datasets, three,
                                     allow_missing, "L")
    write_table(lookback, out / "table_lookback", pct_columns=["Mean reduction %", "Reduction of mean %"],
                caption="Look-back window at H = 3 (later batch, three seeds).")
    _sweep_figure(lookback, lookback_sets, "L", "Look-back window L (H = 3)", "(b) Sensitivity to the look-back window", out / "fig7_lookback")

    # §4.5 architectural and loss-weight sensitivity (three seeds).
    arms = {_key(name, value): (f"{name} = {list(value) if isinstance(value, tuple) else value}", tsb(**{name: value}))
            for name, values in SENSITIVITY.items() for value in values}
    sensitivity, sensitivity_change = _arms_table(frame, tsb(), arms, datasets, three, allow_missing)
    write_table(sensitivity, out / "table_sensitivity", pct_columns=["Change %"], p_columns=["Wilcoxon p"],
                caption="Architectural and loss-weight sensitivity of TS-Booster+Linear (three seeds): change in the "
                        "cross-dataset mean RMSE relative to the default; positive is worse.")
    losses, loss_change = _arms_table(frame, tsb(), {key: (label, filters) for key, (label, filters) in LOSS_COMPONENTS.items()},
                                      datasets, three, allow_missing)
    write_table(losses, out / "table_loss_components", pct_columns=["Change %"], p_columns=["Wilcoxon p"],
                caption="Removing one term of the objective at a time (three seeds).")
    kernels, kernel_change = _arms_table(frame, tsb(), {_key("expert_kernels", k): (f"Kernels {list(k)}", tsb(expert_kernels=k))
                                                        for k in EXPERT_KERNELS}, datasets, three, allow_missing)
    write_table(kernels, out / "table_expert_kernels", pct_columns=["Change %"], p_columns=["Wilcoxon p"],
                caption="Single-kernel and alternative multi-kernel experts against the default {3, 5, 7} (three seeds; R2.3).")

    # Decomposition controls of 05 (five seeds).
    controls, control_change = _arms_table(frame, tsb(), {label: (label, filters) for label, filters in DECOMPOSITION_CONTROLS.items()},
                                           datasets, seeds, allow_missing)
    controls.loc[0, "Variant"] = "[x ‖ trend] (default)"
    write_table(controls, out / "table_decomposition_controls", pct_columns=["Change %"], p_columns=["Wilcoxon p"],
                caption="Decomposition controls: same-width inputs, frozen decomposition weights, no decomposition and a "
                        "single wide expert (R1.9/R2.3).")
    rows, decomp_backbones = [], {}
    pairs = [(BACKBONE_LABEL[b], tsb("TSBooster-vanilla", b), tsb("TSBooster-woexpert", b)) for b in BACKBONES]
    pairs.append(("Linear (frozen weights)", tsb("TSBooster-vanilla"), tsb("TSBooster-woexpert", decomp_fixed=True)))
    for label, vanilla, decomposed in pairs:
        table = compare(frame, vanilla, decomposed, datasets, seeds, allow_missing=allow_missing, labels=(f"Vanilla {label}", f"+Decomp {label}"))
        summary = reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])
        decomp_backbones[label] = {"change_pct": -summary["reduction_of_mean_pct"], "significant": int(table["significant"].sum()), "n": len(table)}
        rows.append({"Backbone": label, "Vanilla mean RMSE": summary["base_mean"], "+Decomp mean RMSE": summary["cand_mean"],
                     "Change %": -summary["reduction_of_mean_pct"], "Mean reduction %": summary["mean_of_reductions_pct"],
                     "Datasets better": f"{int((table['reduction_pct'] > 0).sum())}/{len(table)}",
                     "Significant (better)": f"{int(table['significant'].sum())}/{len(table)}"})
    write_table(pd.DataFrame(rows), out / "table_decomp_backbones", pct_columns=["Change %", "Mean reduction %"],
                caption="The decomposition-only variant (+Decomp) against each vanilla backbone (R1.9); positive change is worse.")

    sweep_summary = lambda table, key: {int(row[key]): {"mean_reduction_pct": row["Mean reduction %"], "improved": row["Improved"],
                                                       "significant": row["Significant"]} for row in table.to_dict("records")}
    summary = {"horizon": sweep_summary(horizon, "H"), "lookback": sweep_summary(lookback, "L"),
               "sensitivity_change_pct": sensitivity_change, "loss_components_change_pct": loss_change,
               "expert_kernels_change_pct": kernel_change, "decomposition_controls_change_pct": control_change,
               "decomp_backbones": decomp_backbones}
    write_json({**summary, "paper": PAPER, "tables": {"horizon": horizon.to_dict("records"), "lookback": lookback.to_dict("records"),
                                                      "sensitivity": sensitivity.to_dict("records"), "loss_components": losses.to_dict("records"),
                                                      "expert_kernels": kernels.to_dict("records"), "decomposition_controls": controls.to_dict("records")}},
               out / "sensitivity.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "sensitivity")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print("horizon:", {h: round(v["mean_reduction_pct"], 1) for h, v in summary["horizon"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
