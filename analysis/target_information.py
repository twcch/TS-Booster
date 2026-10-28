"""How much of the gain rides on the returns already known at the origin (R1.1/R1.3/R1.4/R2.4; spec §5.7):
the improvement per forecast step, the future-only target of plan 04 and the observed-window inputs.

    python analysis/target_information.py [--results results] [--output analysis/output/target_information]
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (ALIGNED, BACKBONE_LABEL, BACKBONES, DAGGER, DATASETS, PALETTE, PAPER_DEEP, SEEDS,  # noqa: E402
                             SEEDS_3, STATISTICAL, TREES, choose_learning_rate, classical, dataset_means, deep,
                             display, load_runs, ordered, require, select, tsb, write_figure, write_json, write_table)
from analysis.stats import compare, overall, reductions  # noqa: E402

WINDOW = 7
FUTURE = {"pred_offset": 6}  # steps 7-9: every return of the target window lies after the origin
LATER = {"train_epochs": 120}
OBSERVED = {"enc_in": 5, "obs_features": True}
HORIZON_SWEEP = {1: 86, 3: 71, 5: 57, 10: 30}  # % of constituent returns known, paper §4.5
TSB_FUTURE = "TS-Booster+Linear (steps 7-9)"
DEEP_LABEL = {"DLinear-paper": "DLinear", "PatchTST-paper": "PatchTST", "iTransformer-paper": "iTransformer"}
# article_v2 §4.1.1 and §4.5.
PAPER = {"known_share_pct_default_task": 71, "known_returns_per_step": {1: 6, 2: 5, 3: 4},
         "horizon_mean_reduction_pct": {1: 17.1, 3: 15.9, 5: 12.2, 10: 4.0}}


def _step_table(frame, datasets, seeds, allow_missing, extra=None):
    """Vanilla Linear vs TS-Booster+Linear per forecast step (RMSE, MAE, MDA)."""
    extra = extra or {}
    rows = []
    for step in (1, 2, 3):
        tests = {metric: compare(frame, tsb("TSBooster-vanilla", **extra), tsb(**extra), datasets, seeds,
                                 metric=f"{metric}_h{step}", allow_missing=allow_missing,
                                 labels=(f"Vanilla Linear h{step}", f"TS-Booster+Linear h{step}")) for metric in ("rmse", "mae", "mda")}
        rmse = tests["rmse"].set_index("dataset")
        summary = reductions(rmse["baseline_mean"], rmse["candidate_mean"])
        known = max(WINDOW - step - extra.get("pred_offset", 0), 0)
        rows.append({"Step": step + extra.get("pred_offset", 0), "Known returns": f"{known}/{WINDOW}",
                     "Known share %": 100 * known / WINDOW, "Vanilla RMSE": summary["base_mean"],
                     "TS-Booster RMSE": summary["cand_mean"], "Mean reduction %": summary["mean_of_reductions_pct"],
                     "Reduction of mean %": summary["reduction_of_mean_pct"],
                     "Significant": f"{int(rmse['significant'].sum())}/{len(rmse)}",
                     "MAE improvement %": float(tests["mae"]["reduction_pct"].mean()),
                     "MDA improvement %": float(tests["mda"]["reduction_pct"].mean())})
    return pd.DataFrame(rows)


def _backbone_table(frame, datasets, seeds, allow_missing, extra):
    rows, summary = [], {}
    for backbone in BACKBONES:
        tests = {metric: compare(frame, tsb("TSBooster-vanilla", backbone, **extra), tsb("TSBooster", backbone, **extra), datasets,
                                 seeds, metric=metric, allow_missing=allow_missing,
                                 labels=(f"Vanilla {backbone}", f"TS-Booster {backbone}")) for metric in ("rmse", "mae", "mda")}
        rmse = tests["rmse"].set_index("dataset")
        agg = reductions(rmse["baseline_mean"], rmse["candidate_mean"])
        summary[backbone] = {"reduction_pct": agg["mean_of_reductions_pct"], "reduction_of_mean_pct": agg["reduction_of_mean_pct"],
                             "significant": f"{int(rmse['significant'].sum())}/{len(rmse)}", "improved": int((rmse["reduction_pct"] > 0).sum())}
        rows.append({"Backbone": BACKBONE_LABEL[backbone], "Vanilla": agg["base_mean"], "+TS-Booster": agg["cand_mean"],
                     "Reduction %": agg["mean_of_reductions_pct"], "Reduction of mean %": agg["reduction_of_mean_pct"],
                     "Improved": f"{summary[backbone]['improved']}/{len(rmse)}", "Significant": summary[backbone]["significant"],
                     "MAE improvement %": float(tests["mae"]["reduction_pct"].mean()),
                     "MDA improvement %": float(tests["mda"]["reduction_pct"].mean())})
    return pd.DataFrame(rows), summary


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Improvement per forecast step of the default task, and on the future-only target.
    steps = _step_table(frame, datasets, seeds, allow_missing)
    future_steps = _step_table(frame, datasets, seeds, allow_missing, FUTURE)
    per_step = pd.concat([steps.assign(Target="Default (steps 1-3)"), future_steps.assign(Target="Future only (steps 7-9)")], ignore_index=True)
    write_table(per_step[["Target", *steps.columns]], out / "table_per_horizon",
                pct_columns=["Known share %", "Mean reduction %", "Reduction of mean %", "MAE improvement %", "MDA improvement %"],
                caption="Vanilla Linear vs TS-Booster+Linear per forecast step: the share of the target window's returns "
                        "already known at the origin, and the improvement at that step.")
    sweep = {}
    for horizon, share in HORIZON_SWEEP.items():
        table = compare(frame, tsb("TSBooster-vanilla", pred_len=horizon, **LATER), tsb(pred_len=horizon, **LATER), datasets,
                        seeds[:3], allow_missing=allow_missing, labels=(f"Vanilla H={horizon}", f"TS-Booster H={horizon}"))
        sweep[horizon] = {"known_share_pct": share, "mean_reduction_pct": float(table["reduction_pct"].mean())}
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(per_step["Known share %"], per_step["Mean reduction %"], "o", color=PALETTE[0], label="Per forecast step (5 seeds)")
    lowest = per_step.groupby("Known share %")["Mean reduction %"].transform("min")  # stacked points: label the lowest below
    crowded = per_step.groupby("Known share %")["Mean reduction %"].transform("size") > 1
    for row, below in zip(per_step.to_dict("records"), (per_step["Mean reduction %"] == lowest) & crowded):
        ax.annotate(f"step {row['Step']}", (row["Known share %"], row["Mean reduction %"]), textcoords="offset points",
                    xytext=(4, -10) if below else (4, 4), fontsize=7)
    ax.plot([v["known_share_pct"] for v in sweep.values()], [v["mean_reduction_pct"] for v in sweep.values()], "s--",
            color=PALETTE[1], label="Horizon sweep, H = 1, 3, 5, 10 (3 seeds)")
    for horizon, value in sweep.items():
        ax.annotate(f"H={horizon}", (value["known_share_pct"], value["mean_reduction_pct"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Returns of the target window known at the forecast origin (%)")
    ax.set_ylabel("Mean RMSE reduction over Vanilla Linear (%)")
    ax.set_title("Improvement against the information already observed")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, out / "fig_known_share")

    # The future-only target: backbones and comparators.
    future_backbones, future_summary = _backbone_table(frame, datasets, seeds, allow_missing, FUTURE)
    write_table(future_backbones, out / "table_future_backbones",
                pct_columns=["Reduction %", "Reduction of mean %", "MAE improvement %", "MDA improvement %"],
                caption="Steps 7-9 (every constituent return after the origin): each backbone alone and with TS-Booster.")
    # (filters, seeds, frame to select from): the deep comparators of plan 09 come with the learning
    # rate Table 6 chose per dataset; when more than one rate was run, Table 6's rule picks among them.
    methods = {TSB_FUTURE: (tsb(**FUTURE), seeds, frame)}
    methods.update({model: (classical(model, pred_offset=6), [0], frame) for model in STATISTICAL + ALIGNED})
    methods.update({model: (classical(model, pred_offset=6), seeds, frame) for model in TREES})
    tsb_future_rows = require(frame, tsb(**FUTURE), datasets, seeds, TSB_FUTURE, allow_missing)
    search_seeds = [seed for seed in SEEDS_3 if seed in seeds] or list(seeds)
    for model in PAPER_DEEP:
        rates = sorted(select(frame, **deep(model, **FUTURE))["learning_rate"].dropna().unique())
        chosen = choose_learning_rate(frame, deep(model, **FUTURE), datasets, search_seeds, candidates=rates)
        rows_ = pd.concat([require(frame, deep(model, float(row.chosen), **FUTURE), [row.dataset], seeds,
                                   f"{model} {row.dataset} (steps 7-9)", allow_missing) for row in chosen.itertuples()])
        methods[DEEP_LABEL[model]] = (deep(model, **FUTURE), seeds, pd.concat([tsb_future_rows, rows_]))
    means = {}
    for label, (filters, method_seeds, source) in methods.items():
        grouped = require(source, filters, datasets, method_seeds, label, allow_missing).groupby("dataset")[["rmse", "mae", "mda"]].mean()
        means[label] = grouped.reindex(ordered(grouped.index, datasets))

    def methods_table(order, stem, caption):
        rows = [{"Dataset": display(d), **{display(m): means[m].loc[d, "rmse"] for m in order}} for d in means[TSB_FUTURE].index]
        for metric, label in (("rmse", "Avg. RMSE"), ("mae", "Avg. MAE"), ("mda", "Avg. MDA")):
            rows.append({"Dataset": label, **{display(m): float(means[m][metric].mean()) for m in order}})
        write_table(pd.DataFrame(rows), stem, bold=[display(m) for m in order], higher_rows=[len(rows) - 1], caption=caption)

    deep_labels = [DEEP_LABEL[model] for model in PAPER_DEEP]
    methods_table([m for m in methods if m not in deep_labels], out / "table_future_comparators",
                  "RMSE on the future-only target (steps 7-9) and cross-dataset means; dagger-marked models forecast "
                  "single-day conditional volatility.")
    methods_table([TSB_FUTURE] + deep_labels, out / "table_future_deep",
                  "The paper's deep comparators on the future-only target (steps 7-9), at the learning rate Table 6 "
                  "chose per dataset on the main target; MSE + Adam versus TS-Booster's hybrid objective.")
    significance_rows, future_tests = [], {}
    for label, (filters, method_seeds, source) in methods.items():
        if label == TSB_FUTURE:
            continue
        table = compare(source, filters, tsb(**FUTURE), datasets, seeds, base_seeds=method_seeds if method_seeds == [0] else None,
                        allow_missing=allow_missing, labels=(label, TSB_FUTURE))
        future_tests[label] = table.to_dict("records")
        weak = table[~table["significant"]]
        significance_rows.append({"Comparator": display(label), "Significant": f"{int(table['significant'].sum())}/{len(table)}",
                                  "Mean reduction %": float(table["reduction_pct"].mean()),
                                  "Reduction of mean %": reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])["reduction_of_mean_pct"],
                                  "No significant advantage": "; ".join(f"{display(r.dataset)} ({r.reduction_pct:+.1f}%)" for r in weak.itertuples()) or "—"})
    write_table(pd.DataFrame(significance_rows), out / "table_future_significance", pct_columns=["Mean reduction %", "Reduction of mean %"],
                caption="TS-Booster+Linear against each comparator on the future-only target: Holm-corrected one-sided tests.")
    common = [m for m in methods if m not in DAGGER]
    future_means = pd.Series({m: float(means[m]["rmse"].mean()) for m in common})
    future_deep = {label: reductions(means[label]["rmse"], means[TSB_FUTURE]["rmse"])["reduction_of_mean_pct"] for label in deep_labels}

    # Observed-window features as inputs (R1.4).
    inputs = {"Vanilla Linear": (tsb("TSBooster-vanilla"), "x, r"), "Vanilla Linear + obs": (tsb("TSBooster-vanilla", **OBSERVED), "x, r, obs h1-h3"),
              "TS-Booster+Linear": (tsb(), "x, r"), "TS-Booster+Linear + obs": (tsb(**OBSERVED), "x, r, obs h1-h3"),
              "DLinear + obs (HYBRID)": ({"model": "DLinear-paper", "loss": "HYBRID", "train_mse_weight": 0.0, "seq_len": 30, "pred_len": 3, "pred_offset": 0, **OBSERVED}, "x, r, obs h1-h3"),
              "iTransformer + obs (HYBRID)": ({"model": "iTransformer-paper", "loss": "HYBRID", "train_mse_weight": 0.0, "seq_len": 30, "pred_len": 3, "pred_offset": 0, **OBSERVED}, "x, r, obs h1-h3")}
    rows = []
    for label, (filters, description) in inputs.items():
        rows.append({"Model": label, "Inputs": description,
                     **{f"Mean {m.upper()}": float(dataset_means(frame, filters, datasets, seeds, m, label, allow_missing).mean()) for m in ("rmse", "mae", "mda")}})
    rule = require(frame, classical("ObservedWindowRule"), datasets, [0], "ObservedWindowRule", allow_missing)[["rmse", "mae", "mda"]].mean()
    rows.append({"Model": "Observed-window rule (Eq. 19)", "Inputs": "known returns only", "Mean RMSE": float(rule["rmse"]),
                 "Mean MAE": float(rule["mae"]), "Mean MDA": float(rule["mda"])})
    contrasts = {"information_value": ("Information: Vanilla Linear + obs vs Vanilla Linear", inputs["Vanilla Linear"][0], inputs["Vanilla Linear + obs"][0]),
                 "module_beyond_information": ("Module beyond the information: TS-Booster + obs vs Vanilla + obs", inputs["Vanilla Linear + obs"][0], inputs["TS-Booster+Linear + obs"][0]),
                 "features_for_tsb": ("Features for TS-Booster: TS-Booster + obs vs TS-Booster", inputs["TS-Booster+Linear"][0], inputs["TS-Booster+Linear + obs"][0]),
                 "tsb_vs_dlinear_obs": ("TS-Booster + obs vs DLinear + obs", inputs["DLinear + obs (HYBRID)"][0], inputs["TS-Booster+Linear + obs"][0]),
                 "tsb_vs_itransformer_obs": ("TS-Booster + obs vs iTransformer + obs", inputs["iTransformer + obs (HYBRID)"][0], inputs["TS-Booster+Linear + obs"][0])}
    contrast_rows, observed = [], {}
    for key, (name, base, cand) in contrasts.items():
        table = compare(frame, base, cand, datasets, seeds, allow_missing=allow_missing, labels=(name, name))
        agg = reductions(table.set_index("dataset")["baseline_mean"], table.set_index("dataset")["candidate_mean"])
        tests = overall(table["reduction_pct"])
        observed[key] = {**agg, **tests, "significant": f"{int(table['significant'].sum())}/{len(table)}"}
        contrast_rows.append({"Contrast": name, "Baseline mean RMSE": agg["base_mean"], "Candidate mean RMSE": agg["cand_mean"],
                              "Mean reduction %": agg["mean_of_reductions_pct"], "Reduction of mean %": agg["reduction_of_mean_pct"],
                              "Significant": observed[key]["significant"], "Wilcoxon p": tests["wilcoxon_p"]})
    write_table(pd.DataFrame(rows), out / "table_observed_features", bold=["Mean RMSE"],
                caption="The observed-window features obs_vol_h1..h3 (Eq. 19) as extra inputs (R1.4).",
                note="DLinear and iTransformer here use the hybrid objective and no learning-rate selection, unlike Table 6.")
    write_table(pd.DataFrame(contrast_rows), out / "table_observed_features_contrasts", pct_columns=["Mean reduction %", "Reduction of mean %"],
                p_columns=["Wilcoxon p"], caption="Contrasts on the observed-window inputs; positive reductions favour the candidate.")

    by_step = lambda table: {int(row["Step"]): row["Mean reduction %"] for row in table.to_dict("records")}
    summary = {"per_horizon_reduction_pct": by_step(steps), "future_per_horizon_reduction_pct": by_step(future_steps),
               "horizon_sweep": sweep, "future_backbones": future_summary,
               "future_best_method": str(future_means.idxmin()), "future_mean_rmse": future_means.to_dict(),
               "future_deep_reduction_of_mean_pct": future_deep,
               "observed_features": {"information_value_pct": observed["information_value"]["reduction_of_mean_pct"],
                                     "module_beyond_information_pct": observed["module_beyond_information"]["reduction_of_mean_pct"],
                                     "significant": {key: value["significant"] for key, value in observed.items()}, "contrasts": observed}}
    write_json({**summary, "paper": PAPER, "per_step": per_step.to_dict("records"), "future_tests": future_tests,
                "observed_rows": rows}, out / "target_information.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "target_information")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print("per-step reductions:", {k: round(v, 1) for k, v in summary["per_horizon_reduction_pct"].items()},
          "| future best:", summary["future_best_method"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
