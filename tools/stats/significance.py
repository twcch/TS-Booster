"""Significance tests over completed runs, ported from the journal analysis scripts.

Reads results/<run>/result.json (and pred.npy/true.npy for DM). A method is
MODEL[:key=value,...] matched against each run's config, e.g.
TSBooster:tsb_backbone=lstm or TSBooster:tsb_decomp_kernels=5+11 (list values
join with '+'). Repeated attempts of one seed keep the latest.

  paired    per dataset: one-sided paired t-test over seeds (H1: candidate better),
            or a one-sample test against a deterministic baseline with a single run
            (tools/baselines/classical_baselines.py), 95% CI of the mean % gap,
            paired Cohen's d, Holm across datasets; overall:
            Wilcoxon signed-rank, binomial sign test and one-sample t-test on the gaps
            (experiments/statistical_test.py, revision/phase5_stat_upgrade/mean_std_holm.py)
  dm        Diebold-Mariano with Newey-West variance (h-1 lags) and the HLN (1997)
            correction on per-window MSE over the horizon (phase5 dm_test_runner.py)
  friedman  ranks of the seed-mean metric per dataset, Friedman chi2 and the Nemenyi
            critical difference of Demsar (2006) (experiments/cd_diagram.py)

    python tools/stats/significance.py paired --baseline TSBooster-vanilla --candidate TSBooster
    python tools/stats/significance.py dm --baseline TSBooster-vanilla --candidate TSBooster --seed 42
    python tools/stats/significance.py friedman --methods TSBooster DLinear PatchTST --plot cd.png
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parse_result_to_xlsx import _complete_record, _result_files  # noqa: E402

HIGHER_IS_BETTER = {"mda", "r2"}
# Demsar (2006) Table 5: q_alpha for the two-tailed Nemenyi test, k = 2..10.
NEMENYI_Q = {0.05: {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
             0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920}}


# === Collecting runs ===


def parse_method(spec):
    model, _, rest = spec.partition(":")
    filters = {}
    for item in filter(None, rest.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Method filter must be key=value, got {item!r} in {spec!r}")
        filters[key] = value
    return model, filters


def _render(value):
    """Config values as written in method specs; lists join with '+', e.g. 5+11."""
    return "+".join(map(str, value)) if isinstance(value, list) else str(value)


def _matches(config, filters):
    return all(_render(config.get(key)) == value for key, value in filters.items())


def collect(results, scale="primary", where=()):
    """One row per complete attempt: dataset, model, seed, metrics, config and run path."""
    filters = dict(item.split("=", 1) for item in where)
    rows = []
    for filename in _result_files(results):
        pair = _complete_record(filename)
        if pair is None:
            continue
        result, metadata = pair
        config, details = result["config"], result.get("details", {})
        if not _matches(config, filters):
            continue
        metrics = result["metrics"] if scale == "primary" else details.get(f"metrics_{scale}")
        if metrics is None:
            continue
        dataset = Path(config["data_path"]).stem if config.get("data") == "custom" else config["data"]
        rows.append({"dataset": dataset, "model": config["model"], "seed": config["rand_seed"],
                     "metrics": metrics, "config": config, "path": filename.parent,
                     "metric_space": details.get("metric_space") if scale == "primary" else scale,
                     "finished_at": metadata.get("finished_at")})
    return rows


def select(rows, spec, metric):
    """{dataset: {seed: (value, row)}} for one method, latest attempt per seed."""
    model, filters = parse_method(spec)
    chosen = {}
    for row in sorted(rows, key=lambda r: str(r["finished_at"])):
        if row["model"] != model or not _matches(row["config"], filters):
            continue
        value = row["metrics"].get(metric)
        if value is None:
            continue
        chosen.setdefault(row["dataset"], {})[row["seed"]] = (float(value), row)
    return chosen


def _check_one_space(*selections):
    spaces = {row["metric_space"] for selection in selections for seeds in selection.values()
              for _, row in seeds.values()}
    if len(spaces) > 1:
        raise ValueError(f"Runs mix metric spaces {sorted(spaces)}; pass --scale original or normalized")


# === Tests ===


def holm_bonferroni(p_values, alpha=0.05):
    """Reject decisions after the Holm step-down correction (stops at the first acceptance)."""
    rejected = [False] * len(p_values)
    for rank, index in enumerate(np.argsort(p_values)):
        if p_values[index] <= alpha / (len(p_values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def one_sided(t_stat, p_two):
    return p_two / 2 if t_stat > 0 else 1 - p_two / 2


def paired_test(baseline, candidate, metric):
    """Seed-paired comparison; positive gaps mean the candidate is better."""
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    if metric in HIGHER_IS_BETTER:
        improvement, gaps = c - b, (c - b) / np.abs(b) * 100
        t_stat, p_two = stats.ttest_rel(c, b)
    else:
        improvement, gaps = b - c, (b - c) / b * 100
        t_stat, p_two = stats.ttest_rel(b, c)
    se = stats.sem(gaps)
    mean_gap = float(np.mean(gaps))
    return {"n_seeds": len(b), "baseline_mean": float(b.mean()), "baseline_std": float(b.std(ddof=1)),
            "candidate_mean": float(c.mean()), "candidate_std": float(c.std(ddof=1)),
            "mean_gap_pct": mean_gap, "ci95_gap_pct": [mean_gap - 1.96 * se, mean_gap + 1.96 * se],
            "cohens_d": float(improvement.mean() / (improvement.std(ddof=1) + 1e-12)),
            "t_stat": float(t_stat), "p_one_sided": float(one_sided(t_stat, p_two))}


def one_sample_test(fixed, candidate, metric):
    """Seed results against a deterministic baseline score (the paper's one-sample variant)."""
    c = np.asarray(candidate, float)
    improvement = c - fixed if metric in HIGHER_IS_BETTER else fixed - c
    gaps = improvement / abs(fixed) * 100
    t_stat, p_two = stats.ttest_1samp(improvement, 0)
    se = stats.sem(gaps)
    mean_gap = float(np.mean(gaps))
    return {"test": "one_sample", "n_seeds": len(c), "baseline_mean": float(fixed), "baseline_std": 0.0,
            "candidate_mean": float(c.mean()), "candidate_std": float(c.std(ddof=1)),
            "mean_gap_pct": mean_gap, "ci95_gap_pct": [mean_gap - 1.96 * se, mean_gap + 1.96 * se],
            "cohens_d": float(improvement.mean() / (improvement.std(ddof=1) + 1e-12)),
            "t_stat": float(t_stat), "p_one_sided": float(one_sided(t_stat, p_two))}


def dm_test(e1, e2, h=1):
    """Diebold-Mariano with HLN correction; H1: method 2 has lower loss (d = e1 - e2 > 0)."""
    n = min(len(e1), len(e2))
    d = np.asarray(e1[:n], float) - np.asarray(e2[:n], float)
    d_bar, T = d.mean(), len(d)
    gamma = sum((1 - lag / h) * np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)) for lag in range(1, h))
    lrv = (np.var(d, ddof=0) + 2 * gamma) / T
    dm = d_bar / np.sqrt(max(lrv, 1e-12)) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return float(dm), float(stats.t.cdf(-dm, df=T - 1)), T


def friedman(rank_matrix):
    n, k = rank_matrix.shape
    average = rank_matrix.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(average ** 2) - k * (k + 1) ** 2 / 4)
    return float(chi2), float(1 - stats.chi2.cdf(chi2, k - 1))


def nemenyi_cd(n_datasets, k, alpha=0.05):
    q = NEMENYI_Q[alpha].get(k, stats.norm.ppf(1 - alpha / (k * (k - 1))))
    return float(q * np.sqrt(k * (k + 1) / (6 * n_datasets)))


# === Commands ===


def run_paired(rows, args):
    base, cand = select(rows, args.baseline, args.metric), select(rows, args.candidate, args.metric)
    _check_one_space(base, cand)
    table = []
    for dataset in sorted(set(base) & set(cand)):
        seeds = sorted(set(base[dataset]) & set(cand[dataset]))
        if len(base[dataset]) == 1 and not seeds and len(cand[dataset]) >= 2:
            # A deterministic baseline (one run, e.g. persistence) against every candidate seed.
            (fixed, _), = base[dataset].values()
            seeds = sorted(cand[dataset])
            result = one_sample_test(fixed, [cand[dataset][s][0] for s in seeds], args.metric)
            table.append({"dataset": dataset, "seeds": seeds, **result})
            continue
        if len(seeds) < 2:
            print(f"skip {dataset}: fewer than two common seeds", file=sys.stderr)
            continue
        result = paired_test([base[dataset][s][0] for s in seeds], [cand[dataset][s][0] for s in seeds],
                             args.metric)
        table.append({"dataset": dataset, "seeds": seeds, "test": "paired", **result})
    if not table:
        raise ValueError("No dataset has both methods with at least two common seeds")
    for row, rejected in zip(table, holm_bonferroni([row["p_one_sided"] for row in table], args.alpha)):
        row["holm_rejected"] = rejected
    gaps = np.array([row["mean_gap_pct"] for row in table])
    wins = int((gaps > 0).sum())
    overall = {"n_datasets": len(table), "wins": wins, "mean_gap_pct": float(gaps.mean()),
               "sign_test_p": float(stats.binomtest(wins, len(gaps), 0.5, alternative="greater").pvalue)}
    if len(gaps) > 1:
        t_stat, p_two = stats.ttest_1samp(gaps, 0)
        se = stats.sem(gaps)
        overall.update(t_test_t=float(t_stat), t_test_p=float(one_sided(t_stat, p_two)),
                       ci95_gap_pct=[float(gaps.mean() - 1.96 * se), float(gaps.mean() + 1.96 * se)])
        if np.any(gaps != 0):
            w_stat, w_p = stats.wilcoxon(gaps, alternative="greater")
            overall.update(wilcoxon_w=float(w_stat), wilcoxon_p=float(w_p))
    print(f"{'dataset':<12} {'baseline':>12} {'candidate':>12} {'gap%':>8} {'d':>7} {'p':>9} holm")
    for row in table:
        print(f"{row['dataset']:<12} {row['baseline_mean']:>12.5f} {row['candidate_mean']:>12.5f} "
              f"{row['mean_gap_pct']:>+8.2f} {row['cohens_d']:>7.2f} {row['p_one_sided']:>9.2e} "
              f"{'*' if row['holm_rejected'] else ''}")
    print(json.dumps(overall, indent=1))
    return {"test": "paired", "metric": args.metric, "baseline": args.baseline, "candidate": args.candidate,
            "alpha": args.alpha, "per_dataset": table, "overall": overall}


def _window_errors(row):
    prediction, truth = np.load(row["path"] / "pred.npy"), np.load(row["path"] / "true.npy")
    return ((prediction - truth) ** 2).reshape(len(prediction), -1).mean(axis=1), truth


def run_dm(rows, args):
    base, cand = select(rows, args.baseline, "mse"), select(rows, args.candidate, "mse")
    _check_one_space(base, cand)
    table = []
    for dataset in sorted(set(base) & set(cand)):
        seeds = sorted(set(base[dataset]) & set(cand[dataset]))
        if args.seed is not None:
            seeds = [s for s in seeds if s == args.seed]
        for seed in seeds:
            (_, b_row), (_, c_row) = base[dataset][seed], cand[dataset][seed]
            e_base, true_base = _window_errors(b_row)
            e_cand, true_cand = _window_errors(c_row)
            if true_base.shape != true_cand.shape or not np.allclose(true_base, true_cand):
                raise ValueError(f"{dataset} seed {seed}: the two runs do not share test windows")
            horizon = int(c_row["config"]["pred_len"])
            stat, p_value, n = dm_test(e_base, e_cand, h=horizon)
            table.append({"dataset": dataset, "seed": seed, "dm_hln": stat, "p_one_sided": p_value,
                          "windows": n, "horizon": horizon})
    if not table:
        raise ValueError("No matching run pairs for the DM test")
    for row, rejected in zip(table, holm_bonferroni([row["p_one_sided"] for row in table], args.alpha)):
        row["holm_rejected"] = rejected
    for row in table:
        print(f"{row['dataset']:<12} seed {row['seed']:<5} DM={row['dm_hln']:+8.3f} p={row['p_one_sided']:.2e} "
              f"T={row['windows']} {'*' if row['holm_rejected'] else ''}")
    return {"test": "diebold_mariano_hln", "loss": "per-window MSE averaged over horizon and channels",
            "baseline": args.baseline, "candidate": args.candidate, "alpha": args.alpha, "results": table}


def run_friedman(rows, args):
    selections = {spec: select(rows, spec, args.metric) for spec in args.methods}
    _check_one_space(*selections.values())
    datasets = sorted(set.intersection(*(set(s) for s in selections.values())))
    if len(datasets) < 2:
        raise ValueError("Friedman needs at least two datasets shared by every method")
    means = np.array([[np.mean([v for v, _ in selections[spec][ds].values()]) for spec in args.methods]
                      for ds in datasets])
    ranks = np.vstack([stats.rankdata(-row if args.metric in HIGHER_IS_BETTER else row) for row in means])
    chi2, p_value = friedman(ranks)
    cd = nemenyi_cd(len(datasets), len(args.methods), args.alpha)
    average = dict(zip(args.methods, map(float, ranks.mean(axis=0))))
    for spec, rank in sorted(average.items(), key=lambda item: item[1]):
        print(f"{rank:6.3f}  {spec}")
    print(f"Friedman chi2={chi2:.3f} p={p_value:.2e}; Nemenyi CD={cd:.3f} (alpha={args.alpha}, N={len(datasets)})")
    if args.plot:
        plot_cd(average, cd, args.plot, f"{args.metric.upper()} (N={len(datasets)}, CD={cd:.2f})")
    return {"test": "friedman_nemenyi", "metric": args.metric, "alpha": args.alpha, "datasets": datasets,
            "methods": args.methods, "average_ranks": average, "friedman_chi2": chi2, "friedman_p": p_value,
            "nemenyi_cd": cd, "seed_mean_metric": means.tolist()}


def cd_cliques(ranks, cd):
    """Maximal runs of rank-sorted methods whose mean ranks differ by no more than the critical difference."""
    ranks, cliques = list(ranks), []
    for i in range(len(ranks)):
        j = i
        while j + 1 < len(ranks) and ranks[j + 1] - ranks[i] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    return cliques


def plot_cd(average, cd, path, title):
    """Critical-difference diagram in the layout of Demsar (2006): the rank axis on top with the best rank on
    the right, the CD scale bar above the axis at the left end, labels outside the axis ends, and thick bars
    below the axis joining methods that the Nemenyi critical difference does not separate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ordered = sorted(average.items(), key=lambda item: item[1])
    k = len(ordered)
    cliques = cd_cliques([rank for _, rank in ordered], cd)
    axis_y = k + 0.6 + 0.3 * len(cliques)  # the rank axis; stems start here, clique bars sit just below
    top = axis_y + 1.7                      # room above the axis for the tick labels and the CD bar
    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.32 * k + 0.1 * len(cliques)))
    ax.set_xlim(k, 1), ax.set_ylim(0.5, top), ax.axis("off")
    ax.plot([1, k], [axis_y, axis_y], color="black", linewidth=1.0)
    for r in range(1, k + 1):
        ax.plot([r, r], [axis_y, axis_y + 0.15], color="black", linewidth=1.0)
        ax.text(r, axis_y + 0.22, str(r), ha="center", va="bottom")
    y_cd = axis_y + 0.95  # CD scale bar above the axis, from the worst end inward
    ax.plot([k, k - cd], [y_cd, y_cd], color="red", linewidth=2, solid_capstyle="butt")
    for x in (k, k - cd):
        ax.plot([x, x], [y_cd - 0.08, y_cd + 0.08], color="red", linewidth=1.2)
    ax.text(k - cd / 2, y_cd + 0.12, "CD", ha="center", va="bottom", color="red")
    for level, (i, j) in enumerate(cliques):
        y = axis_y - 0.3 * (level + 1)
        ax.plot([ordered[i][1] - 0.03, ordered[j][1] + 0.03], [y, y], color="black", linewidth=3.2,
                solid_capstyle="butt", zorder=3)
    for index, (name, rank) in enumerate(ordered):
        y = k - index
        right = index < k / 2  # the better half is labelled on the right
        x_end = 1 if right else k
        ax.plot([rank, rank], [axis_y, y], color="black", linewidth=0.8)
        ax.plot([rank, x_end], [y, y], color="black", linewidth=0.8)
        # labels sit beyond the axis ends, so they never cross the connector lines
        ax.text(x_end, y, f" {name} ({rank:.2f})" if right else f"{name} ({rank:.2f}) ", va="center",
                ha="left" if right else "right", clip_on=False)
    ax.set_title(title, pad=6)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--scale", choices=["primary", "original", "normalized"], default="primary")
    parser.add_argument("--where", nargs="*", default=[], help="config filters for every run, e.g. pred_len=3")
    parser.add_argument("--alpha", type=float, default=0.05, choices=[0.05, 0.10])
    parser.add_argument("--output", type=Path, help="write the full report as JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    paired = sub.add_parser("paired")
    paired.add_argument("--baseline", required=True)
    paired.add_argument("--candidate", required=True)
    paired.add_argument("--metric", default="rmse")
    dm = sub.add_parser("dm")
    dm.add_argument("--baseline", required=True)
    dm.add_argument("--candidate", required=True)
    dm.add_argument("--seed", type=int, default=None, help="one seed (the paper used 42); default: all")
    ranks = sub.add_parser("friedman")
    ranks.add_argument("--methods", nargs="+", required=True)
    ranks.add_argument("--metric", default="rmse")
    ranks.add_argument("--plot", type=Path, help="save a critical-difference diagram")
    args = parser.parse_args(argv)
    rows = collect(args.results, args.scale, args.where)
    if not rows:
        parser.error(f"No complete runs found under {args.results}")
    report = {"paired": run_paired, "dm": run_dm, "friedman": run_friedman}[args.command](rows, args)
    report.update(results_root=str(args.results), scale=args.scale, where=args.where)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
