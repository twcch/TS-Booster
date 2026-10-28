"""Statistics of the analysis (spec §4).

The tests themselves come from tools/stats/significance.py (one-sided paired t, one-sample t
against a deterministic baseline, Holm step-down, Diebold–Mariano with the HLN correction,
Friedman and the Nemenyi critical difference). This module adds Holm-adjusted p values, the two
reduction aggregations of the paper, the comparison tables the modules share, and the backtest
pieces: Sharpe ratio, inverse-volatility position sizing and the Ledoit–Wolf (2008) studentized
circular block bootstrap.

Standard deviations reported in tables use ddof=0 (the paper's "divisor 5"); the tests inside
significance.py use ddof=1.
"""

import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stats.significance import (paired_test, one_sample_test, holm_bonferroni, dm_test, friedman,  # noqa: E402,F401
                                      nemenyi_cd, one_sided, plot_cd)
from analysis.common import MissingRuns, require  # noqa: E402

HIGHER_IS_BETTER = {"mda", "r2"}
TRADING_DAYS = 252


def _base_metric(metric):
    """'mda_h2' -> 'mda': the direction of a per-horizon metric is that of its base metric."""
    return metric.split("_h")[0]


# === Multiple comparisons and aggregation ===


def holm_adjusted(p_values):
    """Holm step-down adjusted p values, in the original order: p̃_(i) = max_{j<=i} min(1, (m-j+1) p_(j)).

    NaN p values (degenerate tests) stay NaN and do not count toward m.
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), math.nan)
    finite = np.flatnonzero(~np.isnan(p))
    running = 0.0
    for rank, index in enumerate(finite[np.argsort(p[finite], kind="stable")]):
        running = max(running, min(1.0, (len(finite) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def seed_test(baseline, candidate, metric):
    """One dataset's seed-level test: one-sample when the baseline is a single score, paired otherwise.

    Improvements without spread across seeds (ties, as MDA can produce) make the t statistic
    undefined; they return p = nan instead of scipy's degenerate value and its precision warning.
    """
    b, c = np.atleast_1d(np.asarray(baseline, dtype=float)), np.asarray(candidate, dtype=float)
    direction = _base_metric(metric)
    improvement = c - b if direction in HIGHER_IS_BETTER else b - c
    scale = max(1.0, float(np.abs(np.concatenate([b, c])).max()))
    if len(improvement) < 2 or improvement.std(ddof=1) <= 1e-12 * scale:
        return {"t_stat": math.nan, "p_one_sided": math.nan, "degenerate": True}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if b.size == 1:
            return one_sample_test(float(b[0]), c, direction)
        return paired_test(b, c, direction)


def reductions(base, cand):
    """The paper's two aggregations of a lower-is-better metric over datasets (Series aligned on index):
    the mean of the dataset-specific percentage reductions, and the reduction of the cross-dataset mean."""
    base, cand = pd.Series(base, dtype=float), pd.Series(cand, dtype=float)
    cand = cand.reindex(base.index)
    return {"base_mean": float(base.mean()), "cand_mean": float(cand.mean()),
            "mean_of_reductions_pct": float(((base - cand) / base * 100).mean()),
            "reduction_of_mean_pct": float((base.mean() - cand.mean()) / base.mean() * 100)}


def compare(frame, base_filters, cand_filters, datasets, seeds, *, metric="rmse", base_seeds=None, alpha=0.05,
            allow_missing=False, labels=("baseline", "candidate")):
    """Per-dataset seed-level comparison; positive reduction_pct means the candidate is better.

    A baseline with one run per dataset (a deterministic comparator, base_seeds=[0]) is compared
    with a one-sample test of the candidate's seeds against its score; otherwise the common seeds
    are paired. Means and stds (ddof=0) are over the seeds used; Holm runs across the datasets.
    """
    base = require(frame, base_filters, datasets, base_seeds or seeds, f"{labels[0]} {base_filters}", allow_missing)
    cand = require(frame, cand_filters, datasets, seeds, f"{labels[1]} {cand_filters}", allow_missing)
    higher = _base_metric(metric) in HIGHER_IS_BETTER
    rows = []
    for dataset in [d for d in datasets if d in set(base["dataset"]) & set(cand["dataset"])]:
        b = base[base["dataset"] == dataset].set_index("seed")[metric].astype(float)
        c = cand[cand["dataset"] == dataset].set_index("seed")[metric].astype(float)
        if len(b) == 1 and len(c) >= 2:
            test = "one_sample"
        else:
            common = sorted(set(b.index) & set(c.index))
            if len(common) < 2:
                raise MissingRuns(f"{dataset}: fewer than two common seeds between {base_filters} and {cand_filters}")
            b, c, test = b.loc[common], c.loc[common], "paired"
        result = seed_test(b.to_numpy(), c.to_numpy(), metric)
        b_mean, c_mean = float(b.mean()), float(c.mean())
        reduction = (c_mean - b_mean) / abs(b_mean) * 100 if higher else (b_mean - c_mean) / b_mean * 100
        rows.append({"dataset": dataset, "test": test, "baseline_mean": b_mean, "baseline_std": float(b.std(ddof=0)),
                     "candidate_mean": c_mean, "candidate_std": float(c.std(ddof=0)), "reduction_pct": reduction,
                     "t_stat": result["t_stat"], "p": result["p_one_sided"]})
    table = pd.DataFrame(rows)
    if table.empty:
        raise MissingRuns(f"No dataset has both {base_filters} and {cand_filters}")
    table["p_holm"] = holm_adjusted(table["p"].to_numpy())
    table["significant"] = table["p_holm"] <= alpha
    return table


def overall(gaps):
    """Cross-dataset summary of per-dataset gaps (positive = better): wins, sign test, Wilcoxon, t test."""
    gaps = np.asarray(gaps, dtype=float)
    n, wins = len(gaps), int((gaps > 0).sum())
    out = {"n": n, "wins": wins, "mean_gap_pct": float(gaps.mean()) if n else math.nan,
           "sign_p": float(sp.binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else math.nan,
           "wilcoxon_p": math.nan, "t_p": math.nan}
    if n > 1 and np.any(gaps != 0):
        out["wilcoxon_p"] = float(sp.wilcoxon(gaps, alternative="greater").pvalue)
        if gaps.std(ddof=1) > 0:
            t_stat, p_two = sp.ttest_1samp(gaps, 0)
            out["t_p"] = float(one_sided(t_stat, p_two))
    return out


def _window_errors(path):
    prediction, truth = np.load(Path(path) / "pred.npy"), np.load(Path(path) / "true.npy")
    return ((prediction - truth) ** 2).reshape(len(prediction), -1).mean(axis=1), truth


def dm_compare(frame, base_filters, cand_filters, datasets, seeds, *, alpha=0.05, base_seeds=None,
               allow_missing=False):
    """Diebold–Mariano (HLN) per dataset and candidate seed on per-window MSE; H1: candidate better.

    A deterministic baseline's single run is paired with every candidate seed. Holm runs within
    each seed across the datasets.
    """
    base = require(frame, base_filters, datasets, base_seeds or seeds, f"baseline {base_filters}", allow_missing)
    cand = require(frame, cand_filters, datasets, seeds, f"candidate {cand_filters}", allow_missing)
    rows = []
    for dataset in [d for d in datasets if d in set(base["dataset"]) & set(cand["dataset"])]:
        b_rows = base[base["dataset"] == dataset]
        for c_row in cand[cand["dataset"] == dataset].itertuples():
            matched = b_rows[b_rows["seed"] == c_row.seed] if len(b_rows) > 1 else b_rows
            if len(matched) != 1:
                raise MissingRuns(f"{dataset} seed {c_row.seed}: no baseline run to pair with")
            e_base, true_base = _window_errors(matched["path"].iloc[0])
            e_cand, true_cand = _window_errors(c_row.path)
            if true_base.shape != true_cand.shape or not np.allclose(true_base, true_cand, rtol=1e-5, atol=0):
                raise ValueError(f"{dataset} seed {c_row.seed}: the two runs do not share test windows")
            stat, p_value, n = dm_test(e_base, e_cand, h=int(c_row.pred_len))
            rows.append({"dataset": dataset, "seed": int(c_row.seed), "dm": stat, "p": p_value, "windows": n})
    table = pd.DataFrame(rows)
    if table.empty:
        raise MissingRuns(f"No run pairs for the DM test between {base_filters} and {cand_filters}")
    table["p_holm"] = table.groupby("seed")["p"].transform(lambda p: holm_adjusted(p.to_numpy()))
    table["significant"] = table["p_holm"] <= alpha
    return table


def rank_test(means, lower_is_better=True, alpha=0.05):
    """Friedman test and Nemenyi CD on a dataset × method table of seed means (ties get average ranks)."""
    values = np.round(means.to_numpy(dtype=float), 8)
    ranks = np.vstack([sp.rankdata(row if lower_is_better else -row) for row in values])
    chi2, p_value = friedman(ranks)
    n, k = means.shape
    return {"average_ranks": {str(method): float(rank) for method, rank in zip(means.columns, ranks.mean(axis=0))},
            "friedman_chi2": chi2, "friedman_p": p_value, "cd": nemenyi_cd(n, k, alpha), "k": int(k), "n": int(n)}


# === Backtest ===


def sharpe(returns):
    """Annualized Sharpe ratio with a zero risk-free rate and the population standard deviation."""
    r = np.asarray(returns, dtype=float)
    deviation = r.std()
    return float(np.sqrt(TRADING_DAYS) * r.mean() / deviation) if deviation > 0 else 0.0


def max_drawdown(returns):
    nav = np.cumprod(1 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(nav)
    return float(((nav - peak) / peak).min())


def inverse_volatility(pred_h1, log_return, *, target=15.0, floor=0.1, bounds=(0.2, 3.0), cost_bps=0.0):
    """Eq. 20 of the paper: w_t = clip(target / max(ŷ_{t+1}, floor), bounds); the position earns
    w_t (e^{r_{t+1}} − 1) less cost_bps/10⁴ per unit of |w_t − w_{t−1}| (w_{−1} = 0)."""
    pred, r = np.asarray(pred_h1, dtype=float), np.asarray(log_return, dtype=float)
    if pred.shape != r.shape or pred.ndim != 1:
        raise ValueError(f"predictions {pred.shape} and returns {r.shape} must be one-dimensional and aligned")
    if np.isnan(pred).any() or np.isnan(r).any():
        raise ValueError("predictions and returns must not contain NaN")
    weights = np.clip(target / np.maximum(pred, floor), bounds[0], bounds[1])
    turnover = np.abs(weights - np.concatenate([[0.0], weights[:-1]]))
    strategy = weights * np.expm1(r) - cost_bps / 1e4 * turnover
    nav = float(np.prod(1 + strategy))
    return {"weights": weights, "returns": strategy, "turnover": turnover, "sharpe": sharpe(strategy),
            "cumulative_return": nav - 1, "nav": nav, "max_drawdown": max_drawdown(strategy)}


def _sharpe_differences(r1, r2, bandwidth):
    """Annualized SR(r1) − SR(r2) and its Newey–West delta-method standard error for batches (B, T)."""
    r1, r2 = np.atleast_2d(r1), np.atleast_2d(r2)
    moments = np.stack([r1, r1 ** 2, r2, r2 ** 2], axis=2)  # (B, T, 4)
    mean = moments.mean(axis=1)  # mu1, g1, mu2, g2
    mu1, g1, mu2, g2 = mean.T
    var1, var2 = np.maximum(g1 - mu1 ** 2, 1e-300), np.maximum(g2 - mu2 ** 2, 1e-300)
    delta = mu1 / np.sqrt(var1) - mu2 / np.sqrt(var2)
    gradient = np.stack([g1 / var1 ** 1.5, -mu1 / (2 * var1 ** 1.5), -g2 / var2 ** 1.5, mu2 / (2 * var2 ** 1.5)], axis=1)
    centered = moments - mean[:, None, :]
    T = centered.shape[1]
    psi = np.einsum("bti,btj->bij", centered, centered) / T
    for lag in range(1, bandwidth + 1):
        gamma = np.einsum("bti,btj->bij", centered[:, lag:], centered[:, :-lag]) / T
        psi += (1 - lag / (bandwidth + 1)) * (gamma + np.swapaxes(gamma, 1, 2))
    variance = np.einsum("bi,bij,bj->b", gradient, psi, gradient) / T
    scale = np.sqrt(TRADING_DAYS)
    return delta * scale, np.sqrt(np.maximum(variance, 0.0)) * scale


def ledoit_wolf_pvalue(r1, r2, *, n_boot=5000, seed=0, min_length=30, batch=500):
    """One-sided studentized circular block bootstrap of Ledoit & Wolf (2008); H1: SR(r1) > SR(r2).

    Block length and Newey–West bandwidth are floor(T^(1/3)); p = share of bootstrap statistics
    t*_b = (Δ*_b − Δ̂)/SE*_b at or above t_obs = Δ̂/SE. Series shorter than min_length give p = nan.
    Identical series have SE = 0 everywhere: t_obs = 0 and p = 1 (no evidence either way).
    """
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    n = len(r1)
    if n != len(r2):
        raise ValueError("the two return series must have the same length")
    block = max(1, int(math.floor(n ** (1 / 3))))
    if n < min_length:
        return {"delta": math.nan, "t_obs": math.nan, "p": math.nan, "block_length": block, "n": n}
    delta, se = _sharpe_differences(r1, r2, block)
    delta, se = float(delta[0]), float(se[0])
    t_obs = delta / se if se > 0 else 0.0
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    offsets = np.arange(block)
    t_star = np.empty(n_boot)
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot - start)
        starts = rng.integers(0, n, size=(size, n_blocks))
        index = ((starts[:, :, None] + offsets) % n).reshape(size, -1)[:, :n]
        delta_b, se_b = _sharpe_differences(r1[index], r2[index], block)
        t_star[start:start + size] = np.where(se_b > 0, (delta_b - delta) / np.where(se_b > 0, se_b, 1.0), 0.0)
    return {"delta": delta, "t_obs": t_obs, "p": float(np.mean(t_star >= t_obs)), "block_length": block, "n": n}
