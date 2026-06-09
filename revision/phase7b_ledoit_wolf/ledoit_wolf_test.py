"""
Phase 7b: Ledoit-Wolf (2008) Sharpe Ratio Significance Test

The plain percentile circular-block bootstrap used in phase7_backtest_ext gave
0/16 significant results (p ≈ 0.5). Ledoit & Wolf (2008, "Robust Performance
Hypothesis Testing with the Sharpe Ratio") show that a *studentized* circular
block bootstrap has much better finite-sample size/power than the plain
percentile bootstrap for Sharpe ratio comparisons under serial correlation and
heteroskedasticity (both present in daily financial returns).

Method (Ledoit-Wolf 2008, studentized circular block bootstrap):
  1. Observed statistic:
       Δ̂ = SR(Linear+TSB) - SR(baseline)
       SE(Δ̂) estimated via the delta method using a Newey-West HAC long-run
       covariance of the moment vector v_t = (r1_t, r1_t^2, r2_t, r2_t^2)
       (gradient of SR_i = μ_i / sqrt(E[r_i^2] - μ_i^2) w.r.t. (μ_i, E[r_i^2]))
       t_obs = Δ̂ / SE(Δ̂)
  2. Resample (r1, r2) jointly via circular block bootstrap (block length
     L = floor(T^(1/3)), matching phase7's choice for comparability).
  3. For each bootstrap sample b, recompute Δ*_b, SE*_b, and the studentized
     statistic  t*_b = (Δ*_b - Δ̂) / SE*_b.
  4. One-sided p-value for H1: SR(TSB) > SR(baseline):
       p = (1/B) * #{b : t*_b >= t_obs}
     Under H0: Δ=0, t_obs is pivotal and distributed like t*_b
     (Hall & Wilson 1991 studentized bootstrap principle); a small p means
     t_obs is unusually large relative to the bootstrap reference distribution
     — i.e. evidence for H1.

Loads from existing backtest_daily_returns.json. No re-run needed.

Usage:
  uv run python revision/phase7b_ledoit_wolf/ledoit_wolf_test.py
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

DAILY_RETURNS_PATH = Path("experiments/comprehensive_results/backtest/backtest_daily_returns.json")
OUT_DIR  = Path("revision/phase7b_ledoit_wolf")
OUT_PATH = OUT_DIR / "ledoit_wolf_results.json"

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]
BASELINES = ["Historical Vol", "XGBoost", "LightGBM", "Linear"]
TARGET    = "Linear+TSB"

ANN   = 252
N_BOOT = 5000
SEED   = 42


# ── Sharpe ratio + delta-method SE via Newey-West HAC ─────────────────────────

def sharpe(r, ann=ANN):
    r = np.asarray(r)
    mu, sigma = r.mean(), r.std(ddof=1)
    return float(mu / sigma * np.sqrt(ann)) if sigma > 0 else np.nan


def _moment_vector(r1, r2):
    """v_t = (r1_t, r1_t^2, r2_t, r2_t^2)"""
    return np.column_stack([r1, r1 ** 2, r2, r2 ** 2])


def _gradient(r1, r2):
    """Gradient of Δ = SR1 - SR2 (in *daily*, un-annualized units) w.r.t.
    θ = (μ1, m1, μ2, m2), where m_i = E[r_i^2], σ_i^2 = m_i - μ_i^2.
    SR_i = μ_i / σ_i  ⇒  ∂SR_i/∂μ_i = m_i/σ_i^3,  ∂SR_i/∂m_i = -μ_i/(2σ_i^3)
    """
    mu1, mu2 = r1.mean(), r2.mean()
    m1, m2 = (r1 ** 2).mean(), (r2 ** 2).mean()
    s1 = np.sqrt(max(m1 - mu1 ** 2, 1e-12))
    s2 = np.sqrt(max(m2 - mu2 ** 2, 1e-12))
    return np.array([
        m1 / s1 ** 3,
        -mu1 / (2 * s1 ** 3),
        -m2 / s2 ** 3,
        mu2 / (2 * s2 ** 3),
    ])


def _newey_west_long_run_cov(v, L):
    """Newey-West HAC long-run covariance of the moment vector v (T x 4),
    Bartlett kernel, bandwidth L."""
    T = len(v)
    vc = v - v.mean(axis=0)
    gamma0 = (vc.T @ vc) / T
    psi = gamma0.copy()
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1)
        gamma_j = (vc[j:].T @ vc[:-j]) / T
        psi += w * (gamma_j + gamma_j.T)
    return psi


def sharpe_diff_and_se(r1, r2, ann=ANN):
    """Δ̂ (annualized) and its delta-method / HAC standard error."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    T = len(r1)
    L = max(1, int(T ** (1 / 3)))

    delta_daily = (r1.mean() / r1.std(ddof=1)) - (r2.mean() / r2.std(ddof=1))
    delta = delta_daily * np.sqrt(ann)

    grad = _gradient(r1, r2)
    v = _moment_vector(r1, r2)
    psi = _newey_west_long_run_cov(v, L)
    var_daily = float(grad @ psi @ grad) / T
    se = np.sqrt(max(var_daily, 0.0)) * np.sqrt(ann)
    return float(delta), float(se)


# ── Studentized circular block bootstrap (Ledoit-Wolf 2008) ───────────────────

def circular_block_indices(T, L, rng):
    starts = rng.integers(0, T, size=int(np.ceil(T / L)))
    idx = np.concatenate([np.arange(s, s + L) % T for s in starts])[:T]
    return idx


def ledoit_wolf_pvalue(r1, r2, n_boot=N_BOOT, ann=ANN, seed=SEED):
    """One-sided studentized circular block bootstrap p-value for
    H1: Sharpe(r1) > Sharpe(r2)  (r1 = TSB, r2 = baseline)."""
    r1, r2 = np.asarray(r1), np.asarray(r2)
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]
    L = max(1, int(n ** (1 / 3)))

    delta_obs, se_obs = sharpe_diff_and_se(r1, r2, ann=ann)
    if se_obs == 0 or np.isnan(se_obs):
        return {"delta": delta_obs, "se": se_obs, "t_obs": np.nan, "p_value": np.nan}
    t_obs = delta_obs / se_obs

    rng = np.random.default_rng(seed)
    t_boot = []
    for _ in range(n_boot):
        idx = circular_block_indices(n, L, rng)
        b1, b2 = r1[idx], r2[idx]
        if b1.std(ddof=1) == 0 or b2.std(ddof=1) == 0:
            continue
        d_b, se_b = sharpe_diff_and_se(b1, b2, ann=ann)
        if se_b == 0 or np.isnan(se_b):
            continue
        t_boot.append((d_b - delta_obs) / se_b)

    t_boot = np.array(t_boot)
    # Studentized bootstrap one-sided p-value for H1: Δ > 0 (TSB better).
    # Under H0: Δ=0, t_obs = Δ̂/SE(Δ̂) is pivotal and distributed like
    # t*_b = (Δ*_b - Δ̂)/SE*_b (Hall & Wilson 1991 / Ledoit-Wolf 2008).
    # p = P(t* >= t_obs): fraction of bootstrap draws at least as large as
    # observed — small p means t_obs is unusually large (evidence for H1).
    p = float(np.mean(t_boot >= t_obs))
    return {"delta": delta_obs, "se": se_obs, "t_obs": float(t_obs), "p_value": p}


# ── Driver ─────────────────────────────────────────────────────────────────────

def run_all():
    daily = json.loads(DAILY_RETURNS_PATH.read_text())
    results = {}

    for ds in ALL_DATASETS:
        if ds not in daily:
            continue
        r_tsb = np.array(daily[ds][TARGET])
        results[ds] = {}
        print(f"\n{ds}")
        for base in BASELINES:
            r_base = np.array(daily[ds][base])
            res = ledoit_wolf_pvalue(r_tsb, r_base)
            results[ds][base] = res
            sig = "*" if res["p_value"] < 0.05 else " "
            print(f"  vs {base:<15} Δ={res['delta']:+.3f}  SE={res['se']:.3f}  "
                  f"t={res['t_obs']:+.2f}  p={res['p_value']:.3f} {sig}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT_PATH}")
    return results


def analyze(results):
    print("\n" + "=" * 70)
    print("Ledoit-Wolf studentized bootstrap: significant wins (p < 0.05)")
    print("=" * 70)
    for base in BASELINES:
        sig = sum(1 for ds in results if results[ds][base]["p_value"] < 0.05)
        n = len(results)
        avg_p = np.mean([results[ds][base]["p_value"] for ds in results
                         if not np.isnan(results[ds][base]["p_value"])])
        print(f"  vs {base:<15} {sig}/{n} significant   avg p = {avg_p:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if args.analyze:
        if not OUT_PATH.exists():
            print(f"No results at {OUT_PATH}. Run without --analyze first.")
        else:
            analyze(json.loads(OUT_PATH.read_text()))
    else:
        analyze(run_all())
