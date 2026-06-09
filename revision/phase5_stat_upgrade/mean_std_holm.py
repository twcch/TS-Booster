"""
Phase 5a: Mean ± Std Tables + Holm-Bonferroni Correction

From existing 5-seed JSON data (no re-run needed):
  1. Main experiment: Vanilla Linear vs TSB (16 datasets × 5 seeds)
  2. SOTA DL baselines: DLinear / PatchTST / iTransformer vs TSB (16 × 5 seeds)

Outputs:
  - Console: mean±std tables with ** / * significance markers
  - mean_std_main.json      -- vanilla vs TSB per dataset
  - mean_std_sota.json      -- all SOTA DL models per dataset
  - holm_correction.json    -- raw p-values + corrected decisions

Statistical approach:
  - Per-dataset: one-sided paired t-test (H1: TSB RMSE < baseline) across 5 seeds
  - Multiple comparison correction: Holm-Bonferroni (α=0.05, m=16 tests per comparison)
  - This tests whether the seed-average improvement is reliably non-zero across seeds
    (Note: DM test in dm_test_runner.py tests forecast error significance on test samples)

Usage:
  uv run python revision/phase5_stat_upgrade/mean_std_holm.py
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.append(str(Path(__file__).parent.parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
MAIN_PATH = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
OUT_DIR   = Path("revision/phase5_stat_upgrade")

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]


# ── Holm-Bonferroni ────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Return list of reject decisions after Holm-Bonferroni correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    rejected = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if p_values[idx] <= threshold:
            rejected[idx] = True
        else:
            break  # stop at first non-rejection
    return rejected


def paired_ttest_onesided(a: list[float], b: list[float]) -> float:
    """H1: a < b (a is better / lower RMSE). Returns p-value."""
    diff = np.array(b) - np.array(a)   # positive = a is better
    t, p_two = stats.ttest_1samp(diff, popmean=0)
    return float(p_two / 2 if t > 0 else 1 - p_two / 2)


# ── Formatters ────────────────────────────────────────────────────────────────

def fmt(mean: float, std: float, sig: bool, alpha05: bool) -> str:
    marker = "**" if sig else ("*" if alpha05 else "  ")
    return f"{mean:.4f}±{std:.4f}{marker}"


# ── Main experiment: Vanilla vs TSB ───────────────────────────────────────────

def analyze_main(data: dict) -> dict:
    print("\n" + "=" * 80)
    print("Main Experiment: Vanilla Linear vs TS-Booster")
    print("(mean ± std across 5 seeds, one-sided paired t-test, Holm-Bonferroni)")
    print("=" * 80)
    print(f"{'Dataset':<14} {'Vanilla RMSE':>20} {'TSB RMSE':>20} {'Improve%':>10} {'p-raw':>8}")
    print("-" * 80)

    results = {}
    p_values = []

    for ds in ALL_DATASETS:
        if ds not in data:
            continue
        van  = np.array(data[ds]["vanilla"]["rmse"])
        tsb  = np.array(data[ds]["booster"]["rmse"])
        pval = paired_ttest_onesided(tsb.tolist(), van.tolist())
        impr = float((van.mean() - tsb.mean()) / van.mean() * 100)
        results[ds] = {
            "vanilla_mean": float(van.mean()), "vanilla_std": float(van.std()),
            "tsb_mean":     float(tsb.mean()), "tsb_std":     float(tsb.std()),
            "improvement_pct": impr, "p_raw": pval,
        }
        p_values.append(pval)
        print(f"{ds:<14} {van.mean():>8.4f}±{van.std():.4f}   {tsb.mean():>8.4f}±{tsb.std():.4f}   {impr:>+8.2f}%  {pval:.4f}")

    # Holm-Bonferroni
    datasets_present = [ds for ds in ALL_DATASETS if ds in data]
    rejected = holm_bonferroni(p_values)
    print("\nHolm-Bonferroni correction (α=0.05):")
    for ds, pval, rej in zip(datasets_present, p_values, rejected):
        results[ds]["holm_rejected"] = rej
        marker = "REJECT H0 ✓" if rej else "n.s."
        print(f"  {ds:<14}  p={pval:.4f}  {marker}")

    n_sig = sum(rejected)
    print(f"\n  Significant after correction: {n_sig}/{len(rejected)}")
    avg_imp = np.mean([results[ds]["improvement_pct"] for ds in datasets_present])
    print(f"  Average improvement: {avg_imp:+.2f}%")

    return results


# ── SOTA comparison ───────────────────────────────────────────────────────────

def analyze_sota(data: dict) -> dict:
    BASELINES = ["DLinear", "iTransformer", "PatchTST"]

    print("\n" + "=" * 80)
    print("SOTA DL Baselines vs TS-Booster")
    print("(mean ± std across 5 seeds, one-sided paired t-test H1: TSB < baseline)")
    print("=" * 80)

    all_results = {}

    for baseline in BASELINES:
        print(f"\n── {baseline} ──────────────────────────────────────────────")
        print(f"{'Dataset':<14} {'Baseline RMSE':>22} {'TSB RMSE':>20} {'Improve%':>10} {'p-raw':>8}")
        print("-" * 78)

        p_values = []
        results_bl = {}

        for ds in ALL_DATASETS:
            if ds not in data or baseline not in data[ds]:
                continue
            bl_arr  = np.array(data[ds][baseline]["rmse"])
            tsb_arr = np.array(data[ds]["Linear+TSB"]["rmse"])
            pval    = paired_ttest_onesided(tsb_arr.tolist(), bl_arr.tolist())
            impr    = float((bl_arr.mean() - tsb_arr.mean()) / bl_arr.mean() * 100)
            results_bl[ds] = {
                "baseline_mean": float(bl_arr.mean()), "baseline_std": float(bl_arr.std()),
                "tsb_mean":      float(tsb_arr.mean()), "tsb_std":      float(tsb_arr.std()),
                "improvement_pct": impr, "p_raw": pval,
            }
            p_values.append(pval)
            print(f"{ds:<14} {bl_arr.mean():>9.4f}±{bl_arr.std():.4f}   {tsb_arr.mean():>8.4f}±{tsb_arr.std():.4f}  {impr:>+8.2f}%  {pval:.4f}")

        datasets_present = [ds for ds in ALL_DATASETS if ds in data and baseline in data[ds]]
        rejected = holm_bonferroni(p_values)
        n_sig = sum(rejected)
        avg_imp = np.mean([results_bl[ds]["improvement_pct"] for ds in datasets_present])
        print(f"\n  Holm sig: {n_sig}/{len(rejected)},  avg improvement: {avg_imp:+.2f}%")

        for ds, pval, rej in zip(datasets_present, p_values, rejected):
            results_bl[ds]["holm_rejected"] = rej

        all_results[baseline] = results_bl

    return all_results


# ── LaTeX table helpers ────────────────────────────────────────────────────────

def print_latex_main(main_results: dict):
    print("\n" + "=" * 80)
    print("LaTeX table (Vanilla vs TSB, RMSE mean±std):")
    print("=" * 80)
    print(r"\begin{tabular}{lcccc}")
    print(r"\hline")
    print(r"Dataset & Vanilla RMSE & TSB RMSE & Improve\% & Sig. \\")
    print(r"\hline")
    for ds in ALL_DATASETS:
        if ds not in main_results:
            continue
        r = main_results[ds]
        sig = r"$^{**}$" if r["holm_rejected"] else ""
        print(f"{ds.replace('_', chr(92)+'_')} & "
              f"{r['vanilla_mean']:.4f}$\\pm${r['vanilla_std']:.4f} & "
              f"{r['tsb_mean']:.4f}$\\pm${r['tsb_std']:.4f} & "
              f"{r['improvement_pct']:+.2f}\\% & "
              f"{sig} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    main_data = json.loads(MAIN_PATH.read_text())
    sota_data = json.loads(SOTA_PATH.read_text())

    main_results = analyze_main(main_data)
    sota_results = analyze_sota(sota_data)

    print_latex_main(main_results)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mean_std_main.json").write_text(json.dumps(main_results, indent=2))
    (OUT_DIR / "mean_std_sota.json").write_text(json.dumps(sota_results, indent=2))
    print(f"\nSaved to {OUT_DIR}/mean_std_main.json and mean_std_sota.json")


if __name__ == "__main__":
    main()
