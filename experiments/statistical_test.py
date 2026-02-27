"""
Statistical Significance Testing for TS-Booster Paper.

Compares Vanilla backbone vs TS-Booster across 16 datasets × 5 seeds.

Per-dataset tests (n=5 seeds):
  - Paired t-test (one-sided: H1: TS-Booster < Vanilla RMSE)
  - 95% CI for mean RMSE improvement (%)
  - Cohen's d effect size

Overall tests (n=16 datasets):
  - Wilcoxon signed-rank test on dataset-level mean gaps
  - Binomial sign test (wins under H0: p=0.5)

Usage:
  uv run python experiments/statistical_test.py              # Run all
  uv run python experiments/statistical_test.py --analyze    # Analyze saved results only
"""
import sys, json, random, argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

SEEDS = [42, 123, 456, 789, 2024]

RESULTS_PATH = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
FIGURES_DIR = Path("experiments/comprehensive_results/main_experiment/figures")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, test_loader, scaler, device):
    """Return dict with rmse, mae, mda."""
    model.eval()
    preds, targets, last_obs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)
            output = model(x)
            pred = output[0] if isinstance(output, tuple) else output
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            last_obs_list.append(x[:, -1, 0].cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    last_obs = np.concatenate(last_obs_list)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    if targets.ndim == 3:
        targets = targets.squeeze(-1)
    if scaler:
        preds = preds * scaler.scale_[0] + scaler.mean_[0]
        targets = targets * scaler.scale_[0] + scaler.mean_[0]
        last_obs = last_obs * scaler.scale_[0] + scaler.mean_[0]
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(), preds.flatten())))
    mae = float(np.mean(np.abs(targets.flatten() - preds.flatten())))
    # MDA: direction accuracy relative to last observed value
    pred_dir = np.sign(preds - last_obs[:, None])
    actual_dir = np.sign(targets - last_obs[:, None])
    mda = float(np.mean(pred_dir == actual_dir))
    return {"rmse": rmse, "mae": mae, "mda": mda}


def run_pair(ds, seed, epochs=100):
    """Run Vanilla and TS-Booster for one dataset/seed pair. Returns (vanilla_rmse, booster_rmse)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = BEST_STAGE3_CONFIG.copy()
    config['epochs'] = epochs
    config['input_channels'] = 2
    seq_len, pred_len = 30, 3

    path = f"dataset/{ds}.csv"
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=seq_len, horizon=pred_len, batch_size=32
    )

    base_kwargs = dict(
        seq_len=seq_len, pred_len=pred_len, input_channels=2,
        decomp_kernels=[7, 13],
        expert_kernels=config.get('expert_kernels', [3, 5, 7]),
        expert_hidden_dim=config.get('expert_hidden_dim', 32),
        dropout_p=config.get('dropout', 0.3),
        n_boost_rounds=config.get('n_boost_rounds', 2),
        expert_type=config.get('expert_type', 'cnn'),
    )

    # Vanilla backbone (no decomp, no expert)
    set_seed(seed)
    vanilla = FlexibleTSBooster(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        use_decomp=False, use_experts=False, **base_kwargs
    )
    vanilla = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=vanilla
    )
    vanilla_metrics = evaluate(vanilla, test_loader, scaler, device)

    # TS-Booster (full model)
    set_seed(seed)
    booster = FlexibleTSBooster(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        use_decomp=True, use_experts=True,
        expert_gate='snr', enrich_mode='trend',
        **base_kwargs
    )
    booster = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=booster
    )
    booster_metrics = evaluate(booster, test_loader, scaler, device)

    return vanilla_metrics, booster_metrics


def run_experiments():
    """Run all experiments and save raw RMSE values."""
    results = {}

    for ds in ALL_DATASETS:
        results[ds] = {
            "vanilla": {"rmse": [], "mae": [], "mda": []},
            "booster": {"rmse": [], "mae": [], "mda": []},
        }
        for seed in SEEDS:
            print(f"  {ds:12s} | Seed {seed} ... ", end="", flush=True)
            pair = run_pair(ds, seed)
            if pair is None:
                print("SKIP (file not found)")
                continue
            v_m, b_m = pair
            for key in ("rmse", "mae", "mda"):
                results[ds]["vanilla"][key].append(v_m[key])
                results[ds]["booster"][key].append(b_m[key])
            gap = (v_m["rmse"] - b_m["rmse"]) / v_m["rmse"] * 100
            tag = "WIN" if gap > 0 else "LOSS"
            print(f"RMSE: V={v_m['rmse']:.4f} B={b_m['rmse']:.4f} Gap={gap:+.2f}% {tag}")

        # Save incrementally per dataset
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")
    return results


def _get_metric(results, ds, model, metric):
    """Get metric array, handling both old (flat list) and new (dict) JSON formats."""
    data = results[ds][model]
    if isinstance(data, list):
        return np.array(data) if metric == "rmse" else None
    return np.array(data[metric])


def analyze(results):
    """Compute statistical tests and print publication-ready table."""

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE ANALYSIS: Vanilla Linear vs TS-Booster")
    print("=" * 100)

    # Detect format
    sample = results[ALL_DATASETS[0]]["vanilla"]
    has_extra = isinstance(sample, dict)

    # === Per-dataset RMSE analysis ===
    header = (
        f"{'Dataset':<12} | {'Vanilla RMSE':>13} | {'Booster RMSE':>13} | "
        f"{'Gap%':>8} | {'95% CI':>16} | {'Cohen d':>8} | {'t-stat':>7} | {'p-value':>8} | {'Sig':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    dataset_gaps = []  # mean gap per dataset for overall test
    all_vanilla = []
    all_booster = []
    sig_count = 0
    win_count = 0

    for ds in ALL_DATASETS:
        v = _get_metric(results, ds, "vanilla", "rmse")
        b = _get_metric(results, ds, "booster", "rmse")
        if v is None or len(v) == 0:
            continue

        all_vanilla.extend(v.tolist())
        all_booster.extend(b.tolist())

        # Gap: positive means Booster is better
        gaps = (v - b) / v * 100
        mean_gap = np.mean(gaps)
        dataset_gaps.append(mean_gap)
        is_win = mean_gap > 0
        if is_win:
            win_count += 1

        mean_v = np.mean(v)
        mean_b = np.mean(b)

        # Paired t-test (one-sided: H1: booster RMSE < vanilla RMSE)
        t_stat, p_two = stats.ttest_rel(v, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

        # 95% CI for mean gap
        se = stats.sem(gaps)
        ci_low = mean_gap - 1.96 * se
        ci_high = mean_gap + 1.96 * se

        # Cohen's d (paired)
        diff = v - b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

        # Significance markers
        if p_one < 0.01:
            sig = "**"
            sig_count += 1
        elif p_one < 0.05:
            sig = "*"
            sig_count += 1
        elif p_one < 0.10:
            sig = "+"
        else:
            sig = ""

        ci_str = f"[{ci_low:+.2f}, {ci_high:+.2f}]"
        print(
            f"  {ds:<12} | {mean_v:>13.6f} | {mean_b:>13.6f} | "
            f"{mean_gap:>+7.2f}% | {ci_str:>16} | {d:>8.2f} | {t_stat:>7.2f} | {p_one:>8.4f} | {sig:>5}"
        )

    print("-" * len(header))
    print(f"  Significance: ** p<0.01, * p<0.05, + p<0.10 (one-sided paired t-test)")
    print(f"  Significant datasets (p<0.05): {sig_count}/{len(ALL_DATASETS)}")

    # === MAE & MDA summary ===
    if has_extra:
        print(f"\n{'=' * 100}")
        print("ADDITIONAL METRICS: MAE & MDA")
        print("=" * 100)
        hdr2 = f"{'Dataset':<12} | {'V_MAE':>8} | {'B_MAE':>8} | {'MAE Gap%':>9} | {'V_MDA':>7} | {'B_MDA':>7} | {'MDA Diff':>9}"
        print(f"\n{hdr2}")
        print("-" * len(hdr2))

        mae_wins = 0
        mda_wins = 0
        all_mae_gaps = []
        all_v_mda = []
        all_b_mda = []

        for ds in ALL_DATASETS:
            v_mae = _get_metric(results, ds, "vanilla", "mae")
            b_mae = _get_metric(results, ds, "booster", "mae")
            v_mda = _get_metric(results, ds, "vanilla", "mda")
            b_mda = _get_metric(results, ds, "booster", "mda")
            if v_mae is None or len(v_mae) == 0:
                continue

            mv = np.mean(v_mae)
            mb = np.mean(b_mae)
            mae_gap = (mv - mb) / mv * 100
            all_mae_gaps.append(mae_gap)
            if mae_gap > 0:
                mae_wins += 1

            mvd = np.mean(v_mda) * 100
            mbd = np.mean(b_mda) * 100
            mda_diff = mbd - mvd
            all_v_mda.append(mvd)
            all_b_mda.append(mbd)
            if mda_diff > 0:
                mda_wins += 1

            print(f"  {ds:<12} | {mv:>8.4f} | {mb:>8.4f} | {mae_gap:>+8.2f}% | {mvd:>6.1f}% | {mbd:>6.1f}% | {mda_diff:>+8.2f}%")

        n_ds = len(all_mae_gaps)
        print("-" * len(hdr2))
        print(f"  MAE: Booster wins {mae_wins}/{n_ds}, avg gap: {np.mean(all_mae_gaps):+.2f}%")
        print(f"  MDA: Booster wins {mda_wins}/{n_ds}, avg Vanilla={np.mean(all_v_mda):.1f}%, Booster={np.mean(all_b_mda):.1f}%")

    # === Overall tests ===
    print(f"\n{'=' * 80}")
    print("OVERALL TESTS (across all 16 datasets)")
    print("=" * 80)

    dataset_gaps = np.array(dataset_gaps)
    n_ds = len(dataset_gaps)

    # 1. Wilcoxon signed-rank test on dataset-level mean gaps (n=16)
    w_stat, w_p = stats.wilcoxon(dataset_gaps, alternative='greater')
    print(f"\n  1. Wilcoxon Signed-Rank Test (H1: mean gap > 0)")
    print(f"     W-statistic = {w_stat:.1f}, p-value = {w_p:.6f}", end="")
    print(f"  {'***' if w_p < 0.001 else '**' if w_p < 0.01 else '*' if w_p < 0.05 else 'n.s.'}")

    # 2. Binomial sign test
    binom_p = stats.binomtest(win_count, n_ds, 0.5, alternative='greater').pvalue
    print(f"\n  2. Binomial Sign Test (H0: P(win)=0.5)")
    print(f"     Wins = {win_count}/{n_ds}, p-value = {binom_p:.6f}", end="")
    print(f"  {'***' if binom_p < 0.001 else '**' if binom_p < 0.01 else '*' if binom_p < 0.05 else 'n.s.'}")

    # 3. One-sample t-test on mean gaps
    t_overall, p_overall_two = stats.ttest_1samp(dataset_gaps, 0)
    p_overall = p_overall_two / 2 if t_overall > 0 else 1 - p_overall_two / 2
    print(f"\n  3. One-Sample t-test on Mean Gaps (H1: mean > 0)")
    print(f"     t = {t_overall:.3f}, p-value = {p_overall:.6f}", end="")
    print(f"  {'***' if p_overall < 0.001 else '**' if p_overall < 0.01 else '*' if p_overall < 0.05 else 'n.s.'}")

    # 4. Summary statistics
    overall_mean = np.mean(dataset_gaps)
    overall_se = stats.sem(dataset_gaps)
    overall_ci_low = overall_mean - 1.96 * overall_se
    overall_ci_high = overall_mean + 1.96 * overall_se
    print(f"\n  4. Summary")
    print(f"     Mean improvement: {overall_mean:+.2f}% (95% CI: [{overall_ci_low:+.2f}%, {overall_ci_high:+.2f}%])")
    print(f"     Win rate: {win_count}/{n_ds} = {win_count/n_ds*100:.1f}%")
    print(f"     Median gap: {np.median(dataset_gaps):+.2f}%")

    # === Category breakdown ===
    print(f"\n{'=' * 80}")
    print("CATEGORY BREAKDOWN")
    print("=" * 80)

    categories = {
        "Taiwan Stocks": ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"],
        "TWD FX": ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD"],
        "International FX": ["EUR_USD", "GBP_USD", "USD_JPY"],
    }

    for cat_name, cat_datasets in categories.items():
        cat_gaps = []
        cat_wins = 0
        for ds in cat_datasets:
            v = _get_metric(results, ds, "vanilla", "rmse")
            b = _get_metric(results, ds, "booster", "rmse")
            if v is None or len(v) == 0:
                continue
            g = np.mean((v - b) / v * 100)
            cat_gaps.append(g)
            if g > 0:
                cat_wins += 1

        cat_gaps = np.array(cat_gaps)
        n_cat = len(cat_gaps)
        if n_cat >= 2:
            t_cat, p_cat_two = stats.ttest_1samp(cat_gaps, 0)
            p_cat = p_cat_two / 2 if t_cat > 0 else 1 - p_cat_two / 2
        else:
            p_cat = float('nan')

        print(f"\n  {cat_name} ({cat_wins}/{n_cat} wins):")
        print(f"    Mean gap: {np.mean(cat_gaps):+.2f}%, Median: {np.median(cat_gaps):+.2f}%")
        if n_cat >= 2:
            print(f"    One-sample t-test p-value: {p_cat:.4f}", end="")
            print(f"  {'***' if p_cat < 0.001 else '**' if p_cat < 0.01 else '*' if p_cat < 0.05 else 'n.s.'}")

    # === LaTeX table ===
    print(f"\n{'=' * 80}")
    print("LATEX TABLE (for paper)")
    print("=" * 80)
    print()
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Statistical significance of TS-Booster improvement over Vanilla Linear backbone (5 seeds).}")
    print(r"\label{tab:significance}")
    if has_extra:
        print(r"\begin{tabular}{lrrrrrrrrl}")
        print(r"\toprule")
        print(r"Dataset & V\_RMSE & B\_RMSE & RMSE $\Delta$(\%) & V\_MAE & B\_MAE & MAE $\Delta$(\%) & B\_MDA(\%) & $p$-value & Sig. \\")
    else:
        print(r"\begin{tabular}{lrrrrrl}")
        print(r"\toprule")
        print(r"Dataset & Vanilla RMSE & Booster RMSE & Gap (\%) & Cohen's $d$ & $p$-value & Sig. \\")
    print(r"\midrule")

    for ds in ALL_DATASETS:
        v = _get_metric(results, ds, "vanilla", "rmse")
        b = _get_metric(results, ds, "booster", "rmse")
        if v is None or len(v) == 0:
            continue
        gaps = (v - b) / v * 100
        mean_gap = np.mean(gaps)
        mean_v = np.mean(v)
        mean_b = np.mean(b)
        diff = v - b
        d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        t_stat, p_two = stats.ttest_rel(v, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

        if p_one < 0.01:
            sig = r"$^{**}$"
        elif p_one < 0.05:
            sig = r"$^{*}$"
        elif p_one < 0.10:
            sig = r"$^{+}$"
        else:
            sig = ""

        ds_latex = ds.replace("_", r"\_")
        if has_extra:
            v_mae = _get_metric(results, ds, "vanilla", "mae")
            b_mae = _get_metric(results, ds, "booster", "mae")
            b_mda = _get_metric(results, ds, "booster", "mda")
            mv_mae = np.mean(v_mae)
            mb_mae = np.mean(b_mae)
            mae_gap = (mv_mae - mb_mae) / mv_mae * 100
            mda_pct = np.mean(b_mda) * 100
            print(f"{ds_latex} & {mean_v:.4f} & {mean_b:.4f} & {mean_gap:+.2f} & {mv_mae:.4f} & {mb_mae:.4f} & {mae_gap:+.2f} & {mda_pct:.1f} & {p_one:.4f} & {sig} \\\\")
        else:
            print(f"{ds_latex} & {mean_v:.4f} & {mean_b:.4f} & {mean_gap:+.2f} & {d:.2f} & {p_one:.4f} & {sig} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ──────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────

FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
               "EUR_USD", "GBP_USD", "USD_JPY"]


def plot_improvement_bars(results):
    """Bar chart: RMSE improvement % per dataset with 5-seed error bars and p-value stars."""
    datasets, means, stds, sigs = [], [], [], []
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        v = _get_metric(results, ds, "vanilla", "rmse")
        b = _get_metric(results, ds, "booster", "rmse")
        if v is None or len(v) == 0:
            continue
        datasets.append(ds)
        gaps = (v - b) / v * 100
        means.append(np.mean(gaps))
        stds.append(np.std(gaps, ddof=1))
        # p-value for significance marker
        t_stat, p_two = stats.ttest_rel(v, b)
        p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        sigs.append('**' if p_one < 0.01 else '*' if p_one < 0.05 else '')

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    colors = ['#43A047' if m > 0 else '#E53935' for m in means]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.7,
                  error_kw={'linewidth': 1, 'color': '#424242'})

    # Add significance stars and value labels
    for i, (bar, mean, std, sig) in enumerate(zip(bars, means, stds, sigs)):
        h = bar.get_height()
        # Value label
        ax.text(bar.get_x() + bar.get_width() / 2, h + std + 0.5,
                f'{mean:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        # Significance star
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2, h + std + 2.0,
                    sig, ha='center', va='bottom', fontsize=9, color='#1565C0')

    ax.axhline(y=0, color='#9E9E9E', linewidth=0.8, linestyle='-')

    # FX / Stocks separator
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axvline(x=n_fx - 0.5, color='#424242', linewidth=1.5, linestyle='--', alpha=0.5)
        ax.text(n_fx / 2 - 0.5, ax.get_ylim()[1] * 0.92, 'FX',
                ha='center', fontsize=10, fontweight='bold', color='#1565C0')
        ax.text(n_fx + (len(datasets) - n_fx) / 2 - 0.5, ax.get_ylim()[1] * 0.92, 'Stocks',
                ha='center', fontsize=10, fontweight='bold', color='#C62828')

    # Overall summary
    overall_mean = np.mean(means)
    ax.axhline(y=overall_mean, color='#1565C0', linewidth=1, linestyle=':', alpha=0.7)
    ax.text(len(datasets) - 0.5, overall_mean + 0.5,
            f'avg: {overall_mean:+.1f}%', ha='right', fontsize=9, color='#1565C0', fontstyle='italic')

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax.set_title('TS-Booster vs Vanilla Linear: RMSE Improvement (5 seeds, error bars = 1 std)\n'
                 '** p < 0.01 (paired t-test)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'rmse_improvement.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'rmse_improvement.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: rmse_improvement.png/pdf")


def plot_mae_improvement(results):
    """Bar chart: MAE improvement % per dataset with 5-seed error bars."""
    datasets, means, stds = [], [], []
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        v = _get_metric(results, ds, "vanilla", "mae")
        b = _get_metric(results, ds, "booster", "mae")
        if v is None or len(v) == 0:
            continue
        datasets.append(ds)
        gaps = (v - b) / v * 100
        means.append(np.mean(gaps))
        stds.append(np.std(gaps, ddof=1))

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    colors = ['#43A047' if m > 0 else '#E53935' for m in means]

    bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.7,
                  error_kw={'linewidth': 1, 'color': '#424242'})

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + std + 0.5,
                f'{mean:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(y=0, color='#9E9E9E', linewidth=0.8, linestyle='-')

    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axvline(x=n_fx - 0.5, color='#424242', linewidth=1.5, linestyle='--', alpha=0.5)
        ax.text(n_fx / 2 - 0.5, ax.get_ylim()[1] * 0.92, 'FX',
                ha='center', fontsize=10, fontweight='bold', color='#1565C0')
        ax.text(n_fx + (len(datasets) - n_fx) / 2 - 0.5, ax.get_ylim()[1] * 0.92, 'Stocks',
                ha='center', fontsize=10, fontweight='bold', color='#C62828')

    overall_mean = np.mean(means)
    ax.axhline(y=overall_mean, color='#1565C0', linewidth=1, linestyle=':', alpha=0.7)
    ax.text(len(datasets) - 0.5, overall_mean + 0.5,
            f'avg: {overall_mean:+.1f}%', ha='right', fontsize=9, color='#1565C0', fontstyle='italic')

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('MAE Improvement (%)', fontsize=11)
    ax.set_title('TS-Booster vs Vanilla Linear: MAE Improvement (5 seeds, error bars = 1 std)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'mae_improvement.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'mae_improvement.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: mae_improvement.png/pdf")


def plot_mda_comparison(results):
    """Grouped bar chart: Vanilla vs Booster MDA per dataset."""
    datasets, v_means, b_means, v_stds, b_stds = [], [], [], [], []
    for ds in ALL_DATASETS:
        if ds not in results:
            continue
        v = _get_metric(results, ds, "vanilla", "mda")
        b = _get_metric(results, ds, "booster", "mda")
        if v is None or len(v) == 0:
            continue
        datasets.append(ds)
        v_means.append(np.mean(v) * 100)
        b_means.append(np.mean(b) * 100)
        v_stds.append(np.std(v, ddof=1) * 100)
        b_stds.append(np.std(b, ddof=1) * 100)

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    w = 0.35

    ax.bar(x - w/2, v_means, w, yerr=v_stds, capsize=2, label='Vanilla',
           color='#BDBDBD', edgecolor='white', error_kw={'linewidth': 0.8})
    ax.bar(x + w/2, b_means, w, yerr=b_stds, capsize=2, label='TS-Booster',
           color='#43A047', edgecolor='white', error_kw={'linewidth': 0.8})

    # Value labels on Booster bars
    for i, (bm, bs) in enumerate(zip(b_means, b_stds)):
        ax.text(i + w/2, bm + bs + 0.3, f'{bm:.1f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold', color='#2E7D32')

    n_fx = len([d for d in datasets if d in FX_DATASETS])
    if 0 < n_fx < len(datasets):
        ax.axvline(x=n_fx - 0.5, color='#424242', linewidth=1.5, linestyle='--', alpha=0.5)

    avg_v = np.mean(v_means)
    avg_b = np.mean(b_means)
    ax.axhline(y=avg_v, color='#757575', linewidth=1, linestyle=':', alpha=0.5)
    ax.axhline(y=avg_b, color='#2E7D32', linewidth=1, linestyle=':', alpha=0.5)
    ax.text(0.02, 0.95, f'avg: Vanilla={avg_v:.1f}%, Booster={avg_b:.1f}%',
            transform=ax.transAxes, fontsize=9, color='#424242', fontstyle='italic')

    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('MDA (%)', fontsize=11)
    ax.set_title('Mean Directional Accuracy: Vanilla vs TS-Booster (5 seeds)',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(50, max(b_means) + 8)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / 'mda_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(FIGURES_DIR / 'mda_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: mda_comparison.png/pdf")


def generate_figures(results):
    """Generate all main experiment figures."""
    print("\nGenerating figures...")
    plot_improvement_bars(results)
    # MAE & MDA figures (only if new format with extra metrics)
    sample = results[ALL_DATASETS[0]]["vanilla"]
    if isinstance(sample, dict):
        plot_mae_improvement(results)
        plot_mda_comparison(results)
    print(f"Figures saved to {FIGURES_DIR}/")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true", help="Only analyze saved results")
    args = parser.parse_args()

    if args.analyze:
        if not RESULTS_PATH.exists():
            print(f"No saved results at {RESULTS_PATH}. Run without --analyze first.")
            sys.exit(1)
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = run_experiments()

    analyze(results)
    generate_figures(results)
