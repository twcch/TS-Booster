"""
Oral defense slides figures — 4 experiments, each 1-2 figures.
One figure = one takeaway, large fonts, projection-friendly.

Experiment 1: Overall — TSB improves all 4 backbones × 16 datasets × 3 metrics
Experiment 2: vs ML baselines (ARIMA, XGBoost, LightGBM)
Experiment 3: vs DL baselines (DLinear, PatchTST, iTransformer)
Experiment 4: Backtest — Sharpe + Cumulative Return

+ Supporting: Ablation, Sensitivity, Compute Cost, Interpretability, CD diagrams

Output: experiments/comprehensive_results/slides_figures/
"""
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Paths & Constants ──────────────────────────────────────────────────
RESULTS_DIR = Path('experiments/comprehensive_results')
OUTPUT_DIR = RESULTS_DIR / 'slides_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD",
               "EUR_TWD", "THB_TWD", "EUR_USD", "GBP_USD", "USD_JPY"]
STOCK_DATASETS = ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
ALL_DATASETS = FX_DATASETS + STOCK_DATASETS
BACKBONES = ["Linear", "LSTM", "GRU", "Transformer"]

# ── Style ──────────────────────────────────────────────────────────────
import matplotlib.colors as mcolors

COLOR_TSB = '#7B1FA2'
COLOR_FX = '#1565C0'
COLOR_STOCK = '#E53935'
COLOR_ALL = '#2E7D32'
COLOR_RMSE = '#1565C0'
COLOR_MAE = '#E65100'
COLOR_MDA = '#2E7D32'
METRIC_COLORS = [COLOR_RMSE, COLOR_MAE, COLOR_MDA]
METRIC_NAMES = ['RMSE', 'MAE', 'MDA']

CMAP_IMPROVE = mcolors.LinearSegmentedColormap.from_list(
    'improve', ['#E53935', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#2E7D32'], N=256)

# Display-only renames (data keys in results JSON stay unchanged)
DISPLAY_NAMES = {'Persistence': 'Naïve'}


def _display(name):
    return DISPLAY_NAMES.get(name, name)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 13,
})
DPI = 250


# ── Data loaders ───────────────────────────────────────────────────────
def load_json(relpath):
    return json.loads((RESULTS_DIR / relpath).read_text())


def load_cross_backbone():
    raw = load_json('cross_backbone/cross_backbone_all_results.json')
    out = {}
    for ds in raw:
        out[ds] = {}
        for bb in raw[ds]:
            e = raw[ds][bb]
            entry = {}
            for metric in ['rmse', 'mae', 'mda']:
                vk, bk = f'vanilla_{metric}', f'boosted_{metric}'
                v, b = e.get(vk), e.get(bk)
                if v is not None and b is not None:
                    vm = float(np.mean(v)) if isinstance(v, list) else float(v)
                    bm = float(np.mean(b)) if isinstance(b, list) else float(b)
                    if metric == 'mda':
                        entry[f'{metric}_imp'] = (bm - vm) * 100
                    else:
                        entry[f'{metric}_imp'] = (vm - bm) / vm * 100
            out[ds][bb] = entry
    return out


def load_ablation():
    raw = load_json('ablation/ablation_5seed.json')
    out = {}
    for ds in raw:
        out[ds] = {}
        for cfg in raw[ds]:
            out[ds][cfg] = float(np.mean(raw[ds][cfg]['rmse']))
    return out


def load_stat_comparison():
    """Merge Phase-3 integration per_dataset (Persistence/EWMA/HAR-RV/GARCH/
    GJR-GARCH/TSB) with ARIMA pulled out of the ML benchmark results."""
    integ = json.loads(Path('revision/phase3_integration/integration_results.json').read_text())
    bench = load_json('benchmark/benchmark_baselines_results.json')
    out = {}
    for ds, methods in integ['per_dataset'].items():
        out[ds] = dict(methods)
        if ds in bench and 'ARIMA' in bench[ds]:
            out[ds]['ARIMA'] = bench[ds]['ARIMA']
    return out


def _metric_mean(results, ds, method, metric='rmse'):
    if ds not in results or method not in results[ds]:
        return None
    vals = results[ds][method].get(metric)
    if vals is None:
        return None
    return float(np.mean(vals)) if isinstance(vals, list) else float(vals)


# ====================================================================
# EXPERIMENT 1: Overall — Compact heatmap (4 backbone × 3 metric)
#   + per-backbone win count bar
# ====================================================================
def exp1_overall(cb):
    """
    Fig 1a: 4×3 heatmap showing avg improvement per backbone per metric.
    Fig 1b: Win count breakdown (out of 16) per backbone per metric.
    """
    metrics = ['rmse_imp', 'mae_imp', 'mda_imp']
    metric_labels = ['RMSE', 'MAE', 'MDA']
    is_mda = [False, False, True]

    # --- Fig 1a: Compact heatmap ---
    fig, ax = plt.subplots(figsize=(9, 5))

    # Matrix: rows=backbones, cols=metrics
    matrix = np.zeros((len(BACKBONES), 3))
    win_matrix = np.zeros((len(BACKBONES), 3), dtype=int)
    for i, bb in enumerate(BACKBONES):
        for j, mk in enumerate(metrics):
            vals = [cb[ds][bb][mk] for ds in ALL_DATASETS
                    if ds in cb and bb in cb[ds] and mk in cb[ds][bb]]
            matrix[i, j] = np.mean(vals) if vals else 0
            win_matrix[i, j] = sum(1 for v in vals if v > 0)

    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    im = ax.imshow(matrix, cmap=CMAP_IMPROVE, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(3))
    ax.set_xticklabels(metric_labels, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(BACKBONES)))
    ax.set_yticklabels(BACKBONES, fontsize=14, fontweight='bold')
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Cell text: avg improvement + win count
    for i in range(len(BACKBONES)):
        for j in range(3):
            val = matrix[i, j]
            wins = win_matrix[i, j]
            brightness = abs(val) / vmax if vmax > 0 else 0
            color = 'white' if brightness > 0.5 else 'black'
            unit = '%'
            ax.text(j, i, f'{val:+.1f}{unit}',
                    ha='center', va='center', fontsize=14,
                    fontweight='bold', color=color)

    # Total wins
    total_wins = int(win_matrix.sum())
    total = len(BACKBONES) * 16 * 3  # 4 backbones × 16 datasets × 3 metrics

    ax.set_title(f'TS-Booster Improves All Backbones Across All Metrics\n'
                 f'(avg improvement, 5-seed mean, 16 datasets)',
                 fontsize=15, fontweight='bold', pad=15)

    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.05, label='Improvement')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp1_overall_heatmap.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Exp1: Overall heatmap saved")

    # --- Fig 1b: Per-backbone bar with FX/Stocks breakdown ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    groups = [
        ('FX', [d for d in ALL_DATASETS if d in FX_DATASETS], COLOR_FX),
        ('Stocks', [d for d in ALL_DATASETS if d in STOCK_DATASETS], COLOR_STOCK),
    ]

    for ax_idx, (mk, ml, mda_flag) in enumerate(zip(metrics, metric_labels, is_mda)):
        ax = axes[ax_idx]
        x = np.arange(len(BACKBONES))
        width = 0.35

        for g_idx, (g_name, g_ds, g_color) in enumerate(groups):
            avgs = []
            for bb in BACKBONES:
                vals = [cb[ds][bb][mk] for ds in g_ds
                        if ds in cb and bb in cb[ds] and mk in cb[ds][bb]]
                avgs.append(np.mean(vals) if vals else 0)

            ax.bar(x + (g_idx - 0.5) * width, avgs, width,
                   label=g_name, color=g_color,
                   edgecolor='white', linewidth=0.8, alpha=0.9)

        ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(BACKBONES, fontsize=12, fontweight='bold')
        ax.set_title(ml, fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel('Avg Improvement (%)', fontsize=12)

    # Shared legend centered between suptitle and subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.98), frameon=True, edgecolor='#CCCCCC')

    fig.suptitle('TS-Booster: Consistent Improvement Across Asset Types\n'
                 '(5-seed mean, 16 datasets)',
                 fontsize=15, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp1_overall_by_group.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Exp1: Overall by group saved")


# ====================================================================
# EXPERIMENT 2 & 3: vs ML / vs DL — Improvement bar per competitor
# ====================================================================
def _exp_vs_baselines(results, tsb_method, competitors, comp_colors,
                       title, filename):
    """
    Grouped bar: X = competitor, groups = RMSE/MAE/MDA improvement.
    Shows avg improvement of TSB over each competitor + win counts.
    """
    metrics = ['rmse', 'mae', 'mda']
    metric_labels = ['RMSE', 'MAE', 'MDA']
    is_mda = [False, False, True]

    # 3 subplots (one per metric), each showing bars per competitor.
    # Cap total width so figures with many competitors (e.g. 6 statistical
    # baselines) stay visually consistent with the 2-3 competitor figures.
    fig_width = min(5 * len(competitors), 20)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 5.5))

    for m_idx, (metric, ml, mda_flag, mc) in enumerate(
            zip(metrics, metric_labels, is_mda, METRIC_COLORS)):
        ax = axes[m_idx]
        avgs = []
        wins_list = []
        for comp in competitors:
            improvements = []
            for ds in ALL_DATASETS:
                tsb_val = _metric_mean(results, ds, tsb_method, metric)
                comp_val = _metric_mean(results, ds, comp, metric)
                if tsb_val is None or comp_val is None:
                    continue
                if mda_flag:
                    improvements.append((tsb_val - comp_val) * 100)
                else:
                    improvements.append((comp_val - tsb_val) / comp_val * 100)
            avg = np.mean(improvements) if improvements else 0
            wins = sum(1 for v in improvements if v > 0)
            avgs.append(avg)
            wins_list.append(wins)

        x = np.arange(len(competitors))
        bars = ax.bar(x, avgs, color=mc, edgecolor='#37474F',
                      linewidth=0.8, width=0.55, alpha=0.85)

        for bar, avg, wins in zip(bars, avgs, wins_list):
            unit = '%'
            va = 'bottom' if avg >= 0 else 'top'
            y_offset = max(abs(avg) * 0.06, 0.3) * (1 if avg >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, avg + y_offset,
                    f'{avg:+.1f}{unit}',
                    ha='center', va=va, fontsize=12, fontweight='bold')

        ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels([_display(c) for c in competitors], fontsize=12, fontweight='bold')
        ax.set_title(ml, fontsize=14, fontweight='bold', pad=12)
        ax.grid(axis='y', alpha=0.3)
        # Add top padding so labels don't collide with title
        y_max = max(avgs) if avgs else 1
        ax.set_ylim(top=y_max * 1.25)
        if m_idx == 0:
            ax.set_ylabel('Avg Improvement (%)', fontsize=12)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def exp2_vs_stat(stat):
    """Experiment 2: Linear+TSB vs statistical volatility models."""
    _exp_vs_baselines(
        stat, 'TSB',
        competitors=['Persistence', 'EWMA', 'HAR-RV', 'GARCH', 'GJR-GARCH', 'ARIMA'],
        comp_colors=['#9E9E9E', '#FFA726', '#26A69A', '#5C6BC0', '#8D6E63', '#78909C'],
        title='TS-Booster + Linear vs Statistical Volatility Models\n'
              '(5-seed mean)',
        filename='exp2_vs_stat.png',
    )


def exp3_vs_ml(bench):
    """Experiment 3: Linear+TSB vs ML baselines."""
    _exp_vs_baselines(
        bench, 'Linear+TSB',
        competitors=['XGBoost', 'LightGBM'],
        comp_colors=['#FF7043', '#66BB6A'],
        title='TS-Booster + Linear vs Traditional ML Baselines\n'
              '(5-seed mean)',
        filename='exp3_vs_ml.png',
    )


def exp4_vs_dl(sota):
    """Experiment 4: Linear+TSB vs SOTA DL models."""
    _exp_vs_baselines(
        sota, 'Linear+TSB',
        competitors=['DLinear', 'PatchTST', 'iTransformer'],
        comp_colors=['#FF9800', '#00ACC1', '#E91E63'],
        title='TS-Booster + Linear vs SOTA DL Models\n'
              '(5-seed mean, best LR per dataset)',
        filename='exp4_vs_dl.png',
    )


def _exp_per_dataset_lollipop(results, tsb_method, competitors, title, filename):
    """Lollipop chart: Y=datasets, X=RMSE improvement (%) over best baseline per dataset."""
    datasets = [ds for ds in ALL_DATASETS
                if ds in results and tsb_method in results[ds]]
    fx_ds  = [ds for ds in datasets if ds in FX_DATASETS]
    stk_ds = [ds for ds in datasets if ds in STOCK_DATASETS]
    groups = [('FX', fx_ds), ('Stocks', stk_ds)]

    # Compute improvement vs best baseline per dataset
    all_imps, all_best_comp = {}, {}
    for ds in datasets:
        tsb_val = _metric_mean(results, ds, tsb_method, 'rmse')
        best_val, best_name = None, None
        for comp in competitors:
            comp_val = _metric_mean(results, ds, comp, 'rmse')
            if comp_val is not None and (best_val is None or comp_val < best_val):
                best_val, best_name = comp_val, comp
        all_imps[ds] = (best_val - tsb_val) / best_val * 100 if (tsb_val and best_val) else 0
        all_best_comp[ds] = best_name

    global_max = max(all_imps.values())
    global_min = min(all_imps.values())
    x_min = global_min - 22   # extra room for negative labels on the left
    x_max = global_max + 16   # extra room for positive labels on the right

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    for ax, (group_name, g_ds) in zip(axes, groups):
        short = [ds.replace('_TWD', '').replace('_TW', '') for ds in g_ds]
        imps = [all_imps[ds] for ds in g_ds]

        for i, (ds, imp) in enumerate(zip(g_ds, imps)):
            color = COLOR_TSB if imp >= 0 else '#C62828'
            ax.plot([0, imp], [i, i], color=color, linewidth=2.2, alpha=0.85, zorder=3)
            ax.scatter([imp], [i], color=color, s=100, zorder=5,
                       edgecolors='white', linewidths=1.0)
            best_name = _display(all_best_comp[ds])
            if imp >= 0:
                ax.text(imp + 1.0, i, f'+{imp:.1f}% vs {best_name}',
                        va='center', ha='left', fontsize=10, color=color, fontweight='bold')
            else:
                ax.text(imp - 1.0, i, f'{best_name} wins ({imp:.1f}%)',
                        va='center', ha='right', fontsize=10, color='#C62828', fontweight='bold')

        ax.axvline(x=0, color='#424242', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.axvspan(0, x_max, alpha=0.05, color='#2E7D32', zorder=0)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(range(len(g_ds)))
        ax.set_yticklabels(short, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('RMSE Improvement over Best Baseline (%)', fontsize=11)
        ax.set_title(group_name, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def exp2_vs_stat_detail(stat):
    _exp_per_dataset_lollipop(
        stat, 'TSB',
        competitors=['Persistence', 'EWMA', 'HAR-RV', 'GARCH', 'GJR-GARCH', 'ARIMA'],
        title='RMSE Improvement of TS-Booster + Linear over Statistical Baselines — per Dataset (5-seed mean)',
        filename='exp2_vs_stat_detail.png',
    )


def exp3_vs_ml_detail(bench):
    _exp_per_dataset_lollipop(
        bench, 'Linear+TSB',
        competitors=['XGBoost', 'LightGBM'],
        title='RMSE Improvement of TS-Booster + Linear over ML Baselines — per Dataset (5-seed mean)',
        filename='exp3_vs_ml_detail.png',
    )


def exp4_vs_dl_detail(sota):
    _exp_per_dataset_lollipop(
        sota, 'Linear+TSB',
        competitors=['DLinear', 'PatchTST', 'iTransformer'],
        title='RMSE Improvement of TS-Booster + Linear over DL Baselines — per Dataset (5-seed mean, best LR)',
        filename='exp4_vs_dl_detail.png',
    )


# ====================================================================
# EXPERIMENT 4: Backtest — Sharpe bar + Cumulative Return curves
# ====================================================================
def exp5_backtest():
    try:
        bt = load_json('backtest/backtest_results.json')
        dr = load_json('backtest/backtest_daily_returns.json')
    except FileNotFoundError:
        print("  Exp5: SKIP (no backtest data)")
        return

    # --- Fig 5a: Sharpe lollipop — paired comparison TSB vs its own vanilla
    # Linear backbone. This isolates the effect of TSB's more accurate
    # volatility prediction (controlled comparison; matches tsb_vs_linear.png).
    datasets = [ds for ds in ALL_DATASETS
                if ds in bt and 'Linear+TSB' in bt[ds] and 'Linear' in bt[ds]]

    sharpe_imps = {}
    for ds in datasets:
        tsb_val = bt[ds]['Linear+TSB'].get('sharpe')
        lin_val = bt[ds]['Linear'].get('sharpe')
        if tsb_val is not None and lin_val is not None:
            sharpe_imps[ds] = tsb_val - lin_val

    if sharpe_imps:
        fx_ds  = [ds for ds in datasets if ds in FX_DATASETS  and ds in sharpe_imps]
        stk_ds = [ds for ds in datasets if ds in STOCK_DATASETS and ds in sharpe_imps]
        n_wins = sum(1 for v in sharpe_imps.values() if v > 0)

        all_vals = list(sharpe_imps.values())
        x_min = min(all_vals) - 0.05
        x_max = max(all_vals) + 0.07

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(f'Better Volatility Prediction → Better Sharpe Ratio\n'
                     f'TS-Booster + Linear vs Vanilla Linear, Paired Comparison ({n_wins}/{len(sharpe_imps)} wins)',
                     fontsize=15, fontweight='bold', y=1.02)

        for ax, g_ds, group_name in zip(axes, [fx_ds, stk_ds], ['FX', 'Stocks']):
            short = [ds.replace('_TWD', '').replace('_TW', '') for ds in g_ds]
            for i, ds in enumerate(g_ds):
                imp = sharpe_imps[ds]
                color = COLOR_TSB if imp >= 0 else '#C62828'
                ax.plot([0, imp], [i, i], color=color, linewidth=2.2, alpha=0.85, zorder=3)
                ax.scatter([imp], [i], color=color, s=100, zorder=5,
                           edgecolors='white', linewidths=1.0)
                if imp >= 0:
                    ax.text(imp + 0.004, i, f'+{imp:.3f}',
                            va='center', ha='left', fontsize=10, color=color, fontweight='bold')
                else:
                    ax.text(imp - 0.004, i, f'{imp:.3f}',
                            va='center', ha='right', fontsize=10, color='#C62828', fontweight='bold')

            ax.axvline(x=0, color='#424242', linewidth=1.2, linestyle='--', alpha=0.5)
            ax.axvspan(0, x_max, alpha=0.05, color='#2E7D32', zorder=0)
            ax.set_xlim(x_min, x_max)
            ax.set_yticks(range(len(g_ds)))
            ax.set_yticklabels(short, fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlabel('Sharpe Difference (TS-Booster + Linear − Vanilla Linear)', fontsize=11)
            ax.set_title(group_name, fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'exp5a_sharpe.png', dpi=DPI,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Exp5a: Sharpe lollipop saved")

    # --- Fig 5b: Cumulative Return lollipop — paired comparison TSB vs Linear ---
    nav_imps = {}
    for ds in ALL_DATASETS:
        if ds not in dr:
            continue
        tsb_ret = dr[ds].get('Linear+TSB')
        lin_ret = dr[ds].get('Linear')
        if not tsb_ret or not lin_ret:
            continue
        tsb_nav = float(np.cumprod(1 + np.array(tsb_ret))[-1])
        lin_nav = float(np.cumprod(1 + np.array(lin_ret))[-1])
        nav_imps[ds] = (tsb_nav - lin_nav) / lin_nav * 100

    if nav_imps:
        fx_ds  = [ds for ds in ALL_DATASETS if ds in FX_DATASETS   and ds in nav_imps]
        stk_ds = [ds for ds in ALL_DATASETS if ds in STOCK_DATASETS and ds in nav_imps]
        n_wins = sum(1 for v in nav_imps.values() if v > 0)

        all_vals = list(nav_imps.values())
        x_min = min(all_vals) - 6
        x_max = max(all_vals) + 10

        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        fig.suptitle(f'TS-Booster + Linear Strategy Cumulative Return vs Vanilla Linear\n'
                     f'Paired Comparison, Inverse-Volatility Position Sizing ({n_wins}/{len(nav_imps)} wins)',
                     fontsize=15, fontweight='bold', y=1.02)

        for ax, g_ds, group_name in zip(axes, [fx_ds, stk_ds], ['FX', 'Stocks']):
            short = [ds.replace('_TWD', '').replace('_TW', '') for ds in g_ds]
            for i, ds in enumerate(g_ds):
                imp = nav_imps[ds]
                color = COLOR_TSB if imp >= 0 else '#C62828'
                ax.plot([0, imp], [i, i], color=color, linewidth=2.2, alpha=0.85, zorder=3)
                ax.scatter([imp], [i], color=color, s=100, zorder=5,
                           edgecolors='white', linewidths=1.0)
                if imp >= 0:
                    ax.text(imp + 0.6, i, f'+{imp:.1f}%',
                            va='center', ha='left', fontsize=10, color=color, fontweight='bold')
                else:
                    ax.text(imp - 0.6, i, f'{imp:.1f}%',
                            va='center', ha='right', fontsize=10, color='#C62828', fontweight='bold')

            ax.axvline(x=0, color='#424242', linewidth=1.2, linestyle='--', alpha=0.5)
            ax.axvspan(0, x_max, alpha=0.05, color='#2E7D32', zorder=0)
            ax.set_xlim(x_min, x_max)
            ax.set_yticks(range(len(g_ds)))
            ax.set_yticklabels(short, fontsize=12, fontweight='bold')
            ax.invert_yaxis()
            ax.set_xlabel('Cumulative Return Difference vs Vanilla Linear (%)', fontsize=11)
            ax.set_title(group_name, fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.2)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'exp5b_cumulative_return.png', dpi=DPI,
                    bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Exp5b: Cumulative return lollipop saved")


# ====================================================================
# SUPPORTING: Ablation, Sensitivity, Compute Cost, Interpretability, CD
# ====================================================================
def exp6_ablation(abl):
    configs = ['Vanilla', '+Decomp', '+Expert', 'TS-Booster']
    config_colors = ['#90A4AE', '#FFB74D', '#42A5F5', COLOR_TSB]

    avg_rmse = []
    for cfg in configs:
        vals = [abl[ds][cfg] for ds in ALL_DATASETS if ds in abl and cfg in abl[ds]]
        avg_rmse.append(np.mean(vals))

    vanilla = avg_rmse[0]
    improvements = [(vanilla - v) / vanilla * 100 for v in avg_rmse]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                     gridspec_kw={'width_ratios': [1, 1.2]})

    # Left: RMSE bars
    x = np.arange(len(configs))
    bars = ax1.bar(x, avg_rmse, color=config_colors, edgecolor='#37474F',
                   linewidth=1, width=0.6)
    for bar, rmse in zip(bars, avg_rmse):
        ax1.text(bar.get_x() + bar.get_width() / 2, rmse - 0.05,
                 f'{rmse:.3f}', ha='center', va='top', fontsize=11,
                 fontweight='bold', color='white')
    for i in range(1, len(configs)):
        if improvements[i] > improvements[i - 1]:
            delta = improvements[i] - improvements[i - 1]
            ax1.annotate(f'{delta:+.1f}%',
                         xy=(i, avg_rmse[i] + 0.02), ha='center', va='bottom',
                         fontsize=10, fontweight='bold',
                         color=COLOR_TSB if i == 3 else '#E65100')

    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Avg RMSE', fontsize=12)
    ax1.set_title('Progressive Improvement', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(min(avg_rmse) * 0.97, max(avg_rmse) * 1.01)

    # Right: % improvement per config (relative to Vanilla)
    cfg_labels = configs[1:]
    imp_vals = [improvements[i] for i in range(1, len(configs))]

    x2 = np.arange(len(cfg_labels))
    bars2 = ax2.bar(x2, imp_vals, color=config_colors[1:],
                    edgecolor='#37474F', linewidth=1, width=0.55)
    for bar, imp in zip(bars2, imp_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, imp + 0.3,
                 f'{imp:+.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(cfg_labels, fontsize=11, fontweight='bold')
    ax2.set_ylabel('RMSE Improvement over Vanilla (%)', fontsize=12)
    ax2.set_title('Improvement by Component', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('Ablation: Expert Module Is the Key Component',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp6_ablation.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Exp6: Ablation saved")


def exp6_sensitivity(sens):
    param_info = {
        'decomp':       ('Decomp Kernels',      '[7,13]'),
        'boost_rounds': ('Boost Rounds',         '2'),
        'aux_weight':   ('Aux Weight (λ_aux)',   '0.3'),
        'base_weight':  ('Base Weight (λ_base)', '0.1'),
    }
    params = [p for p in param_info if p in sens]
    n = len(params)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5))
    if n == 1:
        axes = [axes]

    param_data = {}
    all_means = []
    for p in params:
        labels = sorted(sens[p].keys(), key=str)
        means = []
        for label in labels:
            ds_means = [np.mean(v) for ds, v in sens[p][label].items()
                        if ds in ALL_DATASETS and v]
            means.append(np.mean(ds_means) if ds_means else 0)
        param_data[p] = (labels, means)
        all_means.extend(means)

    g_min, g_max = min(all_means), max(all_means)
    y_center = (g_min + g_max) / 2
    y_range = max(g_max - g_min, 0.05) * 2.5
    y_lo, y_hi = y_center - y_range / 2, y_center + y_range / 2

    for ax_idx, p in enumerate(params):
        ax = axes[ax_idx]
        title, default = param_info[p]
        labels, means = param_data[p]
        x = np.arange(len(labels))
        bar_colors = ['#43A047' if l == default else '#90CAF9' for l in labels]
        bars = ax.bar(x, means, color=bar_colors, edgecolor='#37474F',
                      linewidth=0.8, width=0.55)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m + (y_range * 0.012),
                    f'{m:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        if default in labels:
            di = labels.index(default)
            ax.annotate('★ default', xy=(di, y_lo + y_range * 0.05),
                        ha='center', fontsize=9, color='#1B5E20', fontweight='bold')
        ax.set_xlabel(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if ax_idx == 0:
            ax.set_ylabel('Mean RMSE', fontsize=11)
        rng = max(means) - min(means)
        ax.text(0.95, 0.95, f'Δ={rng:.4f}\n({rng/np.mean(means)*100:.2f}%)',
                transform=ax.transAxes, fontsize=9, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8))

    fig.suptitle('Hyperparameter Sensitivity: All Architecture Params Are Robust',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'exp6_sensitivity.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Exp6: Sensitivity saved")


def support_compute_cost():
    params = {
        'Linear':      (366,    7156),
        'LSTM':        (50883,  57673),
        'GRU':         (38211,  45001),
        'Transformer': (67331,  74121),
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(BACKBONES))
    width = 0.35
    vanilla_vals = [params[bb][0] for bb in BACKBONES]
    tsb_vals = [params[bb][1] for bb in BACKBONES]

    ax.bar(x - width / 2, vanilla_vals, width, label='Vanilla',
           color='#90CAF9', edgecolor='#37474F', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, tsb_vals, width, label='+ TS-Booster',
                   color=COLOR_TSB, edgecolor='#37474F', linewidth=0.8, alpha=0.9)

    for bar, bb in zip(bars2, BACKBONES):
        ov = params[bb][1] - params[bb][0]
        pct = ov / params[bb][0] * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f'+{ov:,}\n(+{pct:.0f}%)', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color=COLOR_TSB)

    ax.text(0.98, 0.95, 'TSB fixed overhead:\n+6,790 params',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONES, fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter Count', fontsize=12)
    ax.set_title('Lightweight: Fixed +6,790 Params Regardless of Backbone',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    ax.set_ylim(100, 200000)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'support_compute_cost.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Support: Compute cost saved")


def support_revin_check():
    """Phase-4 RevIN fairness check: PatchTST (w/ vs w/o RevIN) vs TSB+Linear — RMSE per dataset."""
    path = Path('revision/phase4_revin_fix/patchtst_no_revin_results.json')
    if not path.exists():
        print("  Support: RevIN check SKIP")
        return
    raw = json.loads(path.read_text())

    datasets = [ds for ds in ALL_DATASETS if ds in raw]
    fx_ds = [ds for ds in datasets if ds in FX_DATASETS]
    stk_ds = [ds for ds in datasets if ds in STOCK_DATASETS]
    groups = [('FX', fx_ds), ('Stocks', stk_ds)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('RevIN Fairness Check: PatchTST (w/ vs w/o RevIN) vs TS-Booster + Linear — RMSE (5-seed mean)',
                 fontsize=15, fontweight='bold', y=1.02)

    for ax, (group_name, g_ds) in zip(axes, groups):
        short = [ds.replace('_TWD', '').replace('_TW', '') for ds in g_ds]
        x = np.arange(len(g_ds))
        width = 0.26

        no_revin = [float(np.mean(raw[ds]['patchtst_no_revin']['rmse'])) for ds in g_ds]
        w_revin = [raw[ds]['patchtst_orig_rmse'] for ds in g_ds]
        tsb = [raw[ds]['tsb_rmse'] for ds in g_ds]

        ax.bar(x - width, no_revin, width, label='PatchTST (no RevIN)',
               color='#FFCCBC', edgecolor='#37474F', linewidth=0.7)
        ax.bar(x, w_revin, width, label='PatchTST (w/ RevIN)',
               color='#00ACC1', edgecolor='#37474F', linewidth=0.7, alpha=0.9)
        ax.bar(x + width, tsb, width, label='TS-Booster + Linear',
               color=COLOR_TSB, edgecolor='#37474F', linewidth=0.7, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=10, rotation=45, ha='right')
        ax.set_title(group_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'support_revin_check.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Support: RevIN fairness check saved")


def support_interpretability_3panel():
    """2 figures: SNR gate (2 panels) + Decomposition kernel weights (1 panel). Line charts."""
    try:
        interp = load_json('interpretability/interpretability_results.json')
    except FileNotFoundError:
        print("  Support: Interpretability SKIP")
        return

    datasets = [ds for ds in ALL_DATASETS if ds in interp]
    if not datasets:
        return

    short_labels = [ds.replace('_TWD', '').replace('_TW', '') for ds in datasets]
    x = np.arange(len(datasets))
    n_fx = len([d for d in datasets if d in FX_DATASETS])

    # ── Figure 1: SNR Gate (2 panels) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Gate activation (mean) per dataset
    gate_means = [float(np.mean(interp[ds]['gate_values'])) for ds in datasets]
    ax1.plot(x, gate_means, marker='o', color=COLOR_TSB, linewidth=2, markersize=7, zorder=3)
    # Color markers by group
    for i, ds in enumerate(datasets):
        c = COLOR_FX if ds in FX_DATASETS else COLOR_STOCK
        ax1.plot(i, gate_means[i], marker='o', color=c, markersize=8, zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9, rotation=45, ha='right')
    ax1.set_ylabel('Mean Gate Activation', fontsize=11)
    ax1.set_title('SNR Gate Value per Dataset', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(x=n_fx - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.4)
    y_top = ax1.get_ylim()[1]
    y_label = y_top - (y_top - ax1.get_ylim()[0]) * 0.06
    ax1.text(n_fx / 2, y_label, 'FX', ha='center', va='top', fontsize=9, color='#999')
    ax1.text(n_fx + (len(datasets) - n_fx) / 2, y_label, 'Stocks',
             ha='center', va='top', fontsize=9, color='#999')

    # Panel 2: Gate params (w, b) per dataset
    ws = [interp[ds]['snr_w'] for ds in datasets]
    bs = [interp[ds]['snr_b'] for ds in datasets]
    ax2.plot(x, ws, marker='s', color='#42A5F5', linewidth=2, markersize=7, label='w')
    ax2.plot(x, bs, marker='^', color='#EF5350', linewidth=2, markersize=7, label='b')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9, rotation=45, ha='right')
    ax2.set_ylabel('Parameter Value', fontsize=11)
    ax2.set_title('SNR Gate Parameters (w, b)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='#9E9E9E', linewidth=0.8, linestyle='--')
    ax2.axvline(x=n_fx - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.4)

    fig.suptitle('SNR Gate: Learned Gating per Dataset',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'support_snr_gate.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Support: SNR gate saved")

    # ── Figure 2: Decomposition Kernel Weights ──
    fig, ax = plt.subplots(figsize=(10, 5))

    w7 = [interp[ds]['decomp_weights']['7'] for ds in datasets]
    w13 = [interp[ds]['decomp_weights']['13'] for ds in datasets]

    ax.plot(x, w7, marker='o', color='#66BB6A', linewidth=2, markersize=8, label='Kernel 7')
    ax.plot(x, w13, marker='s', color='#FFA726', linewidth=2, markersize=8, label='Kernel 13')
    ax.fill_between(x, w7, w13, alpha=0.1, color='#66BB6A')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=10, rotation=45, ha='right')
    ax.set_ylabel('Softmax Weight', fontsize=12)
    ax.set_title('Decomposition Kernel Weights: Short-term (7) vs Mid-term (13)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axvline(x=n_fx - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 1)
    ax.text(n_fx / 2, 0.97, 'FX', ha='center', fontsize=10, color='#666')
    ax.text(n_fx + (len(datasets) - n_fx) / 2, 0.97, 'Stocks',
            ha='center', fontsize=10, color='#666')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'support_decomp_weights.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Support: Decomp weights saved")


def copy_architecture():
    src = RESULTS_DIR / 'architecture' / 'ts_booster_architecture_v2.png'
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / 'slide_architecture.png')
        print("  Slide: Architecture copied")


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Slides Figures (6 experiments + supporting)")
    print("=" * 60)

    print("\nLoading data...")
    cb = load_cross_backbone()
    abl = load_ablation()
    stat = load_stat_comparison()
    sota = load_json('sota_baselines/sota_baselines_results.json')
    bench = load_json('benchmark/benchmark_baselines_results.json')
    sens = load_json('sensitivity/sensitivity_results.json')

    print("\n--- Experiment figures ---")
    exp1_overall(cb)              # Exp 1: 泛化性與穩健性驗證 (4 backbone × 3 metrics)
    exp2_vs_stat(stat)            # Exp 2: 與統計模型比較 (summary)
    exp2_vs_stat_detail(stat)     # Exp 2: 與統計模型比較 (per-dataset)
    exp3_vs_ml(bench)             # Exp 3: 與機器學習基線比較 (summary)
    exp3_vs_ml_detail(bench)      # Exp 3: 與機器學習基線比較 (per-dataset)
    exp4_vs_dl(sota)              # Exp 4: 與 SOTA 深度學習模型比較 (summary)
    exp4_vs_dl_detail(sota)       # Exp 4: 與 SOTA 深度學習模型比較 (per-dataset)
    exp5_backtest()               # Exp 5: 回測實驗 (Sharpe + Cumulative Return)
    exp6_ablation(abl)            # Exp 6: 消融與敏感度分析 (ablation)
    exp6_sensitivity(sens)        # Exp 6: 消融與敏感度分析 (sensitivity)

    print("\n--- Supporting figures ---")
    support_compute_cost()
    support_revin_check()
    support_interpretability_3panel()

    print(f"\n{'=' * 60}")
    print(f"All slide figures saved to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  {f.name}")
