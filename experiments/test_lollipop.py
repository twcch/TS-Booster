"""
Test: Lollipop plot showing relative RMSE improvement of TSB over each baseline,
per dataset. Replaces the grouped horizontal bar chart.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('experiments/comprehensive_results')

FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD",
               "EUR_TWD", "THB_TWD", "EUR_USD", "GBP_USD", "USD_JPY"]
STOCK_DATASETS = ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
ALL_DATASETS = FX_DATASETS + STOCK_DATASETS

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 13,
})
DPI = 250


def load_json(relpath):
    return json.loads((RESULTS_DIR / relpath).read_text())


def metric_mean(results, ds, method, metric='rmse'):
    if ds not in results or method not in results[ds]:
        return None
    vals = results[ds][method].get(metric)
    if vals is None:
        return None
    return float(np.mean(vals)) if isinstance(vals, list) else float(vals)


def lollipop_detail(results, tsb_method, competitors, comp_colors,
                    title, filename):
    """
    Lollipop chart: Y=datasets, X=average relative RMSE improvement (%) of TSB
    over all competitors. One dot per dataset. Positive = TSB wins.
    """
    datasets = [ds for ds in ALL_DATASETS
                if ds in results and tsb_method in results[ds]]
    fx_ds  = [ds for ds in datasets if ds in FX_DATASETS]
    stk_ds = [ds for ds in datasets if ds in STOCK_DATASETS]
    groups = [('FX', fx_ds), ('Stocks', stk_ds)]

    DOT_COLOR = '#7B1FA2'
    marker_size = 100
    lw = 2.2

    # Compute all improvements — vs best baseline per dataset
    all_imps = {}
    all_best_comp = {}
    for g_ds in [fx_ds, stk_ds]:
        for ds in g_ds:
            tsb_val = metric_mean(results, ds, tsb_method, 'rmse')
            best_val, best_name = None, None
            for comp in competitors:
                comp_val = metric_mean(results, ds, comp, 'rmse')
                if comp_val is not None:
                    if best_val is None or comp_val < best_val:
                        best_val = comp_val
                        best_name = comp
            all_imps[ds] = (best_val - tsb_val) / best_val * 100 if (tsb_val and best_val) else 0
            all_best_comp[ds] = best_name

    global_max = max(all_imps.values())
    global_min = min(all_imps.values())
    # leave room: negative side for "X wins" label, positive side for "+N% vs X" label
    x_min = global_min - 14
    x_max = global_max + 14

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)

    for ax, (group_name, g_ds) in zip(axes, groups):
        short = [ds.replace('_TWD', '').replace('_TW', '') for ds in g_ds]
        n_ds = len(g_ds)
        imps = [all_imps[ds] for ds in g_ds]

        for i, imp in enumerate(imps):
            color = DOT_COLOR if imp >= 0 else '#C62828'
            # Stem
            ax.plot([0, imp], [i, i], color=color, linewidth=lw, alpha=0.85, zorder=3)
            # Dot
            ax.scatter([imp], [i], color=color, s=marker_size, zorder=5,
                       edgecolors='white', linewidths=1.0)
            # Label — always outside the dot, away from y-axis labels
            best_name = all_best_comp[g_ds[i]]
            if imp >= 0:
                label = f'+{imp:.1f}% vs {best_name}'
                ax.text(imp + 1.0, i, label,
                        va='center', ha='left', fontsize=10, color=color, fontweight='bold')
            else:
                label = f'{best_name} wins ({imp:.1f}%)'
                # Place below the dot to avoid covering the y-axis label
                ax.text(imp, i + 0.35, label,
                        va='top', ha='center', fontsize=10, color='#C62828', fontweight='bold')

        ax.axvline(x=0, color='#424242', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.axvspan(0, x_max, alpha=0.05, color='#2E7D32', zorder=0)

        ax.set_xlim(x_min, x_max)
        ax.set_yticks(range(n_ds))
        ax.set_yticklabels(short, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel('RMSE Improvement over Best Baseline (%)', fontsize=11)
        ax.set_title(group_name, fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)

    plt.tight_layout()
    out = RESULTS_DIR / 'slides_figures' / filename
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    bench = load_json('benchmark/benchmark_baselines_results.json')
    sota  = load_json('sota_baselines/sota_baselines_results.json')

    lollipop_detail(
        bench, 'Linear+TSB',
        competitors=['ARIMA', 'XGBoost', 'LightGBM'],
        comp_colors=['#1565C0', '#FF7043', '#66BB6A'],
        title='RMSE Improvement of TSB over ML Baselines — per Dataset (5-seed mean)',
        filename='exp2_vs_ml_detail_lollipop.png',
    )

    lollipop_detail(
        sota, 'Linear+TSB',
        competitors=['DLinear', 'PatchTST', 'iTransformer'],
        comp_colors=['#FF9800', '#00ACC1', '#E91E63'],
        title='RMSE Improvement of TSB over DL Baselines — per Dataset (5-seed mean)',
        filename='exp3_vs_dl_detail_lollipop.png',
    )
