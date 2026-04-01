"""
Paper-ready figures v2: Clean, intuitive, one-glance clarity.

Design principles:
  - Title states the FINDING, not just the content
  - Heatmaps for multi-method × multi-dataset (never grouped bars)
  - 3-metric panels (RMSE/MAE/MDA) where data is available
  - Consistent green=good, red=bad
  - FX / Stocks group separation
  - Win counts + averages at bottom
  - Unified style across all experiments

Output: experiments/comprehensive_results/paper_figures_v2/
"""
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path('experiments/comprehensive_results')
OUTPUT_DIR = RESULTS_DIR / 'paper_figures_v2'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD",
               "EUR_TWD", "THB_TWD", "EUR_USD", "GBP_USD", "USD_JPY"]
STOCK_DATASETS = ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
ALL_DATASETS = FX_DATASETS + STOCK_DATASETS
N_FX = len(FX_DATASETS)

BACKBONES = ["Linear", "LSTM", "GRU", "Transformer"]

# ── Unified style ──────────────────────────────────────────────────────
CMAP_IMPROVE = mcolors.LinearSegmentedColormap.from_list(
    'improve', ['#E53935', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#2E7D32'], N=256)

FONT_TITLE = 14
FONT_SUBTITLE = 12
FONT_LABEL = 11
FONT_CELL = 9
FONT_SUMMARY = 10
DPI = 250

COLOR_FX = '#1565C0'
COLOR_STOCK = '#C62828'
COLOR_WIN = '#1B5E20'
COLOR_LOSE = '#B71C1C'
COLOR_TSB = '#7B1FA2'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ── Data loaders ───────────────────────────────────────────────────────
def load_json(relpath):
    return json.loads((RESULTS_DIR / relpath).read_text())


def load_cross_backbone():
    """Load cross-backbone results with all 3 metrics."""
    raw = load_json('cross_backbone/cross_backbone_all_results.json')
    out = {}
    for ds in raw:
        out[ds] = {}
        for bb in raw[ds]:
            e = raw[ds][bb]
            entry = {}
            for metric in ['rmse', 'mae', 'mda']:
                vk = f'vanilla_{metric}'
                bk = f'boosted_{metric}'
                v, b = e.get(vk), e.get(bk)
                if v is not None and b is not None:
                    vm = float(np.mean(v)) if isinstance(v, list) else float(v)
                    bm = float(np.mean(b)) if isinstance(b, list) else float(b)
                    entry[f'vanilla_{metric}'] = vm
                    entry[f'boosted_{metric}'] = bm
                    if metric == 'mda':
                        # MDA: higher is better, so improvement = boosted - vanilla (in %)
                        # Show as percentage point difference
                        entry[f'{metric}_improvement'] = (bm - vm) * 100  # pp
                    else:
                        # RMSE/MAE: lower is better
                        entry[f'{metric}_improvement'] = (vm - bm) / vm * 100
            out[ds][bb] = entry
    return out


def load_ablation():
    """Load ablation results (RMSE only)."""
    raw = load_json('ablation/ablation_5seed.json')
    out = {}
    for ds in raw:
        out[ds] = {}
        for cfg in raw[ds]:
            out[ds][cfg] = float(np.mean(raw[ds][cfg]['rmse']))
    return out


def _metric_mean(results, ds, method, metric='rmse'):
    if ds not in results or method not in results[ds]:
        return None
    vals = results[ds][method].get(metric)
    if vals is None:
        return None
    return float(np.mean(vals)) if isinstance(vals, list) else float(vals)


# ── Helper: single improvement heatmap panel ───────────────────────────
def _draw_single_heatmap(ax, matrix, row_labels, col_labels,
                          metric_name='RMSE', show_row_wins=False,
                          row_win_total=None, is_mda=False):
    """Draw one heatmap panel on the given axes."""
    n_rows, n_cols = matrix.shape

    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    im = ax.imshow(matrix, cmap=CMAP_IMPROVE, vmin=-vmax, vmax=vmax, aspect='auto')

    # Axes
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=FONT_LABEL, fontweight='bold')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=FONT_CELL + 1)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # FX / Stocks separator
    if n_rows == len(ALL_DATASETS):
        ax.axhline(y=N_FX - 0.5, color='#424242', linewidth=2)

    # Cell text
    unit = 'pp' if is_mda else '%'
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            brightness = abs(val) / vmax if vmax > 0 else 0
            color = 'white' if brightness > 0.55 else 'black'
            ax.text(j, i, f'{val:+.1f}{unit}', ha='center', va='center',
                    fontsize=FONT_CELL, fontweight='bold', color=color)

    # Per-row win count (right side)
    if show_row_wins:
        total = row_win_total or n_cols
        for i in range(n_rows):
            wins = int(np.sum(matrix[i, :] > 0))
            c = COLOR_WIN if wins == total else '#E65100' if wins >= total / 2 else COLOR_LOSE
            ax.text(n_cols - 0.5 + 0.7, i, f'{wins}/{total}',
                    ha='left', va='center', fontsize=FONT_CELL, fontweight='bold', color=c)

    # Bottom summary: wins + avg
    y_wins = n_rows + 0.3
    y_avg = n_rows + 0.75

    for j in range(n_cols):
        col = matrix[:, j]
        wins = int(np.sum(col > 0))
        avg = float(np.nanmean(col))
        wc = COLOR_WIN if wins > n_rows / 2 else COLOR_LOSE
        ac = COLOR_WIN if avg > 0 else COLOR_LOSE
        ax.text(j, y_wins, f'{wins}/{n_rows}',
                ha='center', va='center', fontsize=FONT_SUMMARY, fontweight='bold', color=wc)
        ax.text(j, y_avg, f'{avg:+.1f}{unit}',
                ha='center', va='center', fontsize=FONT_SUMMARY, fontweight='bold', color=ac)

    # Left labels for summary rows
    ax.text(-0.6, y_wins, 'Wins:', ha='right', va='center',
            fontsize=FONT_SUMMARY, fontweight='bold', color='#616161')
    ax.text(-0.6, y_avg, 'Avg:', ha='right', va='center',
            fontsize=FONT_SUMMARY, fontweight='bold', color='#616161')

    ax.set_title(metric_name, fontsize=FONT_SUBTITLE, fontweight='bold', pad=8)
    return im


# ── Helper: 3-metric panel heatmap ────────────────────────────────────
def _draw_triple_heatmap(matrices, row_labels, col_labels, suptitle, filename,
                          metric_names=('RMSE ↓', 'MAE ↓', 'MDA ↑'),
                          show_row_wins=False, row_win_total=None,
                          is_mda_flags=(False, False, True)):
    """
    Draw 3 side-by-side heatmaps (RMSE, MAE, MDA) sharing the same row labels.
    """
    n_rows = matrices[0].shape[0]
    n_cols = matrices[0].shape[1]
    # Each panel: ~(n_cols * 1.6 + 1.5) wide, total with gaps
    panel_w = n_cols * 1.5 + 1.2
    fig_width = panel_w * 3 + 2.5  # extra for y-labels and spacing
    fig_height = max(10, n_rows * 0.58 + 3.5)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height),
                              gridspec_kw={'wspace': 0.25})

    for idx, (ax, mat, name, is_mda) in enumerate(
            zip(axes, matrices, metric_names, is_mda_flags)):
        show_ylabels = (idx == 0)
        im = _draw_single_heatmap(
            ax, mat, row_labels if show_ylabels else [''] * len(row_labels),
            col_labels, metric_name=name,
            show_row_wins=(show_row_wins and idx == 2),  # wins on rightmost
            row_win_total=row_win_total, is_mda=is_mda,
        )

        # FX / Stocks labels only on leftmost panel
        if show_ylabels and n_rows == len(ALL_DATASETS):
            ax.annotate('FX', xy=(-0.35, 1 - (N_FX / 2) / n_rows),
                        xycoords='axes fraction', fontsize=12,
                        fontweight='bold', color=COLOR_FX,
                        ha='center', va='center', rotation=90)
            n_stock = n_rows - N_FX
            ax.annotate('Stocks', xy=(-0.35, 1 - (N_FX + n_stock / 2) / n_rows),
                        xycoords='axes fraction', fontsize=12,
                        fontweight='bold', color=COLOR_STOCK,
                        ha='center', va='center', rotation=90)

    fig.suptitle(suptitle, fontsize=FONT_TITLE, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


# ── Helper: rank bar chart ─────────────────────────────────────────────
def _draw_rank_bar(results, methods, method_colors, title, filename, metric='rmse'):
    """Average-rank bar chart. Lower = better."""
    ranks_per = {m: [] for m in methods}
    for ds in ALL_DATASETS:
        avail = {}
        for m in methods:
            v = _metric_mean(results, ds, m, metric)
            if v is not None:
                avail[m] = v
        if not avail:
            continue
        sorted_m = sorted(avail, key=lambda k: avail[k])
        for rank, m in enumerate(sorted_m, 1):
            ranks_per[m].append(rank)

    labels, avg_ranks, colors = [], [], []
    for m, c in zip(methods, method_colors):
        if ranks_per[m]:
            labels.append(m)
            avg_ranks.append(np.mean(ranks_per[m]))
            colors.append(c)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avg_ranks, color=colors, edgecolor='#37474F',
                  linewidth=0.8, width=0.55)

    # Highlight best
    best_idx = int(np.argmin(avg_ranks))
    bars[best_idx].set_edgecolor('#FFD600')
    bars[best_idx].set_linewidth(3)

    for bar, rank in zip(bars, avg_ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                f'{rank:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Average Rank (lower = better)', fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT_LABEL)
    ax.set_ylim(0, max(avg_ranks) + 0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


# ====================================================================
# Fig 1: Architecture
# ====================================================================
def fig1_architecture():
    src = RESULTS_DIR / 'architecture' / 'ts_booster_architecture_v2.png'
    if src.exists():
        shutil.copy2(src, OUTPUT_DIR / 'fig1_architecture.png')
        print("  Fig 1: Architecture copied")
    else:
        print("  Fig 1: SKIP (not found)")


# ====================================================================
# Fig 2: Main Experiment — 4 Backbones × 16 Datasets × 3 Metrics
# ====================================================================
def fig2_main_triple(cb):
    datasets = [ds for ds in ALL_DATASETS if ds in cb]
    metrics = [('rmse', False), ('mae', False), ('mda', True)]
    matrices = []
    metric_names = []

    for metric, is_mda in metrics:
        mat = np.full((len(datasets), len(BACKBONES)), np.nan)
        for i, ds in enumerate(datasets):
            for j, bb in enumerate(BACKBONES):
                if bb in cb[ds]:
                    key = f'{metric}_improvement'
                    if key in cb[ds][bb]:
                        mat[i, j] = cb[ds][bb][key]
        matrices.append(mat)
        arrow = '↑' if is_mda else '↓'
        metric_names.append(f'{metric.upper()} {arrow}')

    _draw_triple_heatmap(
        matrices, datasets, BACKBONES,
        suptitle='TS-Booster Improves All 4 Backbones Across All 16 Datasets (5-seed mean)',
        filename='fig2_main_3metrics.png',
        metric_names=metric_names,
        show_row_wins=True, row_win_total=4,
        is_mda_flags=[False, False, True],
    )


# ====================================================================
# Fig 3: Ablation — Progressive Component Addition (RMSE only)
# ====================================================================
def fig3_ablation(abl):
    datasets = [ds for ds in ALL_DATASETS if ds in abl]
    configs = ['+Decomp', '+Expert', 'TS-Booster']

    matrix = np.zeros((len(datasets), len(configs)))
    for i, ds in enumerate(datasets):
        vanilla = abl[ds]['Vanilla']
        for j, cfg in enumerate(configs):
            matrix[i, j] = (vanilla - abl[ds][cfg]) / vanilla * 100

    # Single panel (only RMSE available)
    fig, ax = plt.subplots(figsize=(7, 10))
    im = _draw_single_heatmap(ax, matrix, datasets, configs,
                               metric_name='RMSE Improvement over Vanilla (%)')

    # FX / Stocks labels
    ax.annotate('FX', xy=(-0.28, 1 - (N_FX / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=12,
                fontweight='bold', color=COLOR_FX,
                ha='center', va='center', rotation=90)
    n_stock = len(datasets) - N_FX
    ax.annotate('Stocks', xy=(-0.28, 1 - (N_FX + n_stock / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=12,
                fontweight='bold', color=COLOR_STOCK,
                ha='center', va='center', rotation=90)

    ax.set_title('Each Component Adds Value: Progressive Ablation\n'
                 '(RMSE improvement over Vanilla %, 5-seed mean)',
                 fontsize=FONT_TITLE, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.5, pad=0.08, label='Improvement (%)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_ablation.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: fig3_ablation.png")


# ====================================================================
# Fig 4: SOTA — 3-metric improvement heatmap (TSB vs each SOTA method)
# ====================================================================
def fig4_sota_improvement(sota):
    sota_methods = ['DLinear', 'PatchTST', 'iTransformer']
    datasets = [ds for ds in ALL_DATASETS if ds in sota]
    metrics_info = [('rmse', False), ('mae', False), ('mda', True)]
    matrices = []
    metric_names = []

    for metric, is_mda in metrics_info:
        mat = np.full((len(datasets), len(sota_methods)), np.nan)
        for i, ds in enumerate(datasets):
            tsb = _metric_mean(sota, ds, 'Linear+TSB', metric)
            if tsb is None:
                continue
            for j, m in enumerate(sota_methods):
                val = _metric_mean(sota, ds, m, metric)
                if val is None:
                    continue
                if is_mda:
                    # MDA higher is better: improvement = (TSB - other) in pp
                    mat[i, j] = (tsb - val) * 100
                else:
                    # RMSE/MAE lower is better: improvement = (other - TSB) / other
                    mat[i, j] = (val - tsb) / val * 100
        matrices.append(mat)
        arrow = '↑' if is_mda else '↓'
        metric_names.append(f'{metric.upper()} {arrow}')

    _draw_triple_heatmap(
        matrices, datasets, sota_methods,
        suptitle='TS-Booster Outperforms All SOTA DL Models (5-seed mean)',
        filename='fig4_sota_improvement.png',
        metric_names=metric_names,
        is_mda_flags=[False, False, True],
    )


# ====================================================================
# Fig 5: SOTA — Average Rank
# ====================================================================
def fig5_sota_rank(sota):
    methods = ['DLinear', 'PatchTST', 'iTransformer', 'Linear', 'Linear+TSB']
    colors = ['#FF9800', '#00ACC1', '#E91E63', '#42A5F5', COLOR_TSB]
    _draw_rank_bar(
        sota, methods, colors,
        title='TS-Booster Ranks #1 Across 16 Datasets (vs SOTA DL)',
        filename='fig5_sota_rank.png',
    )


# ====================================================================
# Fig 6: Benchmark — 3-metric improvement heatmap
# ====================================================================
def fig6_benchmark_improvement(bench):
    ext_methods = ['ARIMA', 'XGBoost', 'LightGBM']
    datasets = [ds for ds in ALL_DATASETS if ds in bench]
    metrics_info = [('rmse', False), ('mae', False), ('mda', True)]
    matrices = []
    metric_names = []

    for metric, is_mda in metrics_info:
        mat = np.full((len(datasets), len(ext_methods)), np.nan)
        for i, ds in enumerate(datasets):
            tsb = _metric_mean(bench, ds, 'Linear+TSB', metric)
            if tsb is None:
                continue
            for j, m in enumerate(ext_methods):
                val = _metric_mean(bench, ds, m, metric)
                if val is None:
                    continue
                if is_mda:
                    mat[i, j] = (tsb - val) * 100
                else:
                    mat[i, j] = (val - tsb) / val * 100
        matrices.append(mat)
        arrow = '↑' if is_mda else '↓'
        metric_names.append(f'{metric.upper()} {arrow}')

    _draw_triple_heatmap(
        matrices, datasets, ext_methods,
        suptitle='TS-Booster Outperforms Traditional Baselines (5-seed mean)',
        filename='fig6_benchmark_improvement.png',
        metric_names=metric_names,
        is_mda_flags=[False, False, True],
    )


# ====================================================================
# Fig 7: Benchmark — Average Rank
# ====================================================================
def fig7_benchmark_rank(bench):
    methods = ['ARIMA', 'XGBoost', 'LightGBM', 'Linear', 'Linear+TSB']
    colors = ['#9E9E9E', '#FF7043', '#66BB6A', '#42A5F5', COLOR_TSB]
    _draw_rank_bar(
        bench, methods, colors,
        title='TS-Booster Ranks #1 Across 16 Datasets (vs Baselines)',
        filename='fig7_benchmark_rank.png',
    )


# ====================================================================
# Fig 8: Sensitivity — Robustness (shared y-axis)
# ====================================================================
def fig8_sensitivity(sens):
    param_info = {
        'decomp':       ('Decomp Kernels',              '[7,13]'),
        'boost_rounds': ('Boost Rounds',                 '2'),
        'aux_weight':   ('Expert Aux Weight (λ_aux)',    '0.3'),
        'base_weight':  ('Base Weight (λ_base)',         '0.1'),
    }

    params = [p for p in param_info if p in sens]
    n = len(params)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.5))
    if n == 1:
        axes = [axes]

    # Precompute all data
    param_data = {}
    all_means = []
    for p in params:
        labels = sorted(sens[p].keys(), key=lambda x: str(x))
        means = []
        for label in labels:
            ds_means = [np.mean(v) for ds, v in sens[p][label].items()
                        if ds in ALL_DATASETS and v]
            means.append(np.mean(ds_means) if ds_means else 0)
        param_data[p] = (labels, means)
        all_means.extend(means)

    # Shared y-range
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
                    f'{m:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        # Mark default with star
        if default in labels:
            di = labels.index(default)
            ax.annotate('★ default', xy=(di, means[di]),
                        xytext=(di, y_lo + y_range * 0.05),
                        ha='center', fontsize=8, color=COLOR_WIN, fontweight='bold')

        ax.set_xlabel(title, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(y_lo, y_hi)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if ax_idx == 0:
            ax.set_ylabel('Mean RMSE (16 datasets)', fontsize=10)

        # Delta annotation
        rng = max(means) - min(means)
        ax.text(0.95, 0.95, f'Δ = {rng:.4f}\n({rng / np.mean(means) * 100:.2f}%)',
                transform=ax.transAxes, fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', alpha=0.8))

    fig.suptitle('Hyperparameter Sensitivity: Architecture Params Are Robust\n'
                 '(3 seeds × 16 datasets, shared y-axis)',
                 fontsize=FONT_TITLE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig8_sensitivity.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: fig8_sensitivity.png")


# ====================================================================
# Fig 9: Backtest — Sharpe Ratio Difference
# ====================================================================
def fig9_backtest():
    try:
        bt = load_json('backtest/backtest_results.json')
    except FileNotFoundError:
        print("  Fig 9: SKIP (no backtest data)")
        return

    datasets = [ds for ds in ALL_DATASETS if ds in bt]
    tsb_sharpe, lin_sharpe, ds_labels = [], [], []
    for ds in datasets:
        t = bt[ds].get('Linear+TSB', {}).get('sharpe')
        l = bt[ds].get('Linear', {}).get('sharpe')
        if t is not None and l is not None:
            tsb_sharpe.append(t)
            lin_sharpe.append(l)
            ds_labels.append(ds)

    if not ds_labels:
        print("  Fig 9: SKIP (no sharpe data)")
        return

    diffs = [t - l for t, l in zip(tsb_sharpe, lin_sharpe)]
    colors = [COLOR_WIN if d > 0 else COLOR_LOSE for d in diffs]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(ds_labels))
    bars = ax.barh(y, diffs, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for i, (bar, diff) in enumerate(zip(bars, diffs)):
        w = bar.get_width()
        side = 'left' if w >= 0 else 'right'
        offset = 0.003 if w >= 0 else -0.003
        ax.text(w + offset, i, f'{diff:+.3f}', ha=side, va='center',
                fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='#424242', linewidth=1)
    wins = sum(1 for d in diffs if d > 0)

    # FX / Stocks separator
    n_fx_in = len([d for d in ds_labels if d in FX_DATASETS])
    if 0 < n_fx_in < len(ds_labels):
        ax.axhline(y=n_fx_in - 0.5, color='#424242', linewidth=1, linestyle='--', alpha=0.5)

    ax.set_title(f'Better Volatility Prediction → Better Sharpe Ratio\n'
                 f'(TSB wins {wins}/{len(diffs)} datasets)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.set_xlabel('Sharpe Ratio Difference (TSB − Linear)', fontsize=FONT_LABEL)
    ax.set_yticks(y)
    ax.set_yticklabels(ds_labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig9_backtest_sharpe.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: fig9_backtest_sharpe.png")


# ====================================================================
# Fig 10: CD Diagrams (copy existing)
# ====================================================================
def fig10_cd_diagrams():
    for name in ['cd_diagram_sota.png', 'cd_diagram_benchmark.png']:
        src = RESULTS_DIR / 'cd_diagram' / name
        dst = OUTPUT_DIR / f'fig10_{name}'
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Fig 10: Copied {name}")


# ====================================================================
# Fig 11: Interpretability — Gate vs Improvement scatter
# ====================================================================
def fig11_interpretability():
    try:
        interp = load_json('interpretability/interpretability_results.json')
    except FileNotFoundError:
        print("  Fig 11: SKIP (no interpretability data)")
        return

    try:
        cb = load_cross_backbone()
    except FileNotFoundError:
        cb = {}

    datasets, gate_vals, improvements = [], [], []
    for ds in ALL_DATASETS:
        if ds not in interp:
            continue
        entry = interp[ds]

        # Gate mean from gate_values list
        gv = entry.get('gate_values')
        if not gv or not isinstance(gv, list):
            continue
        gate_mean = float(np.mean(gv))

        # Improvement from cross-backbone or from preds
        imp = None
        if ds in cb and 'Linear' in cb[ds]:
            imp = cb[ds]['Linear'].get('rmse_improvement')
        if imp is None:
            v_preds = entry.get('vanilla_preds')
            b_preds = entry.get('booster_preds')
            targets = entry.get('targets')
            if v_preds and b_preds and targets:
                v_rmse = np.sqrt(np.mean((np.array(v_preds) - np.array(targets)) ** 2))
                b_rmse = np.sqrt(np.mean((np.array(b_preds) - np.array(targets)) ** 2))
                imp = (v_rmse - b_rmse) / v_rmse * 100

        if imp is not None:
            datasets.append(ds)
            gate_vals.append(gate_mean)
            improvements.append(float(imp))

    if not datasets:
        print("  Fig 11: SKIP (no gate data)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [COLOR_FX if ds in FX_DATASETS else COLOR_STOCK for ds in datasets]
    ax.scatter(gate_vals, improvements, c=colors, s=100, edgecolors='#37474F',
              linewidth=0.8, zorder=5)

    for ds, gv, imp in zip(datasets, gate_vals, improvements):
        label = ds.replace('_TW', '').replace('_TWD', '')
        ax.annotate(label, (gv, imp), fontsize=7, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points')

    # Trend line
    if len(gate_vals) > 2:
        z = np.polyfit(gate_vals, improvements, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(gate_vals), max(gate_vals), 50)
        ax.plot(x_line, p(x_line), '--', color='#9E9E9E', linewidth=1.5, alpha=0.7)
        corr = np.corrcoef(gate_vals, improvements)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

    ax.set_xlabel('SNR Gate Activation (mean)', fontsize=FONT_LABEL)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=FONT_LABEL)
    ax.set_title('Higher Gate Activation → Larger Improvement\n'
                 '(SNR gate adapts per dataset)',
                 fontsize=FONT_TITLE, fontweight='bold')
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_FX,
               markersize=10, label='FX'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_STOCK,
               markersize=10, label='Stocks'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig11_gate_vs_improvement.png', dpi=DPI,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: fig11_gate_vs_improvement.png")


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Paper Figures v2 (clean, 3-metric panels)")
    print("=" * 60)

    print("\nLoading data...")
    cb = load_cross_backbone()
    abl = load_ablation()
    sota = load_json('sota_baselines/sota_baselines_results.json')
    bench = load_json('benchmark/benchmark_baselines_results.json')
    sens = load_json('sensitivity/sensitivity_results.json')
    print(f"  Datasets: {len(cb)} (cross-backbone), {len(abl)} (ablation)")
    print(f"           {len(sota)} (SOTA), {len(bench)} (benchmark)")

    print("\n--- Generating figures ---")
    fig1_architecture()           # Fig 1: Architecture
    fig2_main_triple(cb)          # Fig 2: Main — 3 metrics × 4 backbones
    fig3_ablation(abl)            # Fig 3: Ablation — RMSE only
    fig4_sota_improvement(sota)   # Fig 4: SOTA — 3 metrics improvement
    fig5_sota_rank(sota)          # Fig 5: SOTA — rank bar
    fig6_benchmark_improvement(bench)  # Fig 6: Benchmark — 3 metrics
    fig7_benchmark_rank(bench)    # Fig 7: Benchmark — rank bar
    fig8_sensitivity(sens)        # Fig 8: Sensitivity — robustness
    fig9_backtest()               # Fig 9: Backtest — Sharpe diff
    fig10_cd_diagrams()           # Fig 10: CD diagrams
    fig11_interpretability()      # Fig 11: Gate vs improvement

    print(f"\n{'=' * 60}")
    print(f"All figures saved to: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  {f.name}")
