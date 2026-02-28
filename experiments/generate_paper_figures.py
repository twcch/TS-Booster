"""
Generate all paper-ready figures for TS-Booster thesis.
Output: experiments/comprehensive_results/paper_figures/
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path('experiments/comprehensive_results')
OUTPUT_DIR = RESULTS_DIR / 'paper_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset ordering: FX first, then stocks
FX_DATASETS = ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
                "EUR_USD", "GBP_USD", "USD_JPY"]
STOCK_DATASETS = ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
ALL_DATASETS = FX_DATASETS + STOCK_DATASETS

CONFIGS = ["Vanilla", "+Decomp", "+Expert", "TS-Booster"]
BACKBONES = ["Linear", "LSTM", "GRU", "Transformer"]

# Color palette
CMAP_DIV = mcolors.LinearSegmentedColormap.from_list('rg',
    ['#EF5350', '#FFCDD2', '#FFFFFF', '#C8E6C9', '#43A047'], N=256)


def load_ablation_results():
    """Load ablation results (prefer 5-seed unified JSON, fallback to per-dataset)."""
    unified = RESULTS_DIR / 'ablation' / 'ablation_5seed.json'
    if unified.exists():
        raw = json.loads(unified.read_text())
        # Convert to same format: {ds: {config: {"RMSE": mean_val}}}
        results = {}
        for ds in raw:
            results[ds] = {}
            for cfg in raw[ds]:
                rmse_list = raw[ds][cfg]["rmse"]
                results[ds][cfg] = {"RMSE": float(np.mean(rmse_list))}
        return results
    # Fallback: old per-dataset single-seed JSONs
    results = {}
    for f in sorted((RESULTS_DIR / 'ablation').glob('ablation_results_*.json')):
        data = json.loads(f.read_text())
        ds = data['metadata']['currency_pair']
        results[ds] = data['results']
    return results


def load_cross_backbone_results():
    """Load cross-backbone JSON and compute improvement from 5-seed mean."""
    f = RESULTS_DIR / 'cross_backbone' / 'cross_backbone_all_results.json'
    raw = json.loads(f.read_text())
    # Normalize: compute 'improvement' from vanilla/boosted RMSE lists
    for ds in raw:
        for bb in raw[ds]:
            entry = raw[ds][bb]
            v = entry.get('vanilla_rmse')
            b = entry.get('boosted_rmse')
            if v is not None and b is not None:
                v_mean = float(np.mean(v)) if isinstance(v, list) else float(v)
                b_mean = float(np.mean(b)) if isinstance(b, list) else float(b)
                entry['improvement'] = (v_mean - b_mean) / v_mean * 100
    return raw


# =========================================================================
# Figure 1: Architecture (just copy existing)
# =========================================================================
def copy_architecture():
    import shutil
    for ext in ['png', 'pdf']:
        src = RESULTS_DIR / 'architecture' / f'ts_booster_architecture_v2.{ext}'
        dst = OUTPUT_DIR / f'fig1_architecture.{ext}'
        if src.exists():
            shutil.copy2(src, dst)
    print("Fig 1: Architecture diagram copied")


# =========================================================================
# Figure 2: Ablation Heatmap (% improvement vs Vanilla)
# =========================================================================
def fig2_ablation_heatmap(ablation):
    datasets = [ds for ds in ALL_DATASETS if ds in ablation]
    configs = ["+Decomp", "+Expert", "TS-Booster"]  # skip Vanilla (baseline)

    # Build improvement matrix (RMSE reduction %)
    matrix = np.zeros((len(datasets), len(configs)))
    for i, ds in enumerate(datasets):
        vanilla_rmse = ablation[ds]['Vanilla']['RMSE']
        for j, cfg in enumerate(configs):
            cfg_rmse = ablation[ds][cfg]['RMSE']
            matrix[i, j] = (vanilla_rmse - cfg_rmse) / vanilla_rmse * 100

    fig, ax = plt.subplots(figsize=(8, 10))

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))

    # Y-axis labels: prepend group tag
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    ylabels = []
    for ds in datasets:
        ylabels.append(ds)
    ax.set_yticklabels(ylabels, fontsize=10)

    # Group separator line between FX and stocks
    ax.axhline(y=n_fx - 0.5, color='#424242', linewidth=2, linestyle='-')

    # Group bracket labels (outside plot, using fig text)
    n_stock = len(datasets) - n_fx
    # Use transform to place text in axes coordinates
    ax.annotate('FX', xy=(-0.18, 1 - (n_fx / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#1565C0', ha='center', va='center', rotation=90)
    ax.annotate('Stocks', xy=(-0.18, 1 - (n_fx + n_stock / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#C62828', ha='center', va='center', rotation=90)

    # Text annotations
    for i in range(len(datasets)):
        for j in range(len(configs)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Win count per config (bottom)
    for j, cfg in enumerate(configs):
        wins = sum(1 for i in range(len(datasets)) if matrix[i, j] > 0)
        ax.text(j, len(datasets) + 0.3, f'{wins}/{len(datasets)}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#1B5E20' if wins > len(datasets) / 2 else '#B71C1C')
    ax.text(-0.5, len(datasets) + 0.3, 'Wins:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    # Avg improvement per config
    for j, cfg in enumerate(configs):
        avg = matrix[:, j].mean()
        ax.text(j, len(datasets) + 0.8, f'{avg:+.1f}%',
                ha='center', va='center', fontsize=10,
                color='#1B5E20' if avg > 0 else '#B71C1C')
    ax.text(-0.5, len(datasets) + 0.8, 'Avg:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    ax.set_title('Ablation Study: RMSE Improvement over Vanilla (%)\n'
                 'Progressive Component Addition',
                 fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('RMSE Improvement (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_heatmap.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig2_ablation_heatmap.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig 2: Ablation heatmap saved")


# =========================================================================
# Figure 3: Cross-Backbone Heatmap (% improvement)
# =========================================================================
def fig3_cross_backbone_heatmap(cb_results):
    datasets = [ds for ds in ALL_DATASETS if ds in cb_results]

    matrix = np.zeros((len(datasets), len(BACKBONES)))
    for i, ds in enumerate(datasets):
        for j, bb in enumerate(BACKBONES):
            if bb in cb_results[ds]:
                matrix[i, j] = cb_results[ds][bb]['improvement']

    fig, ax = plt.subplots(figsize=(8, 10))

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    im = ax.imshow(matrix, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(BACKBONES)))
    ax.set_xticklabels(BACKBONES, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    # Group separator
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    n_stock = len(datasets) - n_fx
    ax.axhline(y=n_fx - 0.5, color='#424242', linewidth=2, linestyle='-')
    ax.annotate('FX', xy=(-0.16, 1 - (n_fx / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#1565C0', ha='center', va='center', rotation=90)
    ax.annotate('Stocks', xy=(-0.16, 1 - (n_fx + n_stock / 2) / len(datasets)),
                xycoords='axes fraction', fontsize=11, fontweight='bold',
                color='#C62828', ha='center', va='center', rotation=90)

    # Text annotations
    for i in range(len(datasets)):
        for j in range(len(BACKBONES)):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Per-dataset win count (right)
    for i, ds in enumerate(datasets):
        wins = sum(1 for j in range(len(BACKBONES)) if matrix[i, j] > 0)
        color = '#1B5E20' if wins >= 3 else '#E65100' if wins >= 2 else '#B71C1C'
        ax.text(len(BACKBONES) + 0.15, i, f'{wins}/4', ha='left', va='center',
                fontsize=9, fontweight='bold', color=color)

    # Win count per backbone (bottom)
    for j, bb in enumerate(BACKBONES):
        wins = sum(1 for i in range(len(datasets)) if matrix[i, j] > 0)
        ax.text(j, len(datasets) + 0.3, f'{wins}/{len(datasets)}',
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='#1B5E20' if wins > len(datasets) / 2 else '#B71C1C')
    ax.text(-0.6, len(datasets) + 0.3, 'Wins:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    # Avg improvement per backbone
    for j, bb in enumerate(BACKBONES):
        avg = matrix[:, j].mean()
        ax.text(j, len(datasets) + 0.8, f'{avg:+.1f}%',
                ha='center', va='center', fontsize=10,
                color='#1B5E20' if avg > 0 else '#B71C1C')
    ax.text(-0.6, len(datasets) + 0.8, 'Avg:', ha='right', va='center',
            fontsize=10, fontweight='bold', color='#424242')

    ax.set_title('Cross-Backbone Validation: TS-Booster RMSE Improvement (%)\n'
                 'Vanilla → + TS-Booster (per backbone, per dataset)',
                 fontsize=13, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label('RMSE Improvement (%)', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_cross_backbone_heatmap.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig3_cross_backbone_heatmap.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig 3: Cross-backbone heatmap saved")


# =========================================================================
# Figure 4: Component Contribution Summary Bar (avg across datasets)
# =========================================================================
def fig4_component_summary(ablation):
    datasets = [ds for ds in ALL_DATASETS if ds in ablation]
    configs = ["+Decomp", "+Expert", "TS-Booster"]

    # Compute per-config average improvement
    fx_ds = [d for d in datasets if d in FX_DATASETS]
    stock_ds = [d for d in datasets if d in STOCK_DATASETS]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(configs))
    width = 0.25

    for group_idx, (group_name, group_ds, color) in enumerate([
        ('FX', fx_ds, '#42A5F5'),
        ('Stocks', stock_ds, '#EF5350'),
        ('All', datasets, '#66BB6A'),
    ]):
        avgs = []
        stds = []
        for cfg in configs:
            improvements = []
            for ds in group_ds:
                v = ablation[ds]['Vanilla']['RMSE']
                c = ablation[ds][cfg]['RMSE']
                improvements.append((v - c) / v * 100)
            avgs.append(np.mean(improvements))
            stds.append(np.std(improvements))

        bars = ax.bar(x + (group_idx - 1) * width, avgs, width,
                      label=group_name, color=color, edgecolor='#37474F',
                      linewidth=0.8, alpha=0.85, yerr=stds, capsize=3)

        for bar, avg in zip(bars, avgs):
            h = bar.get_height()
            offset = 0.5 if h >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset + (1.5 if stds[0] > 0 else 0),
                    f'{avg:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Avg RMSE Improvement over Vanilla (%)', fontsize=11)
    ax.set_title('Component Contribution Summary', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_component_summary.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig4_component_summary.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig 4: Component contribution summary saved")


# =========================================================================
# Figure 5: Cross-Backbone Summary Bar (avg per backbone)
# =========================================================================
def fig5_backbone_summary(cb_results):
    datasets = [ds for ds in ALL_DATASETS if ds in cb_results]
    fx_ds = [d for d in datasets if d in FX_DATASETS]
    stock_ds = [d for d in datasets if d in STOCK_DATASETS]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(BACKBONES))
    width = 0.25

    for group_idx, (group_name, group_ds, color) in enumerate([
        ('FX', fx_ds, '#42A5F5'),
        ('Stocks', stock_ds, '#EF5350'),
        ('All', datasets, '#66BB6A'),
    ]):
        avgs = []
        stds = []
        for bb in BACKBONES:
            improvements = []
            for ds in group_ds:
                if bb in cb_results[ds]:
                    improvements.append(cb_results[ds][bb]['improvement'])
            avgs.append(np.mean(improvements))
            stds.append(np.std(improvements))

        bars = ax.bar(x + (group_idx - 1) * width, avgs, width,
                      label=group_name, color=color, edgecolor='#37474F',
                      linewidth=0.8, alpha=0.85, yerr=stds, capsize=3)

        for bar, avg in zip(bars, avgs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                    f'{avg:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=0, color='#9E9E9E', linewidth=1, linestyle='--')
    ax.set_xlabel('Backbone', fontsize=12)
    ax.set_ylabel('Avg RMSE Improvement (%)', fontsize=11)
    ax.set_title('Cross-Backbone: TS-Booster Avg Improvement by Asset Type',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(BACKBONES, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_backbone_summary.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig5_backbone_summary.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig 5: Backbone summary saved")


# =========================================================================
# Figure 6: Ablation RMSE absolute table-style figure
# =========================================================================
def fig6_ablation_table(ablation):
    """Table-style figure with absolute RMSE for all datasets × configs."""
    datasets = [ds for ds in ALL_DATASETS if ds in ablation]
    configs = CONFIGS

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.axis('off')

    # Header
    col_labels = ['Dataset'] + configs + ['Best']
    n_cols = len(col_labels)
    n_rows = len(datasets) + 3  # datasets + header + separator + summary

    # Table data
    cell_text = []
    cell_colors = []

    for ds in datasets:
        row = [ds]
        rmses = []
        for cfg in configs:
            rmse = ablation[ds][cfg]['RMSE']
            rmses.append(rmse)
            row.append(f'{rmse:.3f}')
        best_idx = np.argmin(rmses)
        best_cfg = configs[best_idx]
        row.append(best_cfg)
        cell_text.append(row)

        # Color: highlight best
        colors = ['#FAFAFA'] * (n_cols)
        colors[best_idx + 1] = '#C8E6C9'
        if ds in FX_DATASETS:
            colors[0] = '#E3F2FD'
        else:
            colors[0] = '#FCE4EC'
        cell_colors.append(colors)

    # Summary row
    summary_row = ['Average']
    for cfg in configs:
        avg = np.mean([ablation[ds][cfg]['RMSE'] for ds in datasets])
        summary_row.append(f'{avg:.3f}')
    summary_row.append('')
    cell_text.append(summary_row)
    cell_colors.append(['#E0E0E0'] * n_cols)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=cell_colors,
                     loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # Bold header
    for j in range(n_cols):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#424242')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # FX/Stocks separator: add bottom border to last FX row
    n_fx = len([d for d in datasets if d in FX_DATASETS])
    for j in range(n_cols):
        table[n_fx, j].set_edgecolor('#424242')
        table[n_fx, j].set_linewidth(2)

    # Bold summary row
    last_row = len(datasets) + 1
    for j in range(n_cols):
        table[last_row, j].set_text_props(fontweight='bold')

    ax.set_title('Ablation Study: Test RMSE (lower is better)\n'
                 'Green = best per dataset',
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_ablation_table.png', dpi=250, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_DIR / 'fig6_ablation_table.pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print("Fig 6: Ablation table saved")


# =========================================================================
# Main
# =========================================================================
if __name__ == '__main__':
    print("Loading results...")
    ablation = load_ablation_results()
    cb_results = load_cross_backbone_results()
    print(f"  Ablation: {len(ablation)} datasets")
    print(f"  Cross-backbone: {len(cb_results)} datasets")
    print()

    copy_architecture()       # Fig 1: Architecture
    fig2_ablation_heatmap(ablation)    # Fig 2: Ablation % heatmap
    fig3_cross_backbone_heatmap(cb_results)  # Fig 3: Cross-backbone heatmap
    fig4_component_summary(ablation)   # Fig 4: Component contribution bar
    fig5_backbone_summary(cb_results)  # Fig 5: Backbone summary bar
    fig6_ablation_table(ablation)      # Fig 6: Ablation absolute RMSE table

    print(f"\nAll paper figures saved to: {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  {f.name}")
