"""
Generate temporal robustness figure (horizon + lookback grids).
Output: experiments/comprehensive_results/slides_figures/support_temporal_robustness.png
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path('experiments/comprehensive_results/slides_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_DEFAULT = '#43A047'   # green — main setting
COLOR_OTHER   = '#90CAF9'   # light blue — other settings
DPI = 250

def load(path):
    return json.loads(Path(path).read_text())

def compute_stats(data, param_values):
    """Returns (improvements%, win_rates) for each param value."""
    imps, wins = [], []
    for val in param_values:
        datasets = list(data[val].keys())
        v_list = [np.mean(data[val][ds]['vanilla']) for ds in datasets]
        t_list = [np.mean(data[val][ds]['tsb'])     for ds in datasets]
        imp = (np.mean(v_list) - np.mean(t_list)) / np.mean(v_list) * 100
        win = sum(1 for v, t in zip(v_list, t_list) if t < v)
        imps.append(imp)
        wins.append(win)
    return imps, wins, len(datasets)

# ── Load data ──────────────────────────────────────────────────────────
horizon  = load('revision/phase9_temporal/temporal_results.json')
lookback = load('revision/phase9b_lookback/lookback_results.json')

h_vals = ['1', '3', '5', '10']
l_vals = ['30', '60', '90']

h_imps, h_wins, n_ds = compute_stats(horizon,  h_vals)
l_imps, l_wins, _    = compute_stats(lookback, l_vals)

# ── Plot ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

def draw_panel(ax, labels, imps, wins, n_ds, default_label, xlabel, title):
    x = np.arange(len(labels))
    colors = [COLOR_DEFAULT if l == default_label else COLOR_OTHER for l in labels]
    bars = ax.bar(x, imps, color=colors, edgecolor='#37474F', linewidth=0.8, width=0.5)

    y_max = max(imps)
    y_range = max(y_max * 1.35, 5)

    for bar, imp, win in zip(bars, imps, wins):
        top = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, top + y_range * 0.02,
                f'+{imp:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        txt_color = 'white' if top > y_range * 0.15 else '#37474F'
        ax.text(bar.get_x() + bar.get_width() / 2, top / 2,
                f'{win}/{n_ds}', ha='center', va='center',
                fontsize=10, color=txt_color, fontweight='bold')

    di = labels.index(default_label)
    default_top = imps[di]
    ax.annotate('★ main', xy=(di, default_top + y_range * 0.12),
                ha='center', fontsize=9, color='#1B5E20', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{l}' for l in labels], fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE Improvement (%)', fontsize=11)
    ax.set_ylim(0, y_range)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

draw_panel(axes[0], h_vals, h_imps, h_wins, n_ds,
           default_label='3',
           xlabel='Prediction Horizon H (days)',
           title='Horizon Robustness  (L = 30 fixed)')

draw_panel(axes[1], l_vals, l_imps, l_wins, n_ds,
           default_label='30',
           xlabel='Lookback Window L (days)',
           title='Lookback Robustness  (H = 3 fixed)')

fig.suptitle('TS-Booster + Linear Remains Effective Across Time Scales',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'support_temporal_robustness.png',
            dpi=DPI, bbox_inches='tight', facecolor='white')
print(f'Saved → {OUTPUT_DIR}/support_temporal_robustness.png')
