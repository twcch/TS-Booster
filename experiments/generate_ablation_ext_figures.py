"""
Generate extended ablation figures:
  1. SNR Gate ablation (no_gate / const_0.5 / learned / snr)
  2. Param-Matched comparison (vanilla / mlp_7k / TS-Booster + Linear)
Output: experiments/comprehensive_results/slides_figures/
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path('experiments/comprehensive_results/slides_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_DEFAULT = '#43A047'   # green — our method (matches sensitivity figure)
COLOR_OTHER   = '#90CAF9'   # light blue
COLOR_VANILLA = '#B0BEC5'   # grey — vanilla baseline
DPI = 250

def load(path):
    return json.loads(Path(path).read_text())

# ── SNR Gate Ablation ──────────────────────────────────────────────────
snr = load('revision/phase6_ablation_ext/snr_gate_ablation_results.json')
datasets = list(snr.keys())

configs_snr   = ['no_gate', 'const_0.5', 'learned', 'snr']
labels_snr    = ['No Gate', 'Const (0.5)', 'Learned', 'SNR (Ours)']
default_snr   = 'snr'

rmse_snr = [np.mean([np.mean(snr[ds][c]['rmse']) for ds in datasets]) for c in configs_snr]
mda_snr  = [np.mean([np.mean(snr[ds][c]['mda'])  for ds in datasets]) for c in configs_snr]
wins_rmse_snr = [sum(1 for ds in datasets if
    np.mean(snr[ds][c]['rmse']) == min(np.mean(snr[ds][cc]['rmse']) for cc in configs_snr))
    for c in configs_snr]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

def draw_bar(ax, labels, values, wins, n_ds, default_label, ylabel, title, higher_better=False):
    x = np.arange(len(labels))
    colors = [COLOR_DEFAULT if l == default_label else COLOR_OTHER for l in labels]
    bars = ax.bar(x, values, color=colors, edgecolor='#37474F', linewidth=0.8, width=0.55)

    y_min, y_max = min(values), max(values)
    pad = max((y_max - y_min) * 2.5, abs(y_max) * 0.05)
    y_lo = y_min - pad * 0.3
    y_hi = y_max + pad * 0.7

    for bar, val, win in zip(bars, values, wins):
        top = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, top + (y_hi - y_lo) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, y_lo + (y_hi - y_lo) * 0.05,
                f'{win}/{n_ds}', ha='center', va='bottom', fontsize=9, color='#37474F')

    di = labels.index(default_label)
    ax.annotate('★ Ours', xy=(di, y_hi - (y_hi - y_lo) * 0.08),
                ha='center', fontsize=9, color='#1B5E20', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

draw_bar(axes[0], labels_snr, rmse_snr, wins_rmse_snr, len(datasets),
         'SNR (Ours)', 'Mean RMSE (lower is better)', 'RMSE')
mda_wins_snr = [sum(1 for ds in datasets if
    np.mean(snr[ds][c]['mda']) == max(np.mean(snr[ds][cc]['mda']) for cc in configs_snr))
    for c in configs_snr]
draw_bar(axes[1], labels_snr, mda_snr, mda_wins_snr, len(datasets),
         'SNR (Ours)', 'Mean MDA (higher is better)', 'MDA', higher_better=True)

fig.suptitle('SNR Gate Ablation: Gate Design Comparison\n(5-seed mean, 16 datasets)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'support_snr_gate_ablation.png',
            dpi=DPI, bbox_inches='tight', facecolor='white')
print('Saved → support_snr_gate_ablation.png')

# ── Param-Matched ──────────────────────────────────────────────────────
pm = load('revision/phase6_ablation_ext/param_matched_results.json')
datasets_pm = list(pm.keys())

configs_pm = ['vanilla', 'mlp7k', 'tsb']
labels_pm  = ['Vanilla\n(366 params)', 'MLP\n(~7k params)', 'TS-Booster + Linear\n(7,156 params)']
default_pm = 'TS-Booster + Linear\n(7,156 params)'

rmse_pm = [np.mean([np.mean(pm[ds][c]['rmse']) for ds in datasets_pm]) for c in configs_pm]
mda_pm  = [np.mean([np.mean(pm[ds][c]['mda'])  for ds in datasets_pm]) for c in configs_pm]
wins_rmse_pm = [sum(1 for ds in datasets_pm if
    np.mean(pm[ds][c]['rmse']) == min(np.mean(pm[ds][cc]['rmse']) for cc in configs_pm))
    for c in configs_pm]
wins_mda_pm = [sum(1 for ds in datasets_pm if
    np.mean(pm[ds][c]['mda']) == max(np.mean(pm[ds][cc]['mda']) for cc in configs_pm))
    for c in configs_pm]

fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

colors_pm = [COLOR_VANILLA, COLOR_OTHER, COLOR_DEFAULT]

def draw_bar_pm(ax, labels, values, wins, n_ds, colors, ylabel, title):
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='#37474F', linewidth=0.8, width=0.55)

    y_min, y_max = min(values), max(values)
    pad = max((y_max - y_min) * 2.5, abs(y_max) * 0.05)
    y_lo = y_min - pad * 0.3
    y_hi = y_max + pad * 0.7

    for bar, val, win in zip(bars, values, wins):
        top = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, top + (y_hi - y_lo) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2, y_lo + (y_hi - y_lo) * 0.05,
                f'{win}/{n_ds}', ha='center', va='bottom', fontsize=9, color='#37474F')

    ax.annotate('★ Ours', xy=(2, y_hi - (y_hi - y_lo) * 0.08),
                ha='center', fontsize=9, color='#1B5E20', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

draw_bar_pm(axes2[0], labels_pm, rmse_pm, wins_rmse_pm, len(datasets_pm),
            colors_pm, 'Mean RMSE (lower is better)', 'RMSE')
draw_bar_pm(axes2[1], labels_pm, mda_pm,  wins_mda_pm,  len(datasets_pm),
            colors_pm, 'Mean MDA (higher is better)', 'MDA')

fig2.suptitle('Param-Matched Comparison: Improvement Comes from Architecture, Not Parameter Count\n'
              '(5-seed mean, 16 datasets)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'support_param_matched.png',
            dpi=DPI, bbox_inches='tight', facecolor='white')
print('Saved → support_param_matched.png')
