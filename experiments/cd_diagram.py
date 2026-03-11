"""
Critical Difference (CD) Diagram for TS-Booster Paper.

Performs Friedman test + Nemenyi post-hoc comparison across multiple methods
and generates a CD diagram (Demsar, 2006).

Two comparison groups:
  1. SOTA DL: DLinear, PatchTST, iTransformer, Linear, Linear+TSB
  2. Benchmark: ARIMA, XGBoost, LightGBM, SVR, Linear, Linear+TSB

Usage:
  uv run python experiments/cd_diagram.py                    # Generate both diagrams
  uv run python experiments/cd_diagram.py --group sota       # SOTA only
  uv run python experiments/cd_diagram.py --group benchmark  # Benchmark only
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.append(str(Path(__file__).parent.parent))

SOTA_PATH = Path("experiments/comprehensive_results/sota_baselines/sota_baselines_results.json")
BENCH_PATH = Path("experiments/comprehensive_results/benchmark/benchmark_baselines_results.json")
FIGURES_DIR = Path("experiments/comprehensive_results/cd_diagram")

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

GROUPS = {
    "sota": {
        "path": SOTA_PATH,
        "methods": ["DLinear", "PatchTST", "iTransformer", "Linear", "Linear+TSB"],
        "title": "Critical Difference Diagram: SOTA DL Baselines",
    },
    "benchmark": {
        "path": BENCH_PATH,
        "methods": ["ARIMA", "XGBoost", "LightGBM", "Linear", "Linear+TSB"],
        "title": "Critical Difference Diagram: Traditional Baselines",
    },
}


def load_ranks(results, methods, datasets):
    """Compute per-dataset rank matrix.

    Returns:
        rank_matrix: np.array [n_datasets x n_methods], fractional ranks
        avg_ranks: np.array [n_methods]
    """
    n_ds = len(datasets)
    n_m = len(methods)
    rank_matrix = np.full((n_ds, n_m), np.nan)

    for i, ds in enumerate(datasets):
        if ds not in results:
            continue
        # Get mean RMSE per method
        rmse_vals = {}
        for j, m in enumerate(methods):
            val = results[ds].get(m)
            if val is None:
                continue
            if isinstance(val, dict):
                arr = val.get("rmse", [])
                if arr:
                    rmse_vals[j] = np.mean(arr)
            elif isinstance(val, list):
                if val:
                    rmse_vals[j] = np.mean(val)

        if not rmse_vals:
            continue

        # Fractional ranking (handle ties)
        indices = sorted(rmse_vals.keys(), key=lambda k: rmse_vals[k])
        current_rank = 1
        j_idx = 0
        while j_idx < len(indices):
            # Find group of tied values
            tied = [indices[j_idx]]
            while j_idx + 1 < len(indices) and abs(rmse_vals[indices[j_idx + 1]] - rmse_vals[indices[j_idx]]) < 1e-10:
                j_idx += 1
                tied.append(indices[j_idx])
            avg_rank = current_rank + (len(tied) - 1) / 2.0
            for t in tied:
                rank_matrix[i, t] = avg_rank
            current_rank += len(tied)
            j_idx += 1

    # Filter out datasets with any NaN (incomplete data)
    valid = ~np.any(np.isnan(rank_matrix), axis=1)
    rank_matrix = rank_matrix[valid]

    avg_ranks = np.nanmean(rank_matrix, axis=0)
    return rank_matrix, avg_ranks


def friedman_test(rank_matrix):
    """Friedman chi-squared test.

    Args:
        rank_matrix: [n_datasets x n_methods]

    Returns:
        chi2, p_value
    """
    n, k = rank_matrix.shape
    avg_ranks = np.mean(rank_matrix, axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg_ranks ** 2) - k * (k + 1) ** 2 / 4)
    p_value = 1 - stats.chi2.cdf(chi2, k - 1)
    return chi2, p_value


def nemenyi_cd(n_datasets, n_methods, alpha=0.05):
    """Compute the Nemenyi critical difference.

    Uses the q_alpha values from Demsar (2006) Table 5.
    """
    # q_alpha values for alpha=0.05 (Studentized range / sqrt(2))
    # Indexed by k (number of groups), k=2..10
    q_alpha_005 = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha_010 = {
        2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459,
        6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855, 10: 2.920,
    }

    table = q_alpha_005 if alpha == 0.05 else q_alpha_010
    k = n_methods
    if k not in table:
        # Approximate using normal for large k
        q = stats.norm.ppf(1 - alpha / (k * (k - 1)))
    else:
        q = table[k]

    cd = q * np.sqrt(k * (k + 1) / (6 * n_datasets))
    return cd


def draw_cd_diagram(avg_ranks, methods, cd, n_datasets, title, filename, alpha=0.05):
    """Draw a Critical Difference diagram following Demsar (2006).

    Methods connected by a horizontal bar are NOT significantly different.
    """
    k = len(methods)

    # Sort methods by average rank
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    sorted_methods = [methods[i] for i in sorted_indices]

    # Find cliques (groups of methods not significantly different)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(sorted_ranks[j] - sorted_ranks[i]) < cd:
                # Check if this extends an existing clique
                merged = False
                for clique in cliques:
                    if i in clique and j not in clique:
                        # Check all members
                        if all(abs(sorted_ranks[j] - sorted_ranks[m]) < cd for m in clique):
                            clique.add(j)
                            merged = True
                            break
                    elif j in clique and i not in clique:
                        if all(abs(sorted_ranks[i] - sorted_ranks[m]) < cd for m in clique):
                            clique.add(i)
                            merged = True
                            break
                    elif i in clique and j in clique:
                        merged = True
                        break
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    merged = True
    while merged:
        merged = False
        new_cliques = []
        used = set()
        for i, c1 in enumerate(cliques):
            if i in used:
                continue
            current = set(c1)
            for j, c2 in enumerate(cliques):
                if j <= i or j in used:
                    continue
                if current & c2:
                    # Check if merge is valid
                    combined = current | c2
                    valid = True
                    for a, b in combinations(combined, 2):
                        if abs(sorted_ranks[a] - sorted_ranks[b]) >= cd:
                            valid = False
                            break
                    if valid:
                        current = combined
                        used.add(j)
                        merged = True
            new_cliques.append(current)
            used.add(i)
        cliques = new_cliques

    # === Drawing ===
    fig_width = max(8, k * 1.5)
    fig_height = max(3.5, 2.5 + len(cliques) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    # Layout parameters
    low_rank = 1
    high_rank = k
    left_margin = 0.15
    right_margin = 0.15
    top_y = 0.85
    line_y = 0.7
    label_y_top = 0.88
    label_y_bot = 0.15
    cd_y = 0.95

    def rank_to_x(rank):
        return left_margin + (rank - low_rank) / (high_rank - low_rank) * (1 - left_margin - right_margin)

    # Draw axis line
    ax.plot([rank_to_x(low_rank), rank_to_x(high_rank)], [line_y, line_y],
            'k-', linewidth=1.5, transform=ax.transAxes)

    # Draw tick marks
    for r in range(low_rank, high_rank + 1):
        x = rank_to_x(r)
        ax.plot([x, x], [line_y - 0.02, line_y + 0.02], 'k-', linewidth=1.5,
                transform=ax.transAxes)
        ax.text(x, line_y + 0.04, str(r), ha='center', va='bottom', fontsize=10,
                transform=ax.transAxes)

    # Draw CD bar
    cd_left = rank_to_x(1)
    cd_right = rank_to_x(1 + cd)
    ax.annotate('', xy=(cd_right, cd_y), xytext=(cd_left, cd_y),
                arrowprops=dict(arrowstyle='<->', color='#C62828', lw=2),
                transform=ax.transAxes)
    ax.text((cd_left + cd_right) / 2, cd_y + 0.02,
            f'CD = {cd:.2f}', ha='center', va='bottom', fontsize=10,
            fontweight='bold', color='#C62828', transform=ax.transAxes)

    # Split methods: top half on left side, bottom half on right side
    n_left = (k + 1) // 2
    n_right = k - n_left

    # Draw method labels and connections
    method_positions = {}
    for idx in range(k):
        rank = sorted_ranks[idx]
        name = sorted_methods[idx]
        x_rank = rank_to_x(rank)

        if idx < n_left:
            # Left side (top performers) - labels above
            label_x = left_margin * 0.3
            label_y_pos = label_y_top - idx * (0.12)
            ha = 'left'

            # Draw connecting line
            ax.plot([label_x + 0.08, x_rank], [label_y_pos, label_y_pos],
                    'k-', linewidth=0.8, alpha=0.5, transform=ax.transAxes)
            ax.plot([x_rank, x_rank], [label_y_pos, line_y],
                    'k-', linewidth=0.8, alpha=0.5, transform=ax.transAxes)
        else:
            # Right side - labels below
            right_idx = idx - n_left
            label_x = 1 - right_margin * 0.3
            label_y_pos = label_y_bot + right_idx * 0.12
            ha = 'right'

            ax.plot([label_x - 0.08, x_rank], [label_y_pos, label_y_pos],
                    'k-', linewidth=0.8, alpha=0.5, transform=ax.transAxes)
            ax.plot([x_rank, x_rank], [label_y_pos, line_y],
                    'k-', linewidth=0.8, alpha=0.5, transform=ax.transAxes)

        # Bold the best method
        fw = 'bold' if idx == 0 else 'normal'
        ax.text(label_x, label_y_pos, f'{name} ({rank:.2f})',
                ha=ha, va='center', fontsize=10, fontweight=fw,
                transform=ax.transAxes)

        # Mark rank position on axis
        ax.plot(x_rank, line_y, 'ko', markersize=5, transform=ax.transAxes)
        method_positions[idx] = x_rank

    # Draw clique bars (thick bars connecting non-significantly-different methods)
    bar_colors = ['#1565C0', '#C62828', '#2E7D32', '#FF6F00', '#6A1B9A']
    bar_y_start = line_y - 0.06
    bar_spacing = 0.04

    for c_idx, clique in enumerate(cliques):
        if len(clique) < 2:
            continue
        sorted_clique = sorted(clique)
        left_x = method_positions[sorted_clique[0]]
        right_x = method_positions[sorted_clique[-1]]
        y = bar_y_start - c_idx * bar_spacing
        color = bar_colors[c_idx % len(bar_colors)]

        ax.plot([left_x, right_x], [y, y], '-', color=color,
                linewidth=4, alpha=0.7, solid_capstyle='round',
                transform=ax.transAxes)

    ax.set_title(f'{title}\n(Friedman + Nemenyi, $\\alpha$={alpha}, n={n_datasets} datasets)',
                 fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def run_group(group_name, results_data, group_config):
    """Run Friedman + Nemenyi for one group and generate CD diagram."""
    methods = group_config["methods"]
    title = group_config["title"]

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    rank_matrix, avg_ranks = load_ranks(results_data, methods, ALL_DATASETS)
    n_datasets = rank_matrix.shape[0]

    print(f"\n  Datasets with complete data: {n_datasets}")
    print(f"\n  Average Ranks:")
    sorted_idx = np.argsort(avg_ranks)
    for idx in sorted_idx:
        print(f"    {methods[idx]:<14}: {avg_ranks[idx]:.3f}")

    # Friedman test
    chi2, p_value = friedman_test(rank_matrix)
    print(f"\n  Friedman Test:")
    print(f"    Chi-squared = {chi2:.3f}")
    print(f"    p-value     = {p_value:.6f}")
    print(f"    Significant = {'Yes' if p_value < 0.05 else 'No'} (alpha=0.05)")

    if p_value >= 0.05:
        print(f"\n  WARNING: Friedman test not significant. CD diagram may not be meaningful.")

    if n_datasets == 0:
        print("\n  ERROR: No datasets with complete data for all methods. Skipping.")
        return None

    # Nemenyi CD
    cd = nemenyi_cd(n_datasets, len(methods), alpha=0.05)
    print(f"\n  Nemenyi Critical Difference: {cd:.3f}")

    # Pairwise comparison
    print(f"\n  Pairwise Rank Differences (significant if > {cd:.3f}):")
    print(f"  {'':14}", end="")
    for m in methods:
        print(f" {m:>14}", end="")
    print()

    for i, m1 in enumerate(methods):
        print(f"  {m1:<14}", end="")
        for j, m2 in enumerate(methods):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = "*" if diff > cd else " "
            print(f" {diff:>13.3f}{sig}", end="")
        print()

    # Draw CD diagram
    filename = f"cd_diagram_{group_name}.png"
    draw_cd_diagram(avg_ranks, methods, cd, n_datasets, title, filename)

    # LaTeX-ready summary
    print(f"\n  LaTeX (avg rank row for existing table):")
    rank_strs = []
    for m in methods:
        idx = methods.index(m)
        rank_strs.append(f"{avg_ranks[idx]:.2f}")
    print(f"  Avg.\\ Rank & " + " & ".join(rank_strs) + r" \\")

    return {
        "methods": methods,
        "avg_ranks": avg_ranks.tolist(),
        "friedman_chi2": chi2,
        "friedman_p": p_value,
        "nemenyi_cd": cd,
        "n_datasets": n_datasets,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Critical Difference diagram for TS-Booster")
    parser.add_argument("--group", type=str, choices=["sota", "benchmark", "all"],
                        default="all", help="Which comparison group to analyze")
    args = parser.parse_args()

    groups_to_run = ["sota", "benchmark"] if args.group == "all" else [args.group]
    all_results = {}

    for group_name in groups_to_run:
        config = GROUPS[group_name]
        if not config["path"].exists():
            print(f"WARNING: {config['path']} not found. Skipping {group_name}.")
            continue

        with open(config["path"]) as f:
            data = json.load(f)

        result = run_group(group_name, data, config)
        if result is not None:
            all_results[group_name] = result

    # Save summary
    if all_results:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = FIGURES_DIR / "cd_diagram_results.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")

    print("\nDone!")
