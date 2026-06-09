"""
Phase 3: Integration — Unified Comparison Table + CD Diagram

Combines results from:
  - Phase 1: Persistence baseline
  - Phase 2: EWMA, HAR-RV, GARCH(1,1), GJR-GARCH
  - Main experiment: Linear vanilla, TSB (5-seed mean)

Outputs:
  1. Unified RMSE / MAE / MDA table (all 7 methods × 16 datasets)
  2. Average rank table
  3. CD diagram (Friedman + Nemenyi)

Usage:
  uv run python revision/phase3_integration/integrate.py
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy import stats
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

# ── Paths ────────────────────────────────────────────────────────────────────
P1_PATH  = Path("revision/phase1_persistence/persistence_results.json")
P2_PATH  = Path("revision/phase2_vol_baselines/vol_baselines_results.json")
DL_PATH  = Path("experiments/comprehensive_results/main_experiment/statistical_test_results.json")
OUT_DIR  = Path("revision/phase3_integration")

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

CATEGORIES = {
    "TWD FX":           ["USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD"],
    "International FX": ["EUR_USD", "GBP_USD", "USD_JPY"],
    "Taiwan Stocks":    ["0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"],
}

METHODS = ["Persistence", "EWMA", "HAR-RV", "GARCH", "GJR-GARCH", "Linear", "TSB"]


# ── Load data ─────────────────────────────────────────────────────────────────

def load_all() -> dict:
    p1  = {r["dataset"]: r for r in json.loads(P1_PATH.read_text())}
    p2  = json.loads(P2_PATH.read_text())
    dl  = json.loads(DL_PATH.read_text())

    unified = {}
    for ds in ALL_DATASETS:
        if ds not in p1 or ds not in p2 or ds not in dl:
            continue
        unified[ds] = {
            "Persistence": {"rmse": p1[ds]["per_rmse"], "mae": p1[ds]["per_mae"], "mda": p1[ds]["per_mda"]},
            "EWMA":        p2[ds]["ewma"],
            "HAR-RV":      p2[ds]["har_rv"],
            "GARCH":       p2[ds]["garch"],
            "GJR-GARCH":   p2[ds]["gjr_garch"],
            "Linear":      {"rmse": float(np.mean(dl[ds]["vanilla"]["rmse"])),
                            "mae":  float(np.mean(dl[ds]["vanilla"]["mae"])),
                            "mda":  float(np.mean(dl[ds]["vanilla"]["mda"]))},
            "TSB":         {"rmse": float(np.mean(dl[ds]["booster"]["rmse"])),
                            "mae":  float(np.mean(dl[ds]["booster"]["mae"])),
                            "mda":  float(np.mean(dl[ds]["booster"]["mda"]))},
        }
    return unified


# ── Tables ───────────────────────────────────────────────────────────────────

def print_table(unified: dict, metric: str = "rmse"):
    col_w = 9
    header = f"{'Dataset':<14} " + " ".join(f"{m:>{col_w}}" for m in METHODS)
    print(f"\n{'='*100}")
    print(f"Unified {metric.upper()} Comparison (all methods, test set)")
    print(f"{'='*100}")
    print(header)
    print("-" * 100)

    for ds in ALL_DATASETS:
        if ds not in unified:
            continue
        vals = [unified[ds][m][metric] for m in METHODS]
        best_idx = int(np.argmin(vals))
        row = f"{ds:<14} "
        for j, v in enumerate(vals):
            marker = "*" if j == best_idx else " "
            row += f"{v:>{col_w}.4f}{marker}"[: col_w + 1] + " " if j < len(vals) - 1 else f"{v:>{col_w}.4f}{marker}"
        # Rebuild cleaner
        parts = []
        for j, v in enumerate(vals):
            s = f"{v:{col_w}.4f}"
            if j == best_idx:
                s = s + "*"
            else:
                s = s + " "
            parts.append(s)
        print(f"{ds:<14} " + " ".join(parts))

    print("-" * 100)

    # Average row
    avgs = []
    for m in METHODS:
        v = np.mean([unified[ds][m][metric] for ds in ALL_DATASETS if ds in unified])
        avgs.append(v)
    best_avg = int(np.argmin(avgs))
    parts = []
    for j, v in enumerate(avgs):
        s = f"{v:{col_w}.4f}" + ("*" if j == best_avg else " ")
        parts.append(s)
    print(f"{'Average':<14} " + " ".join(parts))


def print_rank_table(unified: dict):
    """Average rank across 16 datasets (lower = better)."""
    n = len([ds for ds in ALL_DATASETS if ds in unified])
    rank_matrix = np.zeros((n, len(METHODS)))

    for i, ds in enumerate([ds for ds in ALL_DATASETS if ds in unified]):
        rmse_vals = np.array([unified[ds][m]["rmse"] for m in METHODS])
        ranks = stats.rankdata(rmse_vals)  # 1 = best
        rank_matrix[i] = ranks

    avg_ranks = rank_matrix.mean(axis=0)
    print(f"\n{'='*60}")
    print("Average RMSE Rank (1 = best)")
    print(f"{'='*60}")
    for m, r in sorted(zip(METHODS, avg_ranks), key=lambda x: x[1]):
        bar = "█" * int(r * 4)
        print(f"  {m:<12} {r:5.2f}  {bar}")

    return rank_matrix, avg_ranks


# ── CD Diagram ────────────────────────────────────────────────────────────────

def friedman_test(rank_matrix):
    n, k = rank_matrix.shape
    avg_ranks = rank_matrix.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(avg_ranks ** 2) - k * (k + 1) ** 2 / 4)
    p = 1 - stats.chi2.cdf(chi2, k - 1)
    return chi2, p


def nemenyi_cd(n, k, alpha=0.05):
    q_table = {2:1.960, 3:2.344, 4:2.569, 5:2.728, 6:2.850, 7:2.949, 8:3.031}
    q = q_table.get(k, stats.norm.ppf(1 - alpha / (k * (k - 1))))
    return q * np.sqrt(k * (k + 1) / (6 * n))


def draw_cd_diagram(avg_ranks, methods, cd, n, title, save_path):
    k = len(methods)
    order = np.argsort(avg_ranks)
    s_ranks   = avg_ranks[order]
    s_methods = [methods[i] for i in order]

    # Find cliques (not significantly different pairs)
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if abs(s_ranks[j] - s_ranks[i]) < cd:
                merged = False
                for cl in cliques:
                    if i in cl or j in cl:
                        if all(abs(s_ranks[j] - s_ranks[x]) < cd and
                               abs(s_ranks[i] - s_ranks[x]) < cd for x in cl):
                            cl.update([i, j])
                            merged = True
                            break
                if not merged:
                    cliques.append({i, j})

    # Merge overlapping cliques
    changed = True
    while changed:
        changed = False
        new_cliques = []
        used = set()
        for a, c1 in enumerate(cliques):
            if a in used:
                continue
            merged = c1.copy()
            for b, c2 in enumerate(cliques):
                if b <= a or b in used:
                    continue
                if merged & c2:
                    if all(abs(s_ranks[x] - s_ranks[y]) < cd
                           for x, y in combinations(merged | c2, 2)):
                        merged |= c2
                        used.add(b)
                        changed = True
            new_cliques.append(merged)
            used.add(a)
        cliques = new_cliques

    # Remove subset cliques
    cliques = [c for c in cliques
               if not any(c < other for other in cliques)]

    # Draw
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")

    # Rank axis
    ax.axhline(y=1.8, xmin=0.05, xmax=0.95, color="black", lw=1.5)
    for r in range(1, k + 1):
        ax.plot(r, 1.8, "|", color="black", markersize=8)
        ax.text(r, 2.05, str(r), ha="center", va="bottom", fontsize=9)
    ax.text((k + 1) / 2, 2.35, "Average Rank →", ha="center", va="bottom",
            fontsize=9, color="gray")

    # Method markers and labels
    colors = {"TSB": "#d62728", "HAR-RV": "#1f77b4", "Linear": "#ff7f0e"}
    for idx, (m, r) in enumerate(zip(s_methods, s_ranks)):
        col = colors.get(m, "#7f7f7f")
        ax.plot(r, 1.8, "o", color=col, markersize=9, zorder=3)
        va = "top" if idx % 2 == 0 else "bottom"
        y_off = 1.55 if va == "top" else 2.05
        ax.text(r, y_off, m, ha="center", va=va, fontsize=8.5,
                color=col, fontweight="bold" if m == "TSB" else "normal")

    # Clique bars
    bar_y = 1.2
    for ci, cl in enumerate(cliques):
        cl_list = sorted(cl)
        x0 = s_ranks[cl_list[0]] - 0.05
        x1 = s_ranks[cl_list[-1]] + 0.05
        y  = bar_y - ci * 0.2
        ax.plot([x0, x1], [y, y], color="#555555", lw=3.5, solid_capstyle="round")

    # CD bar (top-left)
    ax.annotate("", xy=(1 + cd, 0.4), xytext=(1, 0.4),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(1 + cd / 2, 0.55, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    unified = load_all()
    print(f"Loaded {len(unified)} datasets")

    # ── Unified tables ────────────────────────────────────────────────────
    print_table(unified, "rmse")
    print_table(unified, "mae")
    print_table(unified, "mda")

    # ── Rank table ────────────────────────────────────────────────────────
    rank_matrix, avg_ranks = print_rank_table(unified)

    # ── Friedman + Nemenyi ────────────────────────────────────────────────
    n, k = rank_matrix.shape
    chi2, p_val = friedman_test(rank_matrix)
    cd = nemenyi_cd(n, k)

    print(f"\n── Friedman Test ────────────────────────────────────────")
    print(f"  χ²({k-1}) = {chi2:.4f},  p = {p_val:.6f}")
    print(f"  Nemenyi CD (α=0.05, k={k}, n={n}) = {cd:.4f}")
    print(f"  Friedman significant: {'YES' if p_val < 0.05 else 'NO'}")

    print(f"\n── Pairwise significance (|rank_i - rank_j| > CD={cd:.2f}) ──")
    for i, j in combinations(range(k), 2):
        diff = abs(avg_ranks[i] - avg_ranks[j])
        sig = "**SIG**" if diff > cd else "n.s."
        print(f"  {METHODS[i]:<12} vs {METHODS[j]:<12}  diff={diff:.3f}  {sig}")

    # ── CD diagram ────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_cd_diagram(
        avg_ranks, METHODS, cd, n,
        title=f"Critical Difference Diagram — All Baselines (n={n}, α=0.05)",
        save_path=OUT_DIR / "cd_diagram_all_baselines.png",
    )

    # ── Save unified JSON ─────────────────────────────────────────────────
    summary = {
        "avg_ranks": {m: float(r) for m, r in zip(METHODS, avg_ranks)},
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_val),
        "nemenyi_cd": float(cd),
        "per_dataset": unified,
    }
    out_json = OUT_DIR / "integration_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
