"""
Interpretability Analysis for TS-Booster Paper.

Extracts and visualizes:
1. Learned MultiScaleDecomp weights (kernel 7 vs 13) per dataset
2. Learned SNR gate parameters (w, b) per dataset
3. Gate activation distribution per sample
4. Prediction curves: Vanilla vs TS-Booster vs Ground Truth

Usage:
  uv run python experiments/interpretability.py
"""
import sys, random, json
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data

ALL_DATASETS = [
    "USD_TWD", "JPY_TWD", "KRW_TWD", "CAD_TWD", "CNY_TWD", "EUR_TWD", "THB_TWD",
    "EUR_USD", "GBP_USD", "USD_JPY",
    "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW",
]

CATEGORIES = {
    "0050_TW": "Stock", "2317_TW": "Stock", "2330_TW": "Stock",
    "2454_TW": "Stock", "2603_TW": "Stock", "2881_TW": "Stock",
    "USD_TWD": "TWD FX", "JPY_TWD": "TWD FX", "KRW_TWD": "TWD FX",
    "CAD_TWD": "TWD FX", "CNY_TWD": "TWD FX", "EUR_TWD": "TWD FX", "THB_TWD": "TWD FX",
    "EUR_USD": "Intl FX", "GBP_USD": "Intl FX", "USD_JPY": "Intl FX",
}

SEED = 42
OUT_DIR = Path("experiments/comprehensive_results/interpretability")
FIGURES_DIR = OUT_DIR / "figures"

RESULTS_PATH = OUT_DIR / "interpretability_results.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_and_extract(ds, epochs=100):
    """Train TS-Booster and extract interpretability data."""
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

    # --- Train Vanilla ---
    set_seed(SEED)
    vanilla = FlexibleTSBooster(
        base_model=LinearBackbone(seq_len, 2, pred_len),
        use_decomp=False, use_experts=False, **base_kwargs
    )
    vanilla = train_v11(
        train_loader, val_loader, test_loader, device=device,
        horizon=pred_len, config=config, seq_len=seq_len, verbose=False, model=vanilla
    )

    # --- Train TS-Booster ---
    set_seed(SEED)
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

    # === Extract learned parameters ===
    booster.eval()
    vanilla.eval()

    # 1. Decomp weights
    decomp_weights = booster.get_multiscale_weights()
    # Force a forward pass to populate _last_weights if needed
    if decomp_weights is None:
        with torch.no_grad():
            for batch in test_loader:
                x = batch["raw_input"].to(device)
                _ = booster(x)
                break
        decomp_weights = booster.get_multiscale_weights()

    # 2. SNR gate parameters
    snr_w = booster.snr_gate_w.item() if hasattr(booster, 'snr_gate_w') else None
    snr_b = booster.snr_gate_b.item() if hasattr(booster, 'snr_gate_b') else None

    # 3. Gate activations + predictions on test set
    gate_values = []
    snr_values = []
    vanilla_preds = []
    booster_preds = []
    targets_all = []
    inputs_all = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["raw_input"].to(device)
            y = batch["target"].to(device)

            # Vanilla prediction
            v_out = vanilla(x)
            v_pred = v_out[0]

            # Booster prediction + gate extraction
            b_out = booster(x)
            b_pred = b_out[0]
            trend_part = b_out[3]
            seasonal_part = b_out[4]

            # Compute gate values per sample
            seasonal_std = seasonal_part.std(dim=1).mean(dim=-1)  # [B]
            raw_std = x.std(dim=1).mean(dim=-1)  # [B]
            snr = seasonal_std / (raw_std + 1e-8)  # [B]
            gate = torch.sigmoid(snr_w * snr + snr_b)  # [B]

            gate_values.extend(gate.cpu().numpy().tolist())
            snr_values.extend(snr.cpu().numpy().tolist())
            vanilla_preds.extend(v_pred.cpu().numpy().tolist())
            booster_preds.extend(b_pred.cpu().numpy().tolist())
            targets_all.extend(y.cpu().numpy().tolist())
            inputs_all.extend(x[:, -1, 0].cpu().numpy().tolist())  # last input value

    # Inverse scale predictions and targets
    if scaler:
        vanilla_preds_scaled = np.array(vanilla_preds) * scaler.scale_[0] + scaler.mean_[0]
        booster_preds_scaled = np.array(booster_preds) * scaler.scale_[0] + scaler.mean_[0]
        targets_scaled = np.array(targets_all) * scaler.scale_[0] + scaler.mean_[0]
        inputs_scaled = np.array(inputs_all) * scaler.scale_[0] + scaler.mean_[0]
    else:
        vanilla_preds_scaled = np.array(vanilla_preds)
        booster_preds_scaled = np.array(booster_preds)
        targets_scaled = np.array(targets_all)
        inputs_scaled = np.array(inputs_all)

    return {
        "decomp_weights": decomp_weights,
        "snr_w": snr_w,
        "snr_b": snr_b,
        "gate_values": gate_values,
        "snr_values": snr_values,
        "vanilla_preds": vanilla_preds_scaled.tolist(),
        "booster_preds": booster_preds_scaled.tolist(),
        "targets": targets_scaled.tolist(),
        "inputs": inputs_scaled.tolist(),
    }


def plot_decomp_weights(all_results):
    """Bar chart of learned decomp weights across datasets."""
    datasets = []
    w7_vals = []
    w13_vals = []
    colors = []
    cat_colors = {"Stock": "#2196F3", "TWD FX": "#FF9800", "Intl FX": "#4CAF50"}

    for ds in ALL_DATASETS:
        if ds not in all_results or all_results[ds]["decomp_weights"] is None:
            continue
        dw = all_results[ds]["decomp_weights"]
        datasets.append(ds)
        w7_vals.append(dw.get("7", 0))
        w13_vals.append(dw.get("13", 0))
        colors.append(cat_colors[CATEGORIES[ds]])

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, w7_vals, width, label='Kernel 7 (short-term)', color='#42A5F5', edgecolor='white')
    bars2 = ax.bar(x + width/2, w13_vals, width, label='Kernel 13 (mid-term)', color='#EF5350', edgecolor='white')

    ax.set_ylabel('Softmax Weight', fontsize=12)
    ax.set_title('Learned Decomposition Weights (Kernel 7 vs 13)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal weight')

    # Category background shading
    stock_end = sum(1 for d in ALL_DATASETS[:6] if d in all_results)
    fx_end = stock_end + sum(1 for d in ALL_DATASETS[6:13] if d in all_results)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decomp_weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: decomp_weights.png")


def plot_snr_gate_params(all_results):
    """Bar chart of learned SNR gate w and b across datasets."""
    datasets = []
    w_vals = []
    b_vals = []

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        datasets.append(ds)
        w_vals.append(all_results[ds]["snr_w"])
        b_vals.append(all_results[ds]["snr_b"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # w values
    cat_colors_list = [{"Stock": "#2196F3", "TWD FX": "#FF9800", "Intl FX": "#4CAF50"}[CATEGORIES[d]] for d in datasets]
    ax1.bar(range(len(datasets)), w_vals, color=cat_colors_list, edgecolor='white')
    ax1.set_xticks(range(len(datasets)))
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Learned w', fontsize=12)
    ax1.set_title('SNR Gate Weight (w)', fontsize=13)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # b values
    ax2.bar(range(len(datasets)), b_vals, color=cat_colors_list, edgecolor='white')
    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Learned b', fontsize=12)
    ax2.set_title('SNR Gate Bias (b)', fontsize=13)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Stock'),
        Patch(facecolor='#FF9800', label='TWD FX'),
        Patch(facecolor='#4CAF50', label='Intl FX'),
    ]
    ax1.legend(handles=legend_elements, fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "snr_gate_params.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: snr_gate_params.png")


def plot_gate_distributions(all_results):
    """Violin/box plot of gate activation distributions across datasets."""
    fig, ax = plt.subplots(figsize=(14, 5))

    data_for_plot = []
    labels = []
    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        gates = all_results[ds]["gate_values"]
        data_for_plot.append(gates)
        labels.append(ds)

    parts = ax.violinplot(data_for_plot, positions=range(len(labels)),
                          showmeans=True, showmedians=True, showextrema=False)

    # Color by category
    cat_colors = {"Stock": "#2196F3", "TWD FX": "#FF9800", "Intl FX": "#4CAF50"}
    for i, (pc, ds) in enumerate(zip(parts['bodies'], labels)):
        pc.set_facecolor(cat_colors[CATEGORIES[ds]])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Gate Activation', fontsize=12)
    ax.set_title('SNR Gate Activation Distribution per Dataset (test set)', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', label='Stock'),
        Patch(facecolor='#FF9800', label='TWD FX'),
        Patch(facecolor='#4CAF50', label='Intl FX'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gate_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gate_distributions.png")


def plot_prediction_curves(all_results, showcase_datasets=None):
    """Plot prediction curves for selected datasets."""
    if showcase_datasets is None:
        # Pick representative datasets: best stock, best FX, worst case
        showcase_datasets = ["2330_TW", "EUR_USD", "USD_TWD", "GBP_USD"]

    for ds in showcase_datasets:
        if ds not in all_results:
            continue

        r = all_results[ds]
        targets = np.array(r["targets"])
        vanilla_p = np.array(r["vanilla_preds"])
        booster_p = np.array(r["booster_preds"])

        # Use first prediction step (horizon=0) for clarity
        if targets.ndim == 2:
            targets_1 = targets[:, 0]
            vanilla_1 = vanilla_p[:, 0]
            booster_1 = booster_p[:, 0]
        else:
            targets_1 = targets
            vanilla_1 = vanilla_p
            booster_1 = booster_p

        n = len(targets_1)
        x_axis = np.arange(n)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x_axis, targets_1, color='black', linewidth=1.5, label='Ground Truth', alpha=0.8)
        ax.plot(x_axis, vanilla_1, color='#EF5350', linewidth=1.0, label='Vanilla Linear', alpha=0.7, linestyle='--')
        ax.plot(x_axis, booster_1, color='#2196F3', linewidth=1.0, label='TS-Booster', alpha=0.7)

        ax.set_xlabel('Test Sample Index', fontsize=11)
        ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
        ax.set_title(f'{ds} — Prediction Comparison (1-step ahead)', fontsize=13)
        ax.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"pred_curve_{ds}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: pred_curve_{ds}.png")


def plot_gate_vs_improvement(all_results):
    """Scatter plot: mean gate activation vs RMSE improvement per dataset."""
    datasets = []
    mean_gates = []
    improvements = []
    cats = []

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        targets = np.array(r["targets"])
        vanilla_p = np.array(r["vanilla_preds"])
        booster_p = np.array(r["booster_preds"])

        v_rmse = np.sqrt(np.mean((targets - vanilla_p) ** 2))
        b_rmse = np.sqrt(np.mean((targets - booster_p) ** 2))
        gap = (v_rmse - b_rmse) / v_rmse * 100

        datasets.append(ds)
        mean_gates.append(np.mean(r["gate_values"]))
        improvements.append(gap)
        cats.append(CATEGORIES[ds])

    fig, ax = plt.subplots(figsize=(8, 6))
    cat_colors = {"Stock": "#2196F3", "TWD FX": "#FF9800", "Intl FX": "#4CAF50"}

    for cat in ["Stock", "TWD FX", "Intl FX"]:
        mask = [c == cat for c in cats]
        x = [mean_gates[i] for i, m in enumerate(mask) if m]
        y = [improvements[i] for i, m in enumerate(mask) if m]
        labels = [datasets[i] for i, m in enumerate(mask) if m]
        ax.scatter(x, y, c=cat_colors[cat], s=80, label=cat, edgecolors='white', zorder=3)
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(lab, (xi, yi), fontsize=8, ha='left', va='bottom',
                       xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('Mean Gate Activation', fontsize=12)
    ax.set_ylabel('RMSE Improvement (%)', fontsize=12)
    ax.set_title('Gate Activation vs Improvement', fontsize=14)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gate_vs_improvement.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: gate_vs_improvement.png")


def print_summary_table(all_results):
    """Print a summary table of learned parameters."""
    print(f"\n{'='*90}")
    print("LEARNED PARAMETERS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Dataset':<12} | {'Category':>8} | {'W_7':>6} | {'W_13':>6} | {'SNR w':>7} | {'SNR b':>7} | "
          f"{'Gate mean':>9} | {'Gate std':>8} | {'Gate med':>8}")
    print("-" * 95)

    for ds in ALL_DATASETS:
        if ds not in all_results:
            continue
        r = all_results[ds]
        dw = r["decomp_weights"] or {}
        gates = np.array(r["gate_values"])
        cat = CATEGORIES[ds]

        print(f"  {ds:<12} | {cat:>8} | {dw.get('7', 0):>6.3f} | {dw.get('13', 0):>6.3f} | "
              f"{r['snr_w']:>7.3f} | {r['snr_b']:>7.3f} | "
              f"{np.mean(gates):>9.4f} | {np.std(gates):>8.4f} | {np.median(gates):>8.4f}")

    # Category averages
    print("-" * 95)
    for cat in ["Stock", "TWD FX", "Intl FX"]:
        w7s, w13s, ws, bs, gate_means = [], [], [], [], []
        for ds in ALL_DATASETS:
            if ds not in all_results or CATEGORIES[ds] != cat:
                continue
            r = all_results[ds]
            dw = r["decomp_weights"] or {}
            w7s.append(dw.get("7", 0))
            w13s.append(dw.get("13", 0))
            ws.append(r["snr_w"])
            bs.append(r["snr_b"])
            gate_means.append(np.mean(r["gate_values"]))
        if w7s:
            print(f"  {'Avg '+cat:<12} | {'':>8} | {np.mean(w7s):>6.3f} | {np.mean(w13s):>6.3f} | "
                  f"{np.mean(ws):>7.3f} | {np.mean(bs):>7.3f} | "
                  f"{np.mean(gate_means):>9.4f} |          |")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Check for saved results
    if RESULTS_PATH.exists():
        print(f"Loading saved results from {RESULTS_PATH}")
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}
        for ds in ALL_DATASETS:
            print(f"\n{'='*60}")
            print(f"Dataset: {ds}")
            print(f"{'='*60}")
            result = train_and_extract(ds)
            if result is not None:
                all_results[ds] = result
                # Save incrementally
                with open(RESULTS_PATH, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # === Analysis & Plots ===
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")

    print_summary_table(all_results)
    plot_decomp_weights(all_results)
    plot_snr_gate_params(all_results)
    plot_gate_distributions(all_results)
    plot_prediction_curves(all_results)
    plot_gate_vs_improvement(all_results)

    print(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
