"""
視覺化工具 - 完整實驗結果
==========================
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


class ComprehensiveResultsVisualizer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def visualize_ablation(self, json_file):
        import re
        json_path = Path(json_file)
        # Match both currency pairs (USD_TWD) and stock tickers (2330_TW)
        match = re.search(r'ablation_results_(.+?)_\d{8}_', json_path.name)
        currency_pair = match.group(1) if match else None

        with open(json_file, 'r') as f:
            data = json.load(f)
        results = data.get('results', data)

        self._plot_ablation_progression(results, currency_pair)
        self._plot_ablation_metrics(results, currency_pair)
        self._plot_ablation_improvement(results, currency_pair)

        if currency_pair:
            print(f"📊 Ablation visualizations for {currency_pair} saved to {self.figures_dir}")
        else:
            print(f"📊 Ablation visualizations saved to {self.figures_dir}")

    def visualize_cross_backbone(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        results = data.get('results', data)

        self._plot_backbone_comparison(results)
        self._plot_backbone_improvement(results)
        print(f"📊 Cross-backbone visualizations saved to {self.figures_dir}")

    def _plot_ablation_progression(self, results, currency_pair=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        configs = list(results.keys())
        rmse_values = [results[c]['RMSE'] for c in configs]

        colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']
        if len(configs) > len(colors):
            colors = colors + ['#9b59b6'] * (len(configs) - len(colors))
        bars = ax.bar(range(len(configs)), rmse_values, color=colors[:len(configs)], alpha=0.8)

        for bar, rmse in zip(bars, rmse_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{rmse:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        title = 'Ablation Study: Component Contribution'
        if currency_pair: title += f' ({currency_pair})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'ablation_progression_{currency_pair}.png' if currency_pair else 'ablation_progression.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_ablation_metrics(self, results, currency_pair=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        configs = list(results.keys())
        metrics = ['RMSE', 'MAE', 'R2', 'MDA']

        for ax, metric in zip(axes.flat, metrics):
            values = [results[c][metric] for c in configs]
            bars = ax.bar(range(len(configs)), values, alpha=0.7)

            best_idx = np.argmin(values) if metric in ['RMSE', 'MAE'] else np.argmax(values)
            bars[best_idx].set_color('#2ecc71')

            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}' if metric != 'MDA' else f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=9)

        title = 'Ablation Study: All Metrics'
        if currency_pair: title += f' ({currency_pair})'
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        filename = f'ablation_all_metrics_{currency_pair}.png' if currency_pair else 'ablation_all_metrics.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_ablation_improvement(self, results, currency_pair=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = list(results.keys())
        baseline_rmse = results[configs[0]]['RMSE']

        improvements = [(baseline_rmse - results[c]['RMSE']) / baseline_rmse * 100 for c in configs]
        colors = ['red' if imp < 0 else 'green' for imp in improvements]
        bars = ax.barh(range(len(configs)), improvements, color=colors, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Improvement over Vanilla (%)', fontsize=12)
        title = 'RMSE Improvement Analysis'
        if currency_pair: title += f' ({currency_pair})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs)
        ax.grid(axis='x', alpha=0.3)

        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax.text(bar.get_width(), i, f'{imp:+.2f}%',
                   ha='left' if imp > 0 else 'right', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = f'ablation_improvement_{currency_pair}.png' if currency_pair else 'ablation_improvement.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_backbone_comparison(self, results):
        fig, ax = plt.subplots(figsize=(12, 6))
        backbones = list(results.keys())
        x = np.arange(len(backbones))
        width = 0.35

        vanilla_rmse = [results[b]['vanilla']['RMSE'] for b in backbones]
        boosted_rmse = [results[b]['boosted']['RMSE'] for b in backbones]

        bars1 = ax.bar(x - width/2, vanilla_rmse, width, label='Vanilla', alpha=0.8, color='#e74c3c')
        bars2 = ax.bar(x + width/2, boosted_rmse, width, label='+ TS-Booster', alpha=0.8, color='#2ecc71')

        ax.set_xlabel('Backbone Architecture', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Cross-Backbone: TS-Booster Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(backbones)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'backbone_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_backbone_improvement(self, results):
        fig, ax = plt.subplots(figsize=(10, 6))
        backbones = list(results.keys())
        improvements = [results[b]['improvement'] for b in backbones]

        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(backbones, improvements, color=colors, alpha=0.7)

        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('RMSE Improvement (%)', fontsize=12)
        ax.set_title('TS-Booster Improvement by Backbone', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, imp in zip(bars, improvements):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2.,
                   f'{imp:+.2f}%', ha='left' if imp > 0 else 'right', va='center',
                   fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'backbone_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_cross_dataset(self, json_files):
        import re

        all_data = {}
        for f in json_files:
            with open(f, 'r') as fh:
                data = json.load(fh)
            meta = data.get('metadata', {})
            pair = meta.get('currency_pair', Path(f).stem)
            results = data.get('results', data)
            all_data[pair] = results

        if len(all_data) < 2:
            print("⚠️ Need at least 2 datasets for cross-dataset comparison")
            return

        datasets = list(all_data.keys())
        all_configs = [set(all_data[d].keys()) for d in datasets]
        configs = list(all_configs[0].intersection(*all_configs[1:]))
        # Sort in logical order
        config_order = ['Vanilla', '+Decomp', '+Expert', 'TS-Booster']
        configs = [c for c in config_order if c in configs]
        # Add any remaining configs not in the predefined order
        for c in sorted(all_configs[0].intersection(*all_configs[1:])):
            if c not in configs:
                configs.append(c)

        self._plot_cross_dataset_heatmap(all_data, datasets, configs)
        self._plot_cross_dataset_improvement(all_data, datasets, configs)
        self._plot_multiscale_weights_heatmap(all_data, datasets)

        print(f"📊 Cross-dataset visualizations saved to {self.figures_dir}")

    def _plot_cross_dataset_heatmap(self, all_data, datasets, configs):
        fig, ax = plt.subplots(figsize=(max(10, len(configs)*2), max(5, len(datasets)*0.8)))

        rmse_matrix = np.zeros((len(datasets), len(configs)))
        for i, ds in enumerate(datasets):
            for j, cfg in enumerate(configs):
                rmse_matrix[i, j] = all_data[ds][cfg]['RMSE']

        row_min = rmse_matrix.min(axis=1, keepdims=True)
        row_max = rmse_matrix.max(axis=1, keepdims=True)
        normalized = (rmse_matrix - row_min) / (row_max - row_min + 1e-8)

        im = ax.imshow(normalized, cmap='RdYlGn_r', aspect='auto')

        for i in range(len(datasets)):
            best_j = np.argmin(rmse_matrix[i])
            for j in range(len(configs)):
                val = rmse_matrix[i, j]
                weight = 'bold' if j == best_j else 'normal'
                color = 'white' if normalized[i, j] > 0.7 or normalized[i, j] < 0.3 else 'black'
                text = f'{val:.3f}'
                if j == best_j: text += ' ★'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                       fontweight=weight, color=color)

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=30, ha='right', fontsize=10)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=11)
        ax.set_title('RMSE Heatmap (★ = best per dataset)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Normalized RMSE (green=best)', shrink=0.8)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cross_dataset_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_cross_dataset_improvement(self, all_data, datasets, configs):
        fig, ax = plt.subplots(figsize=(max(12, len(datasets)*2), 7))

        x = np.arange(len(datasets))
        non_vanilla = [c for c in configs if c != 'Vanilla']
        width = 0.8 / max(1, len(non_vanilla))
        colors = ['#e67e22', '#3498db', '#2ecc71', '#9b59b6']

        for k, cfg in enumerate(non_vanilla):
            improvements = []
            for ds in datasets:
                baseline = all_data[ds]['Vanilla']['RMSE']
                rmse = all_data[ds][cfg]['RMSE']
                improvements.append((baseline - rmse) / baseline * 100)
            offset = (k - len(non_vanilla)/2 + 0.5) * width
            ax.bar(x + offset, improvements, width * 0.9, label=cfg,
                  color=colors[k % len(colors)], alpha=0.85)

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('RMSE Improvement over Vanilla (%)', fontsize=12)
        ax.set_title('Component Contribution Across Datasets', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cross_dataset_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_multiscale_weights_heatmap(self, all_data, datasets):
        """MultiScale Decomp learned weights heatmap across datasets"""
        ms_data = {}
        # Look for multiscale_weights in TS-Booster config
        for ds in datasets:
            for cfg_name in ['TS-Booster', 'Full (DNAS)', '+MultiScale+Experts', '+Decomp']:
                if cfg_name in all_data[ds]:
                    weights = all_data[ds][cfg_name].get('multiscale_weights', {})
                    if weights:
                        ms_data[ds] = weights
                        break

        if len(ms_data) < 2:
            return

        all_kernels = set()
        for w in ms_data.values():
            all_kernels.update(w.keys())
        kernels = sorted(all_kernels, key=lambda k: int(k.split('_')[1]) if k.startswith('MA_') else int(k) if k.isdigit() else 999)

        fig, ax = plt.subplots(figsize=(max(8, len(kernels)*1.5), max(4, len(ms_data)*0.8)))

        matrix = np.zeros((len(ms_data), len(kernels)))
        ds_names = list(ms_data.keys())
        for i, ds in enumerate(ds_names):
            for j, k in enumerate(kernels):
                matrix[i, j] = ms_data[ds].get(k, 0.0)

        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        for i in range(len(ds_names)):
            for j in range(len(kernels)):
                val = matrix[i, j]
                color = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=10,
                       fontweight='bold' if val > 0.4 else 'normal', color=color)

        ax.set_xticks(range(len(kernels)))
        ax.set_xticklabels(kernels, fontsize=11)
        ax.set_yticks(range(len(ds_names)))
        ax.set_yticklabels(ds_names, fontsize=11)
        ax.set_title('Learned Decomposition Weights Across Datasets', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Weight', shrink=0.8)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'cross_dataset_multiscale_weights.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    import argparse
    import glob as glob_mod
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--ablation', type=str, help='Path to ablation results JSON')
    parser.add_argument('--cross-backbone', type=str, help='Path to cross-backbone results JSON')
    parser.add_argument('--cross-dataset', action='store_true', help='Generate cross-dataset comparison')
    parser.add_argument('--results-dir', type=str, default='experiments/comprehensive_results',
                       help='Results directory')
    args = parser.parse_args()

    visualizer = ComprehensiveResultsVisualizer(args.results_dir)

    if args.ablation:
        print(f"\n📊 Visualizing ablation results from {args.ablation}...")
        visualizer.visualize_ablation(args.ablation)

    if args.cross_backbone:
        print(f"\n📊 Visualizing cross-backbone results from {args.cross_backbone}...")
        visualizer.visualize_cross_backbone(args.cross_backbone)

    if args.cross_dataset:
        results_dir = Path(args.results_dir)
        all_files = sorted(results_dir.glob('ablation_results_*.json'), reverse=True)
        seen = set()
        latest_files = []
        import re
        for f in all_files:
            match = re.search(r'ablation_results_(.+?)_\d{8}_', f.name)
            if match:
                pair = match.group(1)
                if pair not in seen:
                    seen.add(pair)
                    latest_files.append(str(f))
        if latest_files:
            print(f"\n📊 Generating cross-dataset comparison ({len(latest_files)} datasets)...")
            visualizer.visualize_cross_dataset(latest_files)
        else:
            print("⚠️ No ablation results found for cross-dataset comparison")

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
