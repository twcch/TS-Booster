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
    """完整實驗結果視覺化"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def visualize_ablation(self, json_file):
        """視覺化 Ablation Study 結果"""
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        # 1. RMSE Progression
        self._plot_ablation_progression(results)
        
        # 2. Metrics Comparison
        self._plot_ablation_metrics(results)
        
        # 3. Improvement Analysis
        self._plot_ablation_improvement(results)
        
        print(f"📊 Ablation visualizations saved to {self.figures_dir}")
    
    def visualize_cross_backbone(self, json_file):
        """視覺化 Cross-Backbone 結果"""
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        # 1. Backbone Comparison
        self._plot_backbone_comparison(results)
        
        # 2. Improvement by Backbone
        self._plot_backbone_improvement(results)
        
        print(f"📊 Cross-backbone visualizations saved to {self.figures_dir}")
    
    def _plot_ablation_progression(self, results):
        """繪製 Ablation 進展圖"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = list(results.keys())
        rmse_values = [results[c]['RMSE'] for c in configs]
        
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#3498db']
        bars = ax.bar(range(len(configs)), rmse_values, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, rmse) in enumerate(zip(bars, rmse_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rmse:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Ablation Study: Component Contribution Analysis', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ablation_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_metrics(self, results):
        """繪製多指標比較"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        configs = list(results.keys())
        metrics = ['RMSE', 'MAE', 'R2', 'MDA']
        
        for ax, metric in zip(axes.flat, metrics):
            values = [results[c][metric] for c in configs]
            
            bars = ax.bar(range(len(configs)), values, alpha=0.7)
            
            # Color code: best is green
            if metric in ['RMSE', 'MAE']:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_color('#2ecc71')
            
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}' if metric != 'MDA' else f'{val:.1f}%',
                       ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Ablation Study: All Metrics Comparison', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ablation_all_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ablation_improvement(self, results):
        """繪製相對改善圖"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        configs = list(results.keys())
        baseline_rmse = results[configs[0]]['RMSE']
        
        improvements = []
        for config in configs:
            rmse = results[config]['RMSE']
            improvement = (baseline_rmse - rmse) / baseline_rmse * 100
            improvements.append(improvement)
        
        colors = ['red' if imp < 0 else 'green' for imp in improvements]
        bars = ax.barh(range(len(configs)), improvements, color=colors, alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('Improvement over Vanilla (%)', fontsize=12)
        ax.set_title('Ablation Study: RMSE Improvement Analysis', 
                    fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            width = bar.get_width()
            ax.text(width, i, f'{imp:+.2f}%',
                   ha='left' if imp > 0 else 'right', va='center', 
                   fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'ablation_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_backbone_comparison(self, results):
        """繪製 Backbone 比較圖"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        backbones = list(results.keys())
        x = np.arange(len(backbones))
        width = 0.35
        
        vanilla_rmse = [results[b]['vanilla']['RMSE'] for b in backbones]
        boosted_rmse = [results[b]['boosted']['RMSE'] for b in backbones]
        
        bars1 = ax.bar(x - width/2, vanilla_rmse, width, label='Vanilla', alpha=0.8, color='#e74c3c')
        bars2 = ax.bar(x + width/2, boosted_rmse, width, label='+ TS_Booster', alpha=0.8, color='#2ecc71')
        
        ax.set_xlabel('Backbone Architecture', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('Cross-Backbone Validation: TS_Booster Performance', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(backbones)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'backbone_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_backbone_improvement(self, results):
        """繪製各 Backbone 的改善幅度"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        backbones = list(results.keys())
        improvements = [results[b]['improvement'] for b in backbones]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(backbones, improvements, color=colors, alpha=0.7)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel('RMSE Improvement (%)', fontsize=12)
        ax.set_title('TS_Booster Improvement by Backbone', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{imp:+.2f}%',
                   ha='left' if imp > 0 else 'right', va='center',
                   fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'backbone_improvement.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主執行函數"""
    import argparse
    parser = argparse.ArgumentParser(description='Visualize comprehensive experiment results')
    parser.add_argument('--ablation', type=str, help='Path to ablation results JSON')
    parser.add_argument('--cross-backbone', type=str, help='Path to cross-backbone results JSON')
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
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
