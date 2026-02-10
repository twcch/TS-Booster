"""
Oracle Kernel Search - Ground Truth Experiment
===============================================
测试所有候选 decomposition kernel 的真实性能，验证 DNAS 的选择是否正确。

配置：
- Fixed: use_decomp=True, use_experts=True, use_dnas=False
- Variable: fixed_decomp_kernel in [7, 13, 25, 49]
"""

import sys
import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path.cwd()))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_random_seed(seed=42):
    """设置随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class OracleKernelSearch:
    """Oracle Kernel Search - 暴力测试所有 kernel"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.output_dir = Path('experiments/oracle_results')
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"🔮 Oracle Kernel Search initialized on {device}")

    def run_oracle_search(
        self,
        dataset_path='dataset/USD_TWD.csv',
        epochs=100,
        kernels_to_test=[7, 13, 25, 49],
        quick_test=False
    ):
        """
        Oracle Search: 测试所有 kernel，找出真正的最优解

        Args:
            dataset_path: 资料集路径
            epochs: 训练回合数
            kernels_to_test: 要测试的 kernel 列表
            quick_test: 是否为快速测试模式
        """
        # Extract currency pair
        currency_pair = Path(dataset_path).stem

        if quick_test:
            epochs = 20
            print("⚡ Quick Test Mode: epochs reduced to 20")

        print("\n" + "="*80)
        print(f"🔮 Oracle Kernel Search: Finding Ground Truth ({currency_pair})")
        print("="*80 + "\n")

        # Load data
        df = pd.read_csv(dataset_path)
        train_loader, val_loader, test_loader, scaler = prepare_data(
            df, lookback=30, horizon=3, batch_size=32
        )

        config = BEST_STAGE3_CONFIG.copy()
        config['epochs'] = epochs
        config['input_channels'] = 2

        seq_len = 30
        pred_len = 3

        oracle_results = {}

        for i, kernel in enumerate(kernels_to_test, 1):
            print(f"\n{'='*80}")
            print(f"🔍 [{i}/{len(kernels_to_test)}] Testing Fixed Kernel: MA_{kernel}")
            print(f"{'='*80}\n")

            # Set seed for reproducibility
            set_random_seed(42 + i)

            # Create backbone
            backbone = LinearBackbone(seq_len, 2, pred_len)

            # Create model with fixed kernel
            model = FlexibleTSBooster(
                base_model=backbone,
                seq_len=seq_len,
                pred_len=pred_len,
                input_channels=2,
                use_decomp=True,
                use_experts=True,
                use_gating=False,
                use_dnas=False,  # Important: disable DNAS
                fixed_decomp_kernel=kernel,  # Use fixed kernel
                decomp_dnas_kernels=config.get('decomp_dnas_kernels', [7, 13, 25, 49, 'identity']),
                trend_hidden_dim=config.get('trend_hidden_dim', 32),
                seasonal_hidden_dim=config.get('seasonal_hidden_dim', 32),
                dropout_p=config.get('trend_dropout', 0.3)
            )

            # Train
            trained_model = train_v11(
                train_loader, val_loader, test_loader,
                device=self.device, horizon=pred_len, config=config,
                seq_len=seq_len, verbose=True, model=model
            )

            # Evaluate
            metrics = self._evaluate_model(
                trained_model, test_loader, scaler, f"MA_{kernel}"
            )

            oracle_results[f"MA_{kernel}"] = metrics

            print(f"\n✅ MA_{kernel} Complete!")
            print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")

        # Analyze results
        self._analyze_oracle_results(oracle_results, currency_pair)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_oracle_results(oracle_results, currency_pair, timestamp)

        return oracle_results

    def _evaluate_model(self, model, test_loader, scaler, model_name):
        """评估模型性能"""
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["raw_input"].to(self.device)
                y = batch["target"].cpu().numpy()

                pred = model(x)[0].cpu().numpy()
                predictions.append(pred)
                actuals.append(y)

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        # Flatten for metrics
        pred_flat = predictions.flatten()
        actual_flat = actuals.flatten()

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
        mae = mean_absolute_error(actual_flat, pred_flat)
        r2 = r2_score(actual_flat, pred_flat)

        # MDA (Mean Directional Accuracy)
        if len(actual_flat) > 1:
            actual_direction = np.sign(np.diff(actual_flat))
            pred_direction = np.sign(np.diff(pred_flat))
            mda = np.mean(actual_direction == pred_direction) * 100
        else:
            mda = 0.0

        return {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "MDA": float(mda),
            "model_name": model_name
        }

    def _analyze_oracle_results(self, results, currency_pair):
        """分析 Oracle 结果"""
        print("\n" + "="*80)
        print(f"📊 Oracle Results Analysis ({currency_pair})")
        print("="*80 + "\n")

        # Sort by RMSE (lower is better)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])

        print("Ranking by RMSE (Best → Worst):")
        print("-" * 60)
        for rank, (kernel, metrics) in enumerate(sorted_results, 1):
            marker = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            print(f"{marker} Rank {rank}: {kernel:12s} | RMSE={metrics['RMSE']:.4f} | "
                  f"MAE={metrics['MAE']:.4f} | R2={metrics['R2']:.4f}")

        best_kernel = sorted_results[0][0]
        best_rmse = sorted_results[0][1]['RMSE']

        print(f"\n🎯 Ground Truth: {best_kernel} is the optimal kernel (RMSE={best_rmse:.4f})")

        # Calculate improvement over worst
        worst_rmse = sorted_results[-1][1]['RMSE']
        improvement = (worst_rmse - best_rmse) / worst_rmse * 100
        print(f"📈 Improvement: {improvement:.2f}% better than worst kernel")

    def _save_oracle_results(self, results, currency_pair, timestamp):
        """保存 Oracle 结果"""
        # Save JSON
        json_path = self.output_dir / f"oracle_results_{currency_pair}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                "metadata": {
                    "currency_pair": currency_pair,
                    "timestamp": timestamp,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "results": results
            }, f, indent=2)

        print(f"\n💾 Results saved to: {json_path}")

        # Save Markdown Report
        self._save_oracle_report(results, currency_pair, timestamp)

    def _save_oracle_report(self, results, currency_pair, timestamp):
        """生成 Markdown 报告"""
        report_path = self.output_dir / f"oracle_report_{currency_pair}_{timestamp}.md"

        # Sort by RMSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])

        report = [
            f"# Oracle Kernel Search Report - {currency_pair}",
            f"",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## Objective",
            f"",
            f"Find the **ground truth optimal kernel** for decomposition by exhaustively testing all candidates.",
            f"",
            f"## Configuration",
            f"",
            f"- **Model**: FlexibleTSBooster",
            f"- **Setup**: use_decomp=True, use_experts=True, use_dnas=False",
            f"- **Tested Kernels**: {', '.join([k for k, _ in results.items()])}",
            f"",
            f"## Results",
            f"",
            f"### Ranking by RMSE (Lower is Better)",
            f"",
            f"| Rank | Kernel | RMSE | MAE | R² | MDA (%) |",
            f"|------|--------|------|-----|-------|---------|"
        ]

        for rank, (kernel, metrics) in enumerate(sorted_results, 1):
            marker = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
            report.append(
                f"| {rank} {marker} | {kernel} | {metrics['RMSE']:.4f} | "
                f"{metrics['MAE']:.4f} | {metrics['R2']:.4f} | {metrics['MDA']:.1f} |"
            )

        best_kernel = sorted_results[0][0]
        best_rmse = sorted_results[0][1]['RMSE']
        worst_rmse = sorted_results[-1][1]['RMSE']
        improvement = (worst_rmse - best_rmse) / worst_rmse * 100

        report.extend([
            f"",
            f"## Conclusion",
            f"",
            f"🎯 **Ground Truth Optimal Kernel**: `{best_kernel}`",
            f"",
            f"- **Best RMSE**: {best_rmse:.4f}",
            f"- **Improvement over worst**: {improvement:.2f}%",
            f"",
            f"## Implications for DNAS",
            f"",
            f"Compare DNAS's selection with this ground truth:",
            f"",
            f"- If DNAS selected `{best_kernel}`, it found the optimal solution ✅",
            f"- If DNAS selected a different kernel, investigate why:",
            f"  - Over-fitting to training set?",
            f"  - Entropy penalty interference?",
            f"  - Insufficient search epochs?",
            f"",
            f"---",
            f"*Generated by Oracle Kernel Search Experiment*"
        ])

        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"📄 Report saved to: {report_path}")


def main():
    """主执行函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Oracle Kernel Search Experiment')
    parser.add_argument('--dataset', type=str, default='dataset/USD_TWD.csv',
                       help='Dataset path')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (20 epochs)')
    parser.add_argument('--kernels', type=int, nargs='+', default=[7, 13, 25, 49],
                       help='Kernels to test (default: 7 13 25 49)')
    args = parser.parse_args()

    # Set random seed
    set_random_seed(42)

    # Run oracle search
    oracle = OracleKernelSearch()
    results = oracle.run_oracle_search(
        dataset_path=args.dataset,
        epochs=args.epochs,
        kernels_to_test=args.kernels,
        quick_test=args.quick_test
    )

    print("\n✅ Oracle Kernel Search Complete!")
    print(f"📁 Results saved to: experiments/oracle_results/")


if __name__ == "__main__":
    main()
