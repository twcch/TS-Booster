"""
完整的 TS-Booster 評估實驗
===========================
包含:
1. Ablation Study: 測試每個組件的貢獻
2. Cross-Backbone Validation: 測試跨 Backbone 的泛化能力
"""

import torch
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import copy
import random
import glob

sys.path.append(str(Path.cwd()))

from src.models.layers import LinearBackbone, LSTMBackbone, GRUBackbone, TransformerBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.engine.trainer import train_v11
from src.configs.model_config import BEST_STAGE3_CONFIG
from src.utils.data_loader import prepare_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_random_seed(seed=42):
    """設置隨機種子以確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed set to: {seed}")


class ComprehensiveTSBoosterExperiment:
    """
    完整的 TS-Booster 評估實驗
    """
    def __init__(self, output_dir="experiments/comprehensive_results", clean_old_results=True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 清理舊結果
        if clean_old_results:
            self._clean_old_results()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Device: {self.device}")
        
        self.results = {}
    
    def _clean_old_results(self):
        """清理舊的實驗結果檔案"""
        patterns = [
            'ablation_results_*.json',
            'ablation_report_*.md',
            'cross_backbone_results_*.json',
            'cross_backbone_report_*.md'
        ]
        
        cleaned_count = 0
        for pattern in patterns:
            files = glob.glob(str(self.output_dir / pattern))
            for file in files:
                try:
                    Path(file).unlink()
                    cleaned_count += 1
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete {file}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old result files")
        
    def run_ablation_study(self, dataset_path='dataset/USD_TWD.csv', epochs=100, quick_test=False):
        """
        Ablation Study: 測試每個組件的貢獻
        """
        if quick_test:
            epochs = 20
            print("⚡ Quick Test Mode: epochs reduced to 20")
        
        print("\n" + "="*80)
        print("🔬 Ablation Study: Component Contribution Analysis")
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
        
        # Define ablation experiments
        ablation_configs = [
            {
                'name': 'Vanilla',
                'use_decomp': False,
                'use_experts': False,
                'use_gating': False,
                'use_dnas': False,
                'description': 'Pure Backbone (Baseline)'
            },
            {
                'name': '+Decomp',
                'use_decomp': True,
                'use_experts': False,
                'use_gating': False,
                'use_dnas': False,
                'description': 'Backbone + Fixed Decomposition'
            },
            {
                'name': '+Experts',
                'use_decomp': True,
                'use_experts': True,
                'use_gating': False,
                'use_dnas': False,
                'description': 'Backbone + Decomp + Fixed CNN Experts'
            },
            {
                'name': '+Gating',
                'use_decomp': True,
                'use_experts': True,
                'use_gating': True,
                'use_dnas': False,
                'description': 'Backbone + Decomp + Experts + Adaptive Gating'
            },
            {
                'name': 'Full (DNAS)',
                'use_decomp': True,
                'use_experts': True,
                'use_gating': True,
                'use_dnas': True,
                'description': 'Full TS_Booster with DNAS'
            }
        ]
        
        ablation_results = {}
        
        for i, exp_config in enumerate(ablation_configs, 1):
            print(f"\n{'='*80}")
            print(f"📊 [{i}/{len(ablation_configs)}] {exp_config['name']}: {exp_config['description']}")
            print(f"{'='*80}\n")
            
            # Create backbone
            backbone = LinearBackbone(seq_len, 2, pred_len)
            
            # Create model based on config
            if exp_config['name'] == 'Vanilla':
                # Pure backbone
                model = backbone
            else:
                # Wrap with FlexibleTSBooster
                model = FlexibleTSBooster(
                    base_model=backbone,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    input_channels=2,
                    use_decomp=exp_config['use_decomp'],
                    use_experts=exp_config['use_experts'],
                    use_gating=exp_config['use_gating'],
                    use_dnas=exp_config['use_dnas'],
                    fixed_decomp_kernel=25,
                    fixed_trend_kernel=5,
                    fixed_seasonal_kernel=5,
                    trend_dnas_kernels=config.get('trend_dnas_kernels', [3, 5, 7, 9]),
                    seasonal_dnas_kernels=config.get('seasonal_dnas_kernels', [3, 5, 7, 9]),
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
            metrics = self._evaluate_model(trained_model, test_loader, scaler, exp_config['name'])
            
            # Extract DNAS status if available
            if exp_config['use_dnas'] and hasattr(trained_model, 'get_dnas_status'):
                metrics['dnas_status'] = trained_model.get_dnas_status()
            
            ablation_results[exp_config['name']] = metrics
            
            print(f"\n✅ {exp_config['name']} Complete!")
            print(f"   RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R2: {metrics['R2']:.4f}")
        
        self.results['ablation'] = ablation_results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_ablation_results(ablation_results, timestamp)
        
        return ablation_results
    
    def run_cross_backbone_validation(self, dataset_path='dataset/USD_TWD.csv', epochs=100, quick_test=False):
        """
        Cross-Backbone Validation: 測試 TS_Booster 對不同 Backbone 的提升效果
        """
        if quick_test:
            epochs = 20
            print("⚡ Quick Test Mode: epochs reduced to 20")
        
        print("\n" + "="*80)
        print("🔄 Cross-Backbone Validation")
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
        
        # Define backbones
        backbones = {
            'Linear': lambda: LinearBackbone(seq_len, 2, pred_len),
            'LSTM': lambda: LSTMBackbone(seq_len, 2, pred_len, hidden_dim=64, num_layers=2),
            'GRU': lambda: GRUBackbone(seq_len, 2, pred_len, hidden_dim=64, num_layers=2),
            'Transformer': lambda: TransformerBackbone(seq_len, 2, pred_len, d_model=64, num_layers=2)
        }
        
        cross_backbone_results = {}
        
        for i, (backbone_name, backbone_fn) in enumerate(backbones.items(), 1):
            print(f"\n{'='*80}")
            print(f"🧠 [{i}/{len(backbones)}] Testing {backbone_name} Backbone")
            print(f"{'='*80}\n")
            
            # Test 1: Vanilla Backbone
            print(f"  [1/2] Training Vanilla {backbone_name}...")
            vanilla_backbone = backbone_fn()
            vanilla_trained = train_v11(
                train_loader, val_loader, test_loader,
                device=self.device, horizon=pred_len, config=config,
                seq_len=seq_len, verbose=False, model=vanilla_backbone
            )
            vanilla_metrics = self._evaluate_model(vanilla_trained, test_loader, scaler, f"Vanilla {backbone_name}")
            
            # Test 2: Backbone + TS_Booster
            print(f"  [2/2] Training {backbone_name} + TS_Booster...")
            boosted_backbone = backbone_fn()
            boosted_model = FlexibleTSBooster(
                base_model=boosted_backbone,
                seq_len=seq_len,
                pred_len=pred_len,
                input_channels=2,
                use_decomp=True,
                use_experts=True,
                use_gating=True,
                use_dnas=True,
                trend_dnas_kernels=config.get('trend_dnas_kernels', [3, 5, 7, 9]),
                seasonal_dnas_kernels=config.get('seasonal_dnas_kernels', [3, 5, 7, 9]),
                decomp_dnas_kernels=config.get('decomp_dnas_kernels', [7, 13, 25, 49, 'identity']),
                trend_hidden_dim=config.get('trend_hidden_dim', 32),
                seasonal_hidden_dim=config.get('seasonal_hidden_dim', 32),
                dropout_p=config.get('trend_dropout', 0.3)
            )
            boosted_trained = train_v11(
                train_loader, val_loader, test_loader,
                device=self.device, horizon=pred_len, config=config,
                seq_len=seq_len, verbose=False, model=boosted_model
            )
            boosted_metrics = self._evaluate_model(boosted_trained, test_loader, scaler, f"{backbone_name} + TS_Booster")
            
            # Calculate improvement
            improvement = (vanilla_metrics['RMSE'] - boosted_metrics['RMSE']) / vanilla_metrics['RMSE'] * 100
            
            cross_backbone_results[backbone_name] = {
                'vanilla': vanilla_metrics,
                'boosted': boosted_metrics,
                'improvement': improvement
            }
            
            print(f"\n✅ {backbone_name} Complete!")
            print(f"   Vanilla RMSE: {vanilla_metrics['RMSE']:.4f}")
            print(f"   Boosted RMSE: {boosted_metrics['RMSE']:.4f}")
            print(f"   Improvement: {improvement:+.2f}%")
        
        self.results['cross_backbone'] = cross_backbone_results
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_cross_backbone_results(cross_backbone_results, timestamp)
        
        return cross_backbone_results
    
    def _evaluate_model(self, model, test_loader, scaler, model_name):
        """評估模型"""
        model.eval()
        preds = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                x = batch["raw_input"].to(self.device)
                y = batch["target"].to(self.device)
                
                output = model(x)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output
                    
                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Inverse scaling
        if scaler is not None:
            vol_std = scaler.scale_[0]
            vol_mean = scaler.mean_[0]
            preds = preds * vol_std + vol_mean
            targets = targets * vol_std + vol_mean
        
        # Flatten
        y_true = targets.flatten()
        y_pred = preds.flatten()
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MDA
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        mda = np.mean(np.sign(y_pred_diff) == np.sign(y_true_diff)) * 100
        
        return {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MDA': float(mda),
            'model_name': model_name
        }
    
    def _save_ablation_results(self, results, timestamp):
        """儲存 Ablation 結果"""
        # JSON
        json_file = self.output_dir / f"ablation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Markdown Report
        md_file = self.output_dir / f"ablation_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write("# Ablation Study Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Configuration | RMSE | MAE | R2 | MDA | Description |\n")
            f.write("|---------------|------|-----|----|----|-------------|\n")
            
            for name, metrics in results.items():
                f.write(f"| {name} | {metrics['RMSE']:.4f} | {metrics['MAE']:.4f} | "
                       f"{metrics['R2']:.4f} | {metrics['MDA']:.2f}% | - |\n")
        
        print(f"\n💾 Ablation results saved to {self.output_dir}")
    
    def _save_cross_backbone_results(self, results, timestamp):
        """儲存 Cross-Backbone 結果"""
        # JSON
        json_file = self.output_dir / f"cross_backbone_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Markdown Report
        md_file = self.output_dir / f"cross_backbone_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write("# Cross-Backbone Validation Results\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Table\n\n")
            f.write("| Backbone | Vanilla RMSE | Boosted RMSE | Improvement |\n")
            f.write("|----------|--------------|--------------|-------------|\n")
            
            for backbone, data in results.items():
                f.write(f"| {backbone} | {data['vanilla']['RMSE']:.4f} | "
                       f"{data['boosted']['RMSE']:.4f} | {data['improvement']:+.2f}% |\n")
        
        print(f"\n💾 Cross-backbone results saved to {self.output_dir}")


def main():
    """主執行函數"""
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive TS-Booster Evaluation')
    parser.add_argument('--experiment', type=str, choices=['ablation', 'cross-backbone', 'all'], 
                       default='all', help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--dataset', type=str, default='dataset/USD_TWD.csv', help='Dataset path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--keep-old', action='store_true', help='Keep old result files (default: clean)')
    args = parser.parse_args()
    
    # 設置隨機種子
    set_random_seed(args.seed)
    
    print("\n" + "="*80)
    print("🚀 Comprehensive TS-Booster Evaluation")
    print("="*80 + "\n")
    
    experiment = ComprehensiveTSBoosterExperiment(clean_old_results=not args.keep_old)
    
    if args.experiment in ['ablation', 'all']:
        print("\n🔬 Running Ablation Study...")
        experiment.run_ablation_study(
            dataset_path=args.dataset,
            epochs=args.epochs,
            quick_test=args.quick
        )
    
    if args.experiment in ['cross-backbone', 'all']:
        print("\n🔄 Running Cross-Backbone Validation...")
        experiment.run_cross_backbone_validation(
            dataset_path=args.dataset,
            epochs=args.epochs,
            quick_test=args.quick
        )
    
    print("\n" + "="*80)
    print("✅ All Experiments Complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
