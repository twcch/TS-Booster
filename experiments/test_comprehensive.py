"""
快速驗證腳本 - 測試完整實驗框架
================================
"""

import torch
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd()))

from src.models.layers import LinearBackbone
from src.models.flexible_booster import FlexibleTSBooster
from src.utils.data_loader import prepare_data

def test_flexible_booster():
    """測試 FlexibleTSBooster 的各種配置"""
    print("\n🧪 Testing FlexibleTSBooster Configurations...")
    
    # Load minimal data
    df = pd.read_csv('dataset/USD_TWD.csv')
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=30, horizon=3, batch_size=32
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    x = batch["raw_input"]
    
    print(f"✅ Data loaded: x.shape = {x.shape}")
    
    # Test configurations
    configs = [
        {'name': 'Vanilla', 'use_decomp': False, 'use_experts': False, 'use_gating': False, 'use_dnas': False},
        {'name': '+Decomp', 'use_decomp': True, 'use_experts': False, 'use_gating': False, 'use_dnas': False},
        {'name': '+Experts', 'use_decomp': True, 'use_experts': True, 'use_gating': False, 'use_dnas': False},
        {'name': '+Gating', 'use_decomp': True, 'use_experts': True, 'use_gating': True, 'use_dnas': False},
        {'name': 'Full (DNAS)', 'use_decomp': True, 'use_experts': True, 'use_gating': True, 'use_dnas': True},
    ]
    
    for config in configs:
        print(f"\n  Testing {config['name']}...")
        
        # Create backbone
        backbone = LinearBackbone(30, 2, 3)
        
        # Create model
        if config['name'] == 'Vanilla':
            model = backbone
        else:
            model = FlexibleTSBooster(
                base_model=backbone,
                seq_len=30,
                pred_len=3,
                input_channels=2,
                **{k: v for k, v in config.items() if k != 'name'}
            )
        
        # Forward pass
        try:
            output = model(x)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            
            print(f"    ✅ Forward pass successful: output.shape = {pred.shape}")
            
            # Check DNAS status
            if config['use_dnas'] and hasattr(model, 'get_dnas_status'):
                status = model.get_dnas_status()
                print(f"    ✅ DNAS status keys: {list(status.keys())}")
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return False
    
    return True

def test_cross_backbone():
    """測試不同 Backbone"""
    print("\n🧪 Testing Different Backbones...")
    
    from src.models.layers import LSTMBackbone, GRUBackbone, TransformerBackbone
    
    # Load minimal data
    df = pd.read_csv('dataset/USD_TWD.csv')
    train_loader, val_loader, test_loader, scaler = prepare_data(
        df, lookback=30, horizon=3, batch_size=32
    )
    
    batch = next(iter(train_loader))
    x = batch["raw_input"]
    
    backbones = {
        'Linear': LinearBackbone(30, 2, 3),
        'LSTM': LSTMBackbone(30, 2, 3, hidden_dim=64, num_layers=2),
        'GRU': GRUBackbone(30, 2, 3, hidden_dim=64, num_layers=2),
        'Transformer': TransformerBackbone(30, 2, 3, d_model=64, num_layers=2)
    }
    
    for name, backbone in backbones.items():
        print(f"\n  Testing {name} Backbone...")
        
        # Test vanilla
        try:
            output = backbone(x)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output
            print(f"    ✅ Vanilla {name}: output.shape = {pred.shape}")
        except Exception as e:
            print(f"    ❌ Vanilla {name} error: {e}")
            return False
        
        # Test with TS_Booster
        try:
            # Need to create a new backbone instance for booster
            if name == 'Linear':
                new_backbone = LinearBackbone(30, 2, 3)
            elif name == 'LSTM':
                new_backbone = LSTMBackbone(30, 2, 3, hidden_dim=64, num_layers=2)
            elif name == 'GRU':
                new_backbone = GRUBackbone(30, 2, 3, hidden_dim=64, num_layers=2)
            else:
                new_backbone = TransformerBackbone(30, 2, 3, d_model=64, num_layers=2)
            
            boosted = FlexibleTSBooster(
                base_model=new_backbone,
                seq_len=30,
                pred_len=3,
                input_channels=2,
                use_decomp=True,
                use_experts=True,
                use_gating=True,
                use_dnas=True
            )
            
            output = boosted(x)
            pred = output[0]
            print(f"    ✅ {name} + TS_Booster: output.shape = {pred.shape}")
        except Exception as e:
            print(f"    ❌ {name} + TS_Booster error: {e}")
            return False
    
    return True

def main():
    """執行所有測試"""
    print("\n" + "="*60)
    print("🧪 Comprehensive Experiment Framework Validation")
    print("="*60)
    
    tests = [
        ("FlexibleTSBooster Configurations", test_flexible_booster),
        ("Cross-Backbone Compatibility", test_cross_backbone),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 Test Summary")
    print("="*60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r for _, r in results)
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run full experiments.")
        print("\nNext steps:")
        print("1. Quick ablation test:")
        print("   uv run python experiments/comprehensive_evaluation.py --experiment ablation --quick")
        print("\n2. Quick cross-backbone test:")
        print("   uv run python experiments/comprehensive_evaluation.py --experiment cross-backbone --quick")
        print("\n3. Full experiment:")
        print("   uv run python experiments/comprehensive_evaluation.py --experiment all --epochs 100")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before running experiments.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
