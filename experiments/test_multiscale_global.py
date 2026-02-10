"""
Test Multi-Scale Decomposition with Global Weights
===================================================
验证 MultiScaleDecomp 的功能和性能
"""

import sys
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_data
from src.models.factory import create_model
from src.engine.trainer import Trainer
from src.utils.config import get_parser


def test_global_weights():
    """Test Multi-Scale Decomposition with Global Weights"""

    # Parse arguments
    parser = get_parser()
    args = parser.parse_args([
        '--dataset_path', 'data/USD_TWD.csv',
        '--backbone', 'Linear',
        '--seq_len', '30',
        '--pred_len', '1',
        '--epochs', '100',
        '--batch_size', '128',
        '--learning_rate', '0.001',
    ])

    # Load data
    print("\n" + "="*60)
    print("Loading USD_TWD dataset...")
    print("="*60)
    train_loader, val_loader, test_loader = load_data(
        args.dataset_path,
        args.seq_len,
        args.pred_len,
        args.batch_size
    )

    # Test configurations
    configs = [
        {
            'name': 'Vanilla',
            'use_decomp': False,
            'use_experts': False,
            'use_multiscale_decomp': False,
        },
        {
            'name': '+Decomp (Fixed MA25)',
            'use_decomp': True,
            'use_experts': False,
            'use_multiscale_decomp': False,
            'fixed_decomp_kernel': 25,
        },
        {
            'name': '+Decomp+Experts (Fixed MA25)',
            'use_decomp': True,
            'use_experts': True,
            'use_multiscale_decomp': False,
            'fixed_decomp_kernel': 25,
        },
        {
            'name': '+MultiScale (Global Weights)',
            'use_decomp': True,
            'use_experts': False,
            'use_multiscale_decomp': True,
            'multiscale_kernels': [7, 13, 25, 49],
        },
        {
            'name': '+MultiScale+Experts',
            'use_decomp': True,
            'use_experts': True,
            'use_multiscale_decomp': True,
            'multiscale_kernels': [7, 13, 25, 49],
        },
    ]

    results = {}

    for config in configs:
        print("\n" + "="*60)
        print(f"Testing: {config['name']}")
        print("="*60)

        # Update args with config
        for key, value in config.items():
            if key != 'name':
                setattr(args, key, value)

        # Create model
        model = create_model(args, input_channels=1)
        print(f"\n[Model Created] {config['name']}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        # Check if MultiScaleDecomp is used
        if hasattr(model, 'use_multiscale_decomp') and model.use_multiscale_decomp:
            print("\n[MultiScaleDecomp Status]")
            weights_status = model.decomp.get_weights_status()
            for k, v in weights_status.items():
                print(f"  {k}: {v:.4f}")

        # Train model
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args
        )

        test_rmse = trainer.train()

        # Get final weights if MultiScaleDecomp
        final_weights = None
        if hasattr(model, 'use_multiscale_decomp') and model.use_multiscale_decomp:
            print("\n[Final MultiScaleDecomp Weights]")
            final_weights = model.decomp.get_weights_status()
            for k, v in final_weights.items():
                print(f"  {k}: {v:.4f}")

        # Store results
        results[config['name']] = {
            'test_rmse': float(test_rmse),
            'total_params': total_params,
            'trainable_params': trainable_params,
            'final_weights': final_weights
        }

        print(f"\n[Result] {config['name']}: RMSE = {test_rmse:.6f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, result in results.items():
        rmse = result['test_rmse']
        params = result['total_params']
        print(f"{name:40s} | RMSE: {rmse:.6f} | Params: {params:,}")
        if result['final_weights']:
            print(f"  Final Weights: {result['final_weights']}")

    # Save results
    output_path = Path(__file__).parent / 'multiscale_global_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Compare with fixed kernel
    baseline_rmse = results['+Decomp+Experts (Fixed MA25)']['test_rmse']
    multiscale_rmse = results['+MultiScale+Experts']['test_rmse']
    improvement = (baseline_rmse - multiscale_rmse) / baseline_rmse * 100

    print(f"\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Fixed MA25 + Experts: {baseline_rmse:.6f}")
    print(f"MultiScale + Experts: {multiscale_rmse:.6f}")
    print(f"Improvement: {improvement:+.2f}%")

    if improvement > 0:
        print("\n✓ MultiScale Global Weights is BETTER than fixed kernel!")
    elif improvement < -1:
        print("\n✗ MultiScale Global Weights is WORSE than fixed kernel")
    else:
        print("\n≈ MultiScale Global Weights is SIMILAR to fixed kernel")


if __name__ == '__main__':
    test_global_weights()
