"""
Conference version config — simplified TS-Booster.

Simplifications:
  - Single fixed MA (kernel=7) instead of learnable multi-scale [7,13]
  - Single expert round (no multi-round boosting)
  - No SNR gate
  - MSE + aux residual loss (no HybridDirectionalLoss)
"""

CONFIG = {
    # Architecture
    "backbone_type": "linear",
    "expert_hidden_dim": 32,
    "decomp_kernel": 7,
    "expert_kernels": [3, 5, 7],
    "dropout": 0.3,

    # Training
    "lr": 0.001,
    "weight_decay": 0.0001,
    "aux_weight": 0.3,       # Expert residual loss weight
    "epochs": 100,

    # Data
    "seq_len": 30,
    "pred_len": 3,
    "input_channels": 2,
    "batch_size": 32,
}
