"""
Model Configuration
Single Source of Truth for all model hyperparameters.
"""

BEST_STAGE3_CONFIG = {
    # --- Model Architecture ---
    "expert_hidden_dim": 32,
    "decomp_kernels": [7, 13],
    "expert_kernels": [3, 5, 7],
    "dropout": 0.3,

    # --- Multi-Round Boosting ---
    "n_boost_rounds": 2,

    # --- Enrichment (TS-Booster specific) ---
    "expert_gate": "snr",         # SNR-based adaptive gate
    "enrich_mode": "trend",       # trend enrichment (not seasonal)

    # --- Training Hyperparameters ---
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dir_weight": 0.5,
    "fixed_loss_weights": True,       # Fixed > Adaptive (ablation verified)
    "fixed_aux_weight": 0.3,          # Expert residual loss weight
    "fixed_base_weight": 0.1,         # Backbone auxiliary loss weight

    # --- Architecture ---
    "backbone_type": 'linear',
    "expert_type": 'cnn',
}
