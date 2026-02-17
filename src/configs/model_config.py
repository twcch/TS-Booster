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

    # --- Frequency Feature ---
    "n_freq": 4,

    # --- Multi-Round Boosting ---
    "n_boost_rounds": 2,

    # --- Enrichment (TS-Booster specific) ---
    "expert_gate": "snr",         # SNR-based adaptive gate
    "enrich_mode": "trend",       # trend enrichment (not seasonal)

    # --- Training Hyperparameters ---
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dir_weight": 0.5,

    # --- Architecture ---
    "backbone_type": 'linear',
    "expert_type": 'cnn',
}
