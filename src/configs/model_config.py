"""
模型配置文件
集中管理所有模型超參數，單一來源 (Single Source of Truth)
"""

BEST_STAGE3_CONFIG = {
    # --- 模型架構參數 (Model Architecture) ---
    "expert_hidden_dim": 32,
    "decomp_kernels": [7, 13],
    "expert_kernels": [3, 5, 7],
    "dropout": 0.3,

    # --- Frequency Feature ---
    "n_freq": 4,

    # --- Multi-Round Boosting ---
    "n_boost_rounds": 2,

    # --- 訓練超參數 (Training Hyperparameters) ---
    "lr": 0.001,
    "weight_decay": 0.0001,
    "dir_weight": 0.5,

    # --- 架構選項 ---
    "backbone_type": 'linear',
    "expert_type": 'cnn',
}
