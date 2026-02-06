"""
模型配置文件
集中管理所有模型超參數，單一來源 (Single Source of Truth)
"""

# Stage 3 優化後的最佳參數 (2026-02-01)
# RMSE: 0.6038, Accuracy: 87.5%
BEST_STAGE3_CONFIG = {
    # --- 模型架構參數 (Model Architecture) ---
    "seasonal_hidden_dim": 64,
    "trend_hidden_dim": 16,
    "seasonal_dropout": 0.3,
    "trend_dropout": 0.3,
    "delta_net_hidden_dim": 8,
    
    # --- DNAS 候選池 (Kernel Candidates) ---
    # 必須是奇數，提供多尺度的感受野供模型選擇
    "seasonal_dnas_kernels": [3, 5, 7, 9],
    "trend_dnas_kernels": [3, 5, 7, 9, 11],
    "decomp_dnas_kernels": [7, 13, 25, 49, 'identity'],  # 包含 Identity 選項
    
    # --- 門控初始化 (Gating Initialization) ---
    "trend_base_init": -1.0,  # 改為 -1.0 (約 0.27)
    "seasonal_base_init": 0.0, # 改為 0.0 (約 0.50)
    
    # --- 訓練超參數 (Training Hyperparameters) ---
    "lr": 0.001,
    "weight_decay": 0.0001,
    
    # --- 固定架構開關 (Fixed Architecture Flags) ---
    "use_seasonal_cnn": True,
    "use_trend_cnn": True,
    "use_decomp": True,
    "use_learnable": True,
    "backbone_type": 'linear',
}

# 可以在這裡定義其他實驗的配置
# EXPERIMENT_A_CONFIG = { ... }
