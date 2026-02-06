# TS-Booster: Plug-and-Play Time Series Forecasting Enhancement

**TS-Booster** 是一個即插即用的時間序列預測增強模組,可以顯著提升任何 Backbone 模型的性能 (8.4%-23.6%)。

---

## 🚀 快速開始

### 1. 安裝依賴
```bash
uv sync
```

### 2. 運行實驗
```bash
# 快速測試 (20 epochs, ~30 分鐘)
./run_experiment.sh all --quick

# 完整實驗 (100 epochs, ~3-4 小時)
./run_experiment.sh all --epochs 100

# 只運行 Ablation Study
./run_experiment.sh ablation --epochs 100

# 只運行 Cross-Backbone Validation
./run_experiment.sh cross-backbone --epochs 100
```

### 3. 生成圖表
```bash
# 自動使用最新結果
./visualize_results.sh --all
```

### 4. 查看結果
```bash
# 查看圖表
ls experiments/comprehensive_results/figures/

# 查看報告
cat experiments/comprehensive_results/ablation_report_*.md
cat experiments/comprehensive_results/cross_backbone_report_*.md
```

---

## 📊 實驗結果

### Ablation Study (組件貢獻分析)

| 配置            | RMSE       | 改善幅度     |
| --------------- | ---------- | ------------ |
| Vanilla         | 1.8459     | -            |
| +Decomp         | 1.8389     | +0.4%        |
| +Experts        | 1.6429     | +11.0%       |
| +Gating         | 1.7801     | +3.6%        |
| **Full (DNAS)** | **1.6102** | **+12.8%** 🏆 |

**關鍵發現**:
- CNN Experts 貢獻最大 (+10.6%)
- DNAS 提供額外增益 (+1.8%)
- Gating 需要與 DNAS 結合才有效

### Cross-Backbone Validation (泛化能力驗證)

| Backbone        | Vanilla | +TS_Booster | 改善幅度      |
| --------------- | ------- | ----------- | ------------- |
| Linear          | 1.8151  | 1.6622      | +8.42%        |
| LSTM            | 2.1922  | 1.9806      | +9.65%        |
| GRU             | 2.2993  | 1.9830      | +13.76%       |
| **Transformer** | 2.5585  | 1.9555      | **+23.57%** 🏆 |

**關鍵發現**:
- 所有 Backbone 都有顯著改善 (8.4%-23.6%)
- 對較弱 Backbone 提升更明顯 (Transformer > GRU > LSTM > Linear)
- 證明 TS_Booster 的 plug-and-play 特性

---

## 📁 專案結構

```
paper_booster/
├── README.md                         # 本文件
├── run_experiment.sh                 # 實驗執行腳本
├── visualize_results.sh              # 視覺化腳本
├── verify_setup.sh                   # 環境驗證
│
├── src/                              # 源代碼
│   ├── models/
│   │   ├── flexible_booster.py       # FlexibleTSBooster (支援組件開關)
│   │   ├── ts_booster.py             # TS_Booster (標準版本)
│   │   └── layers.py                 # DNAS, CNN, Backbones
│   ├── configs/model_config.py       # 配置
│   ├── engine/trainer.py             # 訓練引擎
│   └── utils/data_loader.py          # 數據加載
│
├── dataset/                          # 數據集
│   └── USD_TWD.csv
│
└── experiments/                      # 實驗
    ├── comprehensive_evaluation.py   # 核心實驗腳本
    ├── visualize_comprehensive.py    # 視覺化工具
    ├── test_comprehensive.py         # 驗證測試
    └── comprehensive_results/        # 實驗結果
        ├── ablation_results_*.json
        ├── cross_backbone_results_*.json
        └── figures/                  # 5 張論文圖表
            ├── ablation_progression.png
            ├── ablation_all_metrics.png
            ├── ablation_improvement.png
            ├── backbone_comparison.png
            └── backbone_improvement.png
```

---

## 🔧 腳本使用說明

### `run_experiment.sh` - 實驗執行

```bash
# 查看幫助
./run_experiment.sh --help

# 實驗類型
./run_experiment.sh ablation          # Ablation Study
./run_experiment.sh cross-backbone    # Cross-Backbone Validation
./run_experiment.sh all              # 所有實驗
./run_experiment.sh test             # 驗證測試

# 選項
--quick                              # 快速模式 (20 epochs)
--epochs N                           # 指定訓練輪數 (預設: 100)
--dataset NAME                       # 指定數據集 (預設: USD_TWD)
--seed N                             # 隨機種子 (預設: 42, 確保可重現)
--keep-old                           # 保留舊結果 (預設: 自動清理)

# 範例
./run_experiment.sh ablation --quick
./run_experiment.sh all --epochs 100
./run_experiment.sh cross-backbone --dataset JPY_TWD --seed 123
./run_experiment.sh all --epochs 100 --keep-old  # 保留舊結果
```

### `visualize_results.sh` - 視覺化

```bash
# 查看幫助
./visualize_results.sh --help

# 自動使用最新結果
./visualize_results.sh --latest

# 視覺化所有結果
./visualize_results.sh --all

# 指定特定檔案
./visualize_results.sh --ablation experiments/comprehensive_results/ablation_results_*.json
./visualize_results.sh --cross-backbone experiments/comprehensive_results/cross_backbone_results_*.json
```

---

## 🎯 核心組件

### FlexibleTSBooster

支援靈活的組件開關,用於 Ablation Study:

```python
from src.models.flexible_booster import FlexibleTSBooster
from src.models.layers import LinearBackbone

# 創建 Backbone
backbone = LinearBackbone(seq_len=30, input_channels=2, pred_len=3)

# 創建 TS_Booster
model = FlexibleTSBooster(
    base_model=backbone,
    seq_len=30,
    pred_len=3,
    input_channels=2,
    use_decomp=True,    # 使用分解
    use_experts=True,   # 使用 CNN Experts
    use_gating=True,    # 使用自適應 Gating
    use_dnas=True       # 使用 DNAS
)
```

### DNAS 機制

自動學習最佳架構參數:
- **Decomposition**: 學習最佳移動平均窗口 (MA_7, MA_13, MA_25, MA_49)
- **CNN Experts**: 學習最佳卷積核大小 (K_3, K_5, K_7, K_9, K_11)
- **Gating**: 學習最佳融合權重

---

## 💡 使用技巧

### 背景執行 (長時間實驗)
```bash
nohup ./run_experiment.sh all --epochs 100 > experiment.log 2>&1 &
tail -f experiment.log
```

### 快速查看最新結果
```bash
# 最新 Ablation 報告
cat $(ls -t experiments/comprehensive_results/ablation_report_*.md | head -1)

# 最新圖表
ls -lt experiments/comprehensive_results/figures/
```

---

## 📖 論文貢獻

1. **TS_Booster**: 一個有效的 plug-and-play 模組,可提升任何 Backbone
2. **Ablation Study**: 證明每個組件的貢獻 (CNN Experts 最重要)
3. **Cross-Backbone**: 證明泛化能力和架構無關性 (8.4%-23.6% 改善)
4. **DNAS**: 自動學習最佳架構參數

---

## 📝 引用

```bibtex
@article{tsbooster2026,
  title={TS-Booster: A Plug-and-Play Enhancement Module for Time Series Forecasting},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2026}
}
```

---

**最後更新**: 2026-02-06
