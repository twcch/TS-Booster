# TS-Booster：架構與技術文件

## 概述

TS-Booster 是一個**模型無關的時間序列增強框架**，它可以包裝任意骨幹模型（backbone），並透過分解增強的專家模組（expert module）搭配自適應門控（adaptive gating）來提升預測效果。核心洞察在於：時間序列分解不是用來拆分預測任務（如 DLinear），而是用來**豐富**輔助專家模組的輸入資訊。

### 核心公式

```
ŷ = y_base + gate · y_expert
```

其中：
- `y_base`：骨幹模型對原始輸入的預測
- `y_expert`：多輪專家模組對增強輸入的預測總和
- `gate`：基於信噪比（SNR）的逐樣本自適應門控

---

## 架構流程

```
輸入 x ∈ (B, S, C)
│
├──────────────────────────────────── Backbone(x) → y_base ∈ (B, P)
│                                         │
├── MultiScaleDecomp[7,13](x)             │
│   ├── trend   ∈ (B, S, C)              │
│   └── seasonal ∈ (B, S, C)             │
│                                         │
├── 拼接: x_enriched = [x, trend]         │
│   → (B, S, 2C)                          │
│                                         │
├── Expert Round 1(x_enriched) → y_r1     │
├── Expert Round 2(x_enriched) → y_r2     │
│   y_expert = y_r1 + y_r2               │
│                                         │
├── SNR Gate:                             │
│   snr = std(seasonal) / std(x)          │
│   gate = σ(w · snr + b)                │
│   y_expert = gate · y_expert            │
│                                         │
└── 融合: ŷ = y_base + y_expert ──────────┘
```

---

## 元件詳解

### 1. 輸入與資料前處理

**檔案**：`src/utils/data_loader.py`

輸入特徵（2 個通道）：
- **波動率（Volatility）**：滾動 7 日對數報酬標準差，年化處理（×√252×100）
- **對數報酬（Log Return）**：ln(Close_t / Close_{t-1})

資料處理流程：
1. 從原始收盤價計算對數報酬率與滾動波動率
2. StandardScaler 僅在訓練集上擬合（防止資料洩漏）
3. 滑動視窗切割：回看期（lookback）=30，預測期（horizon）=3
4. 訓練/驗證/測試集分割：70%/10%/20%（按時間順序）

### 2. 骨幹模型（Backbone，支援任意架構）

**檔案**：`src/models/layers.py` — `LinearBackbone`、`LSTMBackbone`、`GRUBackbone`、`TransformerBackbone`

骨幹模型接收原始輸入 `x ∈ (B, 30, 2)`，輸出 `y_base ∈ (B, 3)`。

TS-Booster 具有模型無關性——骨幹可以是任意時間序列模型。已驗證四種骨幹：

| 骨幹 | 架構 | 參數量 |
|------|------|--------|
| **Linear** | 展平 → FC(60, 6) → 重塑 → 取通道 0 | ~360 |
| **LSTM** | 2 層 LSTM(2→64) → FC(64, 3) | ~35K |
| **GRU** | 2 層 GRU(2→64) → FC(64, 3) | ~27K |
| **Transformer** | Linear(2→64) → 2 層 TransformerEncoder(d=64, h=4) → FC(64, 3) | ~70K |

骨幹模型**只看到原始輸入**——TS-Booster 不會修改骨幹。這與 "Vanilla" 基準線使用的是完全相同的骨幹，確保公平比較。

### 3. 多尺度分解（Multi-Scale Decomposition）

**檔案**：`src/models/layers.py` — `MultiScaleDecomp`

```python
class MultiScaleDecomp(nn.Module):
    def __init__(self, kernels=[7, 13]):
        self.avgs = [AvgPool1d(k) for k in kernels]
        self.weights = nn.Parameter(torch.zeros(len(kernels)))  # 可學習

    def forward(self, x):
        w = softmax(self.weights)          # 初始為 [0.5, 0.5]
        trend = Σ w_i · AvgPool(x, k_i)   # 加權移動平均集成
        seasonal = x - trend               # 殘差
        return trend, seasonal
```

**設計選擇**：
- **核大小 [7, 13]**：分別對應約 1 週和 2 週的交易日資料（seq_len=30）。核大小 3 曾測試但會損害 THB_TWD 的表現（+8.56% → +0.21%）。
- **可學習權重**：2 個參數搭配 softmax 正規化。模型會學習短期（7 日）和中期（13 日）移動平均的最佳混合比例。
- **非逐樣本自適應**：曾測試使用 MLP 產生逐樣本權重，但會過擬合。簡單的資料集層級 nn.Parameter 效果更好。

**用途**：分解**不是**用來拆分預測任務（DLinear 風格），而是產生：
- **trend**：用於豐富專家輸入（去噪後的結構性參考）
- **seasonal**：用於 SNR 門控來衡量增強品質

### 4. 分解增強專家（Decomposition-Enriched Expert）

**檔案**：`src/models/layers.py` — `MultiScaleExpert`

專家模組接收增強輸入 `x_enriched = [x, trend] ∈ (B, 30, 4)`——原始訊號與趨勢成分的拼接。

```python
class MultiScaleExpert(nn.Module):
    """Inception 風格多尺度 CNN 專家"""
    def __init__(self, seq_len=30, pred_len=3, input_channels=4,
                 hidden_dim=32, kernels=[3, 5, 7]):
        branch_dim = hidden_dim // len(kernels)  # 每分支 10（32//3）
        self.branches = [
            Conv1d(4, 10, kernel_size=3) + BN + LeakyReLU + Dropout,
            Conv1d(4, 10, kernel_size=5) + BN + LeakyReLU + Dropout,
            Conv1d(4, 10, kernel_size=7) + BN + LeakyReLU + Dropout,
        ]
        self.fc = Linear(30 * 30, 3)  # seq_len × concat_dim → pred_len
```

**架構**：Inception 風格，3 個平行 Conv1d 分支使用核大小 [3, 5, 7]，捕捉不同時間尺度的模式。特徵拼接後透過單一全連接層投射到預測長度。

**關鍵設計**：全連接層直接輸出 `pred_len`（而非 `pred_len × channels`）。因為預測目標僅有通道 0（波動率），避免在多通道輸出上浪費參數。

**為何使用趨勢增強（而非季節性）**：
- 趨勢提供**去噪後的結構性參考**——專家同時看到原始訊號和平滑版本，能學習殘差模式。
- 金融資料在 seq_len=30 的尺度下，季節性成分主要是噪音，不是真正的周期性。
- 原始 + 趨勢**並不冗餘**：`trend = weighted_avg(raw)`，專家同時接收帶噪細節和去噪摘要，學習利用兩者之間的差異。

### 5. 多輪增強（Multi-Round Boosting）

**檔案**：`src/models/ts_booster.py` — `TS_Booster.experts`

```python
self.experts = nn.ModuleList([
    MultiScaleExpert(...) for _ in range(n_boost_rounds)  # n=2
])

# 前向傳播：
y_expert_total = 0
for expert in self.experts:
    y_expert_total += expert(x_enriched)
```

兩個獨立的專家模組接收**相同的增強輸入**，輸出以**加法方式合併**。這**不是**序列式殘差增強（round 2 以 round 1 的殘差為目標）——兩輪共同聯合目標完整殘差。

**為何不使用序列式殘差增強**：經測試發現會損害效果。僅 n=2 輪時，限制 round 2 只能針對 round 1 的殘差，會抑制兩輪之間的協作。聯合目標讓兩輪可以各自專注於殘差的不同面向。

**參數開銷**：每個專家約 300 個參數。n=2 輪共增加約 600 個參數，加上分解模組（約 2 個參數），總開銷約 602 個參數（相對骨幹僅增加 9.2%）。

### 6. SNR 門控（信噪比門控）

**檔案**：`src/models/ts_booster.py` — SNR 門控邏輯

```python
# 逐樣本自適應門控
seasonal_std = seasonal.std(dim=time).mean(dim=channel)  # 每樣本一個純量
raw_std = x.std(dim=time).mean(dim=channel)              # 每樣本一個純量
snr = seasonal_std / (raw_std + 1e-8)
gate = sigmoid(w · snr + b)         # w, b 為可學習參數（2 個參數）
y_expert = gate · y_expert           # 縮放專家貢獻
```

**直覺**：門控衡量每個樣本的「可分解程度」：
- `seasonal = x - trend`，因此 `seasonal_std = std(x - trend)` = 原始訊號偏離平滑版本的程度。
- 高 seasonal_std 相對於 raw_std → 該樣本具有豐富的結構性模式 → 更信任趨勢增強的專家（gate → 1）。
- 低 seasonal_std → 訊號本身已很平滑 → 趨勢提供的附加價值小 → 抑制專家（gate → 0）。

**SNR 門控與趨勢增強的關聯**：`seasonal_std` 直接衡量了**趨勢增強的資訊量**。若 `std(x - trend)` 很高，代表趨勢與原始訊號差異大，專家確實獲得了全新的資訊。

**僅 2 個可學習參數**（`w` 和 `b`）：過擬合風險極低，同時實現逐樣本的自適應行為。

### 7. 加法融合（Additive Fusion）

```python
ŷ = y_base + gate · y_expert
```

簡單的加法融合。骨幹提供基礎預測，門控後的專家提供修正項。這確保了：
- **骨幹永遠不會退化**：最差情況下（gate → 0），TS-Booster 退化為原始骨幹。
- **單調改善**：專家只能補充資訊，不能取代骨幹。
- **可解釋性**：`gate · y_expert` 直接衡量專家的貢獻量。

---

## 訓練流程

**檔案**：`src/engine/trainer.py` — `train_v11()`

### 損失函數：HybridDirectionalLoss

**檔案**：`src/utils/metrics.py`

```python
L = (1 - α) · L_huber(ŷ, y) + α · L_direction(ŷ, y, x_{t-1})
```

- **Huber Loss**（δ=1.0）：相比 MSE 對離群值更穩健。兼顧 L1（抗離群值）和 L2（平滑梯度）的特性。
- **方向性損失**（Soft Margin）：懲罰錯誤的方向預測（漲 vs. 跌）。僅在 `|true_delta| > threshold` 的樣本上使用 `soft_margin_loss(pred_delta, sign(true_delta))`。
- **權重** `α=0.5`：對幅度準確性和方向正確性給予相等重視。

### 自適應多目標損失加權

**檔案**：`src/engine/adaptive_scheduler.py` — `AdaptiveWeightScheduler`

總訓練損失包含 3 個目標：

```
L_total = L_main + w_aux · L_expert + w_base · L_base
```

- `L_main`：最終輸出 `ŷ` 的 HybridDirectionalLoss
- `L_expert`：專家輸出與骨幹殘差之間的 MSE `(y - y_base.detach())`
- `L_base`：骨幹輸出與目標之間的 MSE

排程器動態調整 `w_aux` 和 `w_base`：
1. **暖身階段**（前 20% 的 epoch）：低輔助權重，讓模型先學習基本模式。
2. **搜尋階段**：監控骨幹與專家的損失差距。若專家落後 → 增加 `w_aux`；若骨幹落後 → 增加 `w_base`。
3. **EMA 平滑**（α=0.7）作用於損失差距 + **動量**（β=0.7）作用於權重更新，防止震盪。

### 優化器與學習率排程

- **AdamW**（lr=0.001, weight_decay=1e-4）
- **ReduceLROnPlateau**（factor=0.5, patience=5, min_lr=1e-5）
- **梯度裁剪**：max_norm=5.0
- **早停機制**：驗證損失連續 30 個 epoch 未改善則停止

---

## 超參數（最終配置）

**檔案**：`src/configs/model_config.py`

| 參數 | 數值 | 說明 |
|------|------|------|
| `seq_len` | 30 | 約 6 週的交易日資料 |
| `pred_len` | 3 | 預測未來 3 天 |
| `input_channels` | 2 | 波動率 + 對數報酬 |
| `decomp_kernels` | [7, 13] | 約 1 週 + 2 週的移動平均 |
| `expert_kernels` | [3, 5, 7] | 多尺度 CNN 分支 |
| `expert_hidden_dim` | 32 | 每 CNN 分支 10-11 |
| `n_boost_rounds` | 2 | 2 個獨立的專家輪次 |
| `expert_type` | cnn | Inception 風格 CNN（對比 MLP） |
| `expert_gate` | snr | 基於信噪比的自適應門控 |
| `enrich_mode` | trend | 趨勢增強（對比季節性） |
| `dropout` | 0.3 | 應用於專家分支 |
| `lr` | 0.001 | AdamW 學習率 |
| `weight_decay` | 1e-4 | L2 正則化 |
| `dir_weight` | 0.5 | 方向性損失權重 |
| `epochs` | 100 | 搭配早停機制（patience=30） |

---

## 消融實驗（4 種配置）

| 配置 | 骨幹 | 分解 | 專家 | 專家輸入 | SNR 門控 |
|------|------|------|------|----------|----------|
| **Vanilla** | 1× 原始輸入 | - | - | - | - |
| **+Decomp** | 2× 趨勢/季節性 | MultiScale[7,13] | - | - | - |
| **+Expert** | 1× 原始輸入 | - | 2 輪 CNN | 原始（2ch） | - |
| **TS-Booster** | 1× 原始輸入 | MultiScale[7,13] | 2 輪 CNN | [原始, 趨勢]（4ch） | SNR |

關鍵觀察：
- **+Decomp（-0.1%）**：拆分預測反而有害。驗證了「增強而非拆分」的核心理念。
- **+Expert（+12.0%）**：單獨使用專家即可提供顯著提升。多輪 CNN 是主要貢獻。
- **TS-Booster（+13.8%）**：分解增強 + SNR 門控帶來額外的系統性改善。統計顯著（Wilcoxon p=0.0046）。

---

## 實驗結果摘要

### 消融實驗（16 個資料集，單一種子）
- TS-Booster 在 **12/16** 個資料集中表現最佳
- 相對 Vanilla 的平均改善：**+13.8%**
- 相對 +Expert 的平均改善：**+1.8%**（Wilcoxon p=0.005）

### 多種子穩健性（5 個種子 × 16 個資料集）
- **13/16 個資料集**呈正向改善（81.3%）
- 0/16 個資料集出現統計顯著的退化
- 股票：6/6 = 100% 正向，平均 +2.83%
- 匯率：7/10 = 70% 正向，平均 +0.35%

### 跨骨幹驗證（4 種骨幹 × 16 個資料集）
- **61/64 勝（95.3%）**
- Linear：16/16，平均 +13.8%
- LSTM：13/16，平均 +9.5%
- GRU：16/16，平均 +9.0%
- Transformer：16/16，平均 +18.5%

---

## 檔案結構

```
src/
├── models/
│   ├── ts_booster.py         # TS-Booster 正式模組
│   ├── flexible_booster.py   # 消融實驗用的彈性版本（可切換元件）
│   ├── layers.py             # MultiScaleDecomp、MultiScaleExpert、各骨幹
│   └── factory.py            # 模型工廠（建立 Vanilla + Boosted 配對）
├── engine/
│   ├── trainer.py            # 訓練迴圈（train_v11）
│   └── adaptive_scheduler.py # 自適應多目標損失加權
├── utils/
│   ├── data_loader.py        # 資料前處理與滑動視窗
│   └── metrics.py            # HybridDirectionalLoss、評估指標
├── configs/
│   └── model_config.py       # 集中式超參數配置
└── preprocess/
    ├── get_fx.py             # 匯率資料擷取
    ├── get_stock.py          # 股票資料擷取
    └── getUSDTWD.py          # USD/TWD 專用擷取器

experiments/
├── comprehensive_evaluation.py   # 消融 + 跨骨幹評估
├── run_ablation_all.py           # 批次消融執行器（16 個資料集）
├── run_cross_backbone_all.py     # 批次跨骨幹執行器
├── multi_seed_test.py            # 5 種子穩健性測試
├── visualize_comprehensive.py    # 結果視覺化
├── plot_architecture_v2.py       # 架構圖產生器
└── generate_paper_figures.py     # 論文圖表產生器

dataset/                          # 16 個 CSV 檔案（7 TWD 匯率 + 3 國際匯率 + 6 台股）
```
