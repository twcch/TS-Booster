# Phase 1 — Persistence Baseline Test

## 背景與動機

學長指出，本論文的預測目標為 **7 日滾動年化波動率**（σ_t），其定義如下：

```
σ_t = std(r_{t-6}, r_{t-5}, ..., r_t) × sqrt(252) × 100
```

由於相鄰兩個 σ 值共享 6/7 筆報酬率，σ_{t+1} 在結構上與 σ_t 高度相關。
因此，最簡單的 **Persistence baseline**（直接將 σ_t 延展為未來預測）的表現可能已相當強，
若 vanilla Linear 都無法勝過此 baseline，則「相較 Linear 改善 15%」的結論意義有限。

## 測試方法

**Persistence 預測定義：**
```
σ_hat_{t+1} = σ_hat_{t+2} = σ_hat_{t+3} = σ_t
```
即：將 lookback 視窗末端的最後一筆已知波動率，直接 carry-forward 作為未來 3 天的預測。

**評估設定：**
- 16 個資料集（TWD FX ×7、International FX ×3、Taiwan Stocks ×6）
- 完全相同的 train/val/test 分割（70/10/20）與 StandardScaler
- 指標：RMSE、MAE、MDA（與主實驗一致，inverse-transform 回原始尺度後計算）
- Linear vanilla 與 TSB 的數字取自主實驗 5-seed 平均（`statistical_test_results.json`）

## 結果

### 全體總覽

| 比較 | 勝出資料集數 | 平均 RMSE 改善 |
|------|------------|--------------|
| Linear > Persistence | 15 / 16 | +6.32% |
| TSB > Persistence | 15 / 16 | — |

### 分類別

| 類別 | Linear > Persistence | 平均改善 |
|------|---------------------|---------|
| TWD FX (7 datasets) | 7 / 7 | +12.31% |
| International FX (3 datasets) | 3 / 3 | +9.52% |
| Taiwan Stocks (6 datasets) | 5 / 6 | −2.27% |

### 逐資料集 RMSE

| Dataset | Persistence | Linear | TSB | Linear > Per? |
|---------|------------|--------|-----|--------------|
| USD_TWD | 2.1350 | 1.8102 | 1.5282 | ✓ |
| JPY_TWD | 2.8460 | 2.4169 | 2.0504 | ✓ |
| KRW_TWD | 2.0439 | 1.8220 | 1.6839 | ✓ |
| CAD_TWD | 2.1843 | 1.9763 | 1.7736 | ✓ |
| CNY_TWD | 2.2135 | 1.8653 | 1.5986 | ✓ |
| EUR_TWD | 2.2251 | 2.0276 | 1.8049 | ✓ |
| THB_TWD | 2.1727 | 1.9364 | 1.6950 | ✓ |
| EUR_USD | 1.7356 | 1.5952 | 1.3999 | ✓ |
| GBP_USD | 2.0940 | 1.9204 | 1.6570 | ✓ |
| USD_JPY | 2.5978 | 2.2813 | 1.9459 | ✓ |
| **0050_TW** | **6.8347** | **10.3195** | **7.5071** | **✗** |
| 2317_TW | 8.3186 | 7.8338 | 6.6660 | ✓ |
| 2330_TW | 7.9338 | 7.1627 | 5.8629 | ✓ |
| 2454_TW | 9.7847 | 8.8069 | 7.5864 | ✓ |
| 2603_TW | 13.7978 | 12.8879 | 10.2181 | ✓ |
| 2881_TW | 6.8140 | 6.4564 | 5.2176 | ✓ |

### MDA 觀察

Persistence 的 MDA 在所有資料集上幾乎為 0（因為它永遠預測「不變」，方向正確率極低）。
這反過來驗證 Linear / TSB 的高 MDA（~0.62 / ~0.72）是真實的方向預測能力，而非基準過低所致。

## 結論

### 正面：基礎穩固（15/16）
- Linear vanilla 在 15/16 資料集上均能勝過 Persistence，paper 的基礎主張站得住腳。
- FX 類資料集尤其強（10/10 全勝），改善幅度達 9–12%。

### 需要處理：0050_TW 異常

0050_TW 上 Linear RMSE（10.32）比 Persistence（6.83）差了 **51%**，是唯一例外。

**根本原因：未還原的除權事件，非模型能力問題。**

資料調查顯示 2014-01-02 出現一筆 log_return = −1.39（對應股價從 37.41 跌至 9.33，即 4:1 股票分割未做 adjusted close），造成當期滾動波動率飆升至 833，屬於資料品質問題而非真實市場事件。

此極端值導致 StandardScaler 的 scale 參數從正常的 ~13.83 被拉升至 42.20（約 3 倍失真），測試集的正常波動率值（~20）在 scaled space 中被壓縮至 0.04σ 附近，模型喪失分辨能力。

**對現有 DL 實驗的影響：**
所有 DL 模型（Linear vanilla、TSB、各 backbone）共用同一個失真的 scaler，因此模型之間的**相對比較結果不受影響**——TSB 在此資料集仍有 27% 的改善。受影響的是 Persistence 等不依賴 scaler 的方法與 DL 方法之間的絕對指標比較，兩者處於不同尺度基準，不具可比性。

**建議論文用語（Limitation section）：**

> 0050_TW 資料集含有一筆未還原的除權事件（2014-01-02，log_return = −1.39），使 StandardScaler 的標準差被極端值拉升至正常水準的三倍。此問題對所有 DL 模型的相對比較不造成影響（共用同一 scaler），但使得 Persistence 等不依賴 scaler 的方法與 DL 方法之間的絕對指標在此資料集上失去可比性。這是本研究採用原始未調整股價資料的侷限之一，後續研究宜使用還原股價（adjusted close）以確保資料一致性。

### 後續行動（Phase 2）
學長建議的其他必補基線（依優先序）：
1. EWMA (λ=0.94)
2. HAR-RV (Corsi, 2009)
3. GARCH(1,1)
4. EGARCH / GJR-GARCH

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `persistence_baseline.py` | 計算 Persistence 並與 Linear/TSB 比較的主腳本 |
| `persistence_results.json` | 16 個資料集的逐筆數值（RMSE/MAE/MDA × 三個方法） |
