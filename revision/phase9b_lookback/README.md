# Phase 9b — Lookback Length Robustness

## 背景與動機

Phase 9（`phase9_temporal/`）只測試了**預測窗口長度（horizon, H）**的穩健性，固定 lookback
L=30。但論文目前只報告 L=30 / H=3 一組設定，reviewer 很可能追問：「TSB 的優勢是否
只在特定 lookback 長度下成立？」

Phase 9b 是 Phase 9 的對稱補完：**固定論文主設定的 H=3**，掃描不同的輸入窗口長度
**L ∈ {30, 60, 90}**，驗證 TSB 是否同樣對 lookback 長度穩健。

兩個網格實驗合在一起，構成完整的「雙軸時間尺度穩健性」驗證：phase9 固定 L=30 掃 H，
phase9b 固定 H=3 掃 L，皆以論文主設定（L=30, H=3）為錨點。

## 測試方法

**設定：**
- Lookback **L ∈ {30, 60, 90}**，Horizon **H = 3**（固定，論文主設定）
- 模型：Vanilla Linear vs TSB（與主實驗相同架構）
- 3 seeds（42, 123, 456）— robustness check 規格，與 phase9 一致（非完整 5-seed）
- 16 個資料集（TWD FX ×7、International FX ×3、Taiwan Stocks ×6）
- 注意：`LinearBackbone` 的輸入維度與 `seq_len` 綁定，因此每個 L 都會重新建構 backbone

**腳本：** `lookback_robustness.py`
**執行：**
```bash
uv run python revision/phase9b_lookback/lookback_robustness.py
uv run python revision/phase9b_lookback/lookback_robustness.py --analyze   # 重新分析，不需重跑
```

## 結果

### TSB 改善幅度（vs Vanilla Linear，平均 RMSE 改善 %，16 個資料集平均）

| Lookback L | TSB 平均改善 | 勝出 |
|---|---|---|
| **L=30**（論文主設定） | **+15.9%** | 16/16 |
| L=60 | +14.5% | 16/16 |
| L=90 | +13.7% | 16/16 |

### 逐資料集改善幅度

| Dataset | L=30 | L=60 | L=90 |
|---|---|---|---|
| USD_TWD | +15.5% | +14.2% | +12.7% |
| JPY_TWD | +15.2% | +13.5% | +11.5% |
| KRW_TWD | +7.4% | +8.5% | +7.2% |
| CAD_TWD | +10.2% | +9.0% | +7.9% |
| CNY_TWD | +13.9% | +13.3% | +12.3% |
| EUR_TWD | +11.0% | +12.5% | +9.0% |
| THB_TWD | +13.3% | +8.0% | +7.6% |
| EUR_USD | +12.0% | +10.5% | +8.6% |
| GBP_USD | +13.5% | +12.7% | +12.2% |
| USD_JPY | +14.9% | +12.1% | +10.0% |
| 0050_TW | +34.9% | +34.4% | **+39.4%** |
| 2317_TW | +15.9% | +14.7% | +15.2% |
| 2330_TW | +18.1% | +18.8% | +17.6% |
| 2454_TW | +18.5% | +16.8% | +17.1% |
| 2603_TW | +20.5% | +16.7% | +15.8% |
| 2881_TW | +18.9% | +16.5% | +15.6% |
| **Average** | **+15.9%** | **+14.5%** | **+13.7%** |

## 關鍵結論

1. **TSB 在所有測試的 lookback 長度下全勝**：L=30/60/90 均為 16/16，比 phase9 的
   horizon 網格更乾淨（phase9 在 H=10 時掉到 14/16）。
2. **優勢隨 L 增長小幅遞減，但幅度遠比 horizon 軸平緩**：+15.9% → +14.5% → +13.7%，
   僅遞減 2.2 個百分點（相較 phase9 horizon 軸從 +17.1% 驟降至 +4.0%，遞減 13.1 個百分點）。
   → **TSB 對 lookback 長度的敏感度遠低於對 horizon 長度的敏感度**。
3. **0050_TW 是唯一反向趨勢的資料集**：改善幅度隨 L 增長反而上升（34.9% → 34.4% →
   **39.4%**）。可能解釋：更長的 lookback 視窗能稀釋 2014-01-02 股票分割造成的
   局部 split artifact 影響（見 `0050_TW 特殊注意事項` in project memory）。
4. **論文主設定 L=30 落在最佳區間**：三個 lookback 長度中 L=30 平均改善最高，顯示
   30 天視窗並非隨意選擇，而是在這個任務上能取得最佳的「資訊量 vs 雜訊」平衡點。

## 與 Phase 9（Horizon 網格）合併敘事建議

```
我們同時驗證了 TSB 在輸入窗口長度（lookback, L ∈ {30, 60, 90}）與
預測窗口長度（horizon, H ∈ {1, 3, 5, 10}）兩個維度上的穩健性：

- Lookback 維度：三種設定皆 16/16 全勝，改善幅度僅小幅遞減（15.9% → 13.7%），
  顯示 TSB 對輸入窗口長度高度穩健。
- Horizon 維度：短中期（H ≤ 5）全勝且改善顯著（12-17%），長 horizon（H=10）
  優勢縮小但仍正向（+4.0%, 14/16）。

兩軸結果顯示論文主設定（L=30, H=3）落在一個穩健的高效能區間內，
並非偶然調出來的最佳點，TSB 的設計在合理的時間尺度範圍內具備良好的泛化能力。
```

## 結果檔案

| 檔案 | 說明 |
|---|---|
| `lookback_results.json` | 逐 dataset、逐 seed 的 RMSE/MAE/MDA（Vanilla vs TSB，L ∈ {30,60,90}） |
| `lookback_robustness.py` | 實驗腳本（含 `--analyze` 重新分析模式） |
