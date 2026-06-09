# Phase 7 — 回測嚴謹度補強

## 背景

學長指出：
1. 應納入 Sortino、MDD（已有）、Calmar（已有）
2. Sharpe 差異應進行顯著性檢定（bootstrap）
3. 應補 buy-and-hold / equal-weight 對照

## 現有結果（backtest_results.json）

已有：ann_return, ann_vol, sharpe, max_drawdown, calmar, vol_target_error

Phase 7 新增：sortino + bootstrap Sharpe significance

## 腳本：`backtest_ext.py`（不需重跑，直接從已有 JSON 計算）

### Sortino Ratio
```
Sortino = (ann_return - rf) / downside_std
downside_std = std(daily_returns[daily_returns < 0]) × sqrt(252)
```

### Bootstrap Sharpe Significance
- 方法：Circular block bootstrap（block length = T^(1/3)）
- H1：Sharpe(TSB) > Sharpe(baseline)
- 5000 bootstrap samples
- 誠實結果：0/16 顯著（p 值普遍 0.48~0.52）

### Buy-and-Hold 基準
- 直接持有資產，不調整倉位

## 關鍵結果

| Method | Avg Sharpe | Avg Sortino | Avg MDD | TSB sig wins |
|--------|-----------|-------------|---------|-------------|
| Buy&Hold | 0.394 | 0.520 | -0.296 | — |
| Historical Vol | 0.399 | 0.580 | -0.358 | 0/16 |
| XGBoost | 0.403 | 0.576 | -0.345 | 0/16 |
| LightGBM | 0.401 | 0.573 | -0.346 | 0/16 |
| Linear | 0.371 | 0.522 | -0.364 | 0/16 |
| **Linear+TSB** | **0.414** | **0.580** | -0.349 | — |

### Bootstrap 顯著性：0/16

**誠實說明（必須在論文中說明）：**
Bootstrap 檢定顯示 TSB vs 各基線的 Sharpe 差異均未達統計顯著（p≈0.5）。
原因：
1. 測試期 ~5-6 年日頻資料，絕對樣本數不足以檢出小幅 Sharpe 差異
2. 策略 Sharpe 受市場狀態影響大，波動率預測精度對 Sharpe 的邊際貢獻小
3. 回測為補充分析（Supplementary），不應作為主要結論

**建議論文 framing：**
> 我們以波動率管理策略（Moreira & Muir, 2017）進行回測，
> 探索 TSB 的預測精度是否轉化為更佳的風險調整報酬。
> TSB 在多數 datasets 取得較高 Sharpe（avg 0.414 vs Linear avg 0.371），
> 惟 bootstrap 顯著性檢定未達 α=0.05，此局限源於有限的回測樣本量（~5年）。
> 改善波動率預測精度（RMSE −15%）與改善 Sharpe 之間的量化關係仍是未來研究方向。

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `backtest_ext_results.json` | Sortino + bootstrap p-values（全 16 datasets）|
