# Backtest Return-Convention Fix (2026-08-03)

## 問題

原始 `experiments/backtest.py` 的策略日報酬定義為 `w * log_return`（`run_inv_vol_strategy`），
但後續以 `np.cumprod(1 + strategy_returns)` 計算 NAV／最大回撤／累積報酬——
`cumprod(1+·)` 是 **simple return** 的複利公式，把 log return 塞進去在數學上不一致，
得到的累積報酬不對應任何可實際執行的策略（外部稽核 P0-2 確認）。

論文舊數字（例：2603 累積報酬 +17.4% → +72.5%、相對 NAV 提升 +46.9%）即由此混用口徑產生。

## 修正

1. `experiments/backtest.py::run_inv_vol_strategy`：策略日報酬改為
   `w * np.expm1(log_return)`（持有權重 w 一日的真實損益），與 `cumprod(1+·)` 口徑一致。
2. 本目錄 `recompute_from_daily.py`：不重新訓練，直接以
   `experiments/comprehensive_results/backtest/backtest_daily_returns.json`
   歸檔的逐日序列還原權重（`strat / actual`，已驗證全部落在裁剪區間 [0.2, 3.0]）
   後套用修正口徑重算，輸出 `backtest_corrected_results.json`。

注意：`backtest_daily_returns.json` 內既有的策略序列仍是舊口徑（`w * log_return`）；
若日後重跑 `backtest.py`，寫出的序列會直接是 simple-return 口徑，本目錄的
還原步驟屆時不再需要。

## 修正後結果（Linear vs Linear+TSB）

- Sharpe 提升維持 **13/16**；下降的三個資料集不變：2330 (−0.080)、2881 (−0.027)、CAD/TWD (−0.019)。
- Sharpe 提升前三名變為：**2603 +0.291、EUR/USD +0.102、EUR/TWD +0.090**（0050 +0.085 退居第四）。
- 2603 累積報酬：**+62.1% → +125.6%**（相對 NAV 提升 **+39.2%**）。
- 0050 累積報酬：**+181.2% → +243.9%**（相對 NAV 提升 **+22.3%**）。
- 論文 `article_v1.tex` 之 §4.3（回測協定）、§4.5.8 與結論已同步更新，
  `fig_backtest_sharpe.pdf` 已以修正口徑重製。

## 驗證

以舊口徑重算可逐位還原論文舊數字（2603 +17.4%/+72.5%/+46.9%、Sharpe 差 +0.262 等），
確認差異完全來自報酬口徑而非資料或對齊問題；預測—次日報酬之索引對齊經稽核無前視偏誤。
