# Phase 4 — RevIN 公平性修正

## 背景

學長指出：論文 4.4.3 節宣稱「TSB 未使用 RevIN 仍能勝出 PatchTST（使用 RevIN），佐證架構優勢」——
此論證有缺陷：比較混淆了「架構」與「正規化方案」兩個因素。

**問題根源：**
- PatchTST 使用 RevIN（instance-level 正規化）
- DLinear、iTransformer、TSB 均使用全域 Z-score（fit on train only）
- 前處理不一致 → 架構比較不公平

**假設（學長版）：** RevIN 在 30 天短窗、高自相關的波動率任務中，
可能抹除 level 資訊，反而不利。

## 修改內容

### `src/models/sota_baselines.py`
- 新增 `use_revin: bool = True` 參數至 `PatchTST.__init__`
- `self.revin = RevIN(n_features) if use_revin else None`
- Forward 中條件式 normalize / denormalize

### `revision/phase4_revin_fix/rerun_patchtst_no_revin.py`
- 對 16 個 datasets × 5 seeds 重跑 `PatchTST(use_revin=False)`
- 與原始 PatchTST（w/ RevIN）及 TSB 三方比較

## 關鍵結果

**RevIN 對 PatchTST 的效果（實測）：**

| Dataset | PatchTST w/RevIN | PatchTST no RevIN | RevIN 效果 | TSB 勝出(no RevIN) | TSB 勝出(orig) |
|---------|-----------------|------------------|-----------|-------------------|----------------|
| USD_TWD | 1.7881 | 2.7965 | -56.4% | ✓ | ✓ |
| JPY_TWD | 2.5135 | 3.1009 | -23.4% | ✓ | ✓ |
| KRW_TWD | 1.8985 | 2.1994 | -15.8% | ✓ | ✓ |
| CAD_TWD | 2.0043 | 2.3702 | -18.3% | ✓ | ✓ |
| CNY_TWD | 1.8499 | 3.4093 | -84.3% | ✓ | ✓ |
| EUR_TWD | 2.0308 | 2.4474 | -20.5% | ✓ | ✓ |
| THB_TWD | 2.0027 | 2.3437 | -17.0% | ✓ | ✓ |
| EUR_USD | 1.5913 | 2.4386 | -53.3% | ✓ | ✓ |
| GBP_USD | 1.8688 | 2.1736 | -16.3% | ✓ | ✓ |
| USD_JPY | 2.3624 | 3.1673 | -34.1% | ✓ | ✓ |
| 0050_TW | 6.9266 | 7.5768 | -9.4% | ✓ | ✗ |
| 2317_TW | 7.8791 | 8.4682 | -7.5% | ✓ | ✓ |
| 2330_TW | 7.1407 | 7.5773 | -6.1% | ✓ | ✓ |
| 2454_TW | 9.1181 | 9.9048 | -8.6% | ✓ | ✓ |
| 2603_TW | 13.0628 | 14.7543 | -13.0% | ✓ | ✓ |
| 2881_TW | 6.2596 | 7.3618 | -17.5% | ✓ | ✓ |

**結論（與學長假設相反）：**
- RevIN 在所有 16 個 datasets 上均**有益**於 PatchTST（移除 RevIN 後 RMSE 上升 6–84%）
- TSB 仍勝出 PatchTST(no RevIN) 16/16、PatchTST(w/RevIN) 15/16
- 此結果**強化**論文立場：TSB 即使在 PatchTST 最佳設定（w/ RevIN）下仍勝出

## 論文修改建議

**原版論證（有問題）：**
> TSB 未使用 RevIN 仍能勝出使用 RevIN 的 PatchTST，佐證架構優勢。

**修正版論證（誠實 + 更強）：**
> 我們額外以 use_revin=False 重跑 PatchTST（消除正規化差異），
> 結果顯示 RevIN 對 PatchTST 始終有益（16/16 datasets），
> TSB 仍在兩種設定下均勝出（15/16 w/RevIN，16/16 no RevIN）。
> 此結果排除「TSB 受益於 RevIN 缺席」之替代解釋。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `rerun_patchtst_no_revin.py` | 重跑腳本（16 datasets × 5 seeds） |
| `patchtst_no_revin_results.json` | 原始數值 |
| `patchtst_revin_summary.csv` | 三方比較彙整表 |
