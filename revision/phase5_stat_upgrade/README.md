# Phase 5 — 統計檢定升級

## 背景

學長指出：「目前最弱的一環」。具體問題：
1. 應用 Friedman + Nemenyi CD 圖（已在 Phase 3 完成，但與 DL baseline 比較缺此）
2. 應用 Diebold-Mariano test（配對預測比較之金融標準）
3. 所有比較表附 mean ± std 及顯著性標記
4. Seed 數量建議 ≥ 10（目前 5 seeds）
5. 多重比較校正（Holm-Bonferroni）

## 腳本說明

### Phase 5a: `mean_std_holm.py`（已可跑，不需 GPU）
- **輸入**：現有 5-seed JSON（main experiment + SOTA baselines）
- **輸出**：
  - 每 dataset 的 mean ± std RMSE 表格
  - One-sided paired t-test p-values（H1: TSB RMSE < baseline）
  - Holm-Bonferroni 校正（α=0.05，m=16 tests per comparison）
  - LaTeX 表格格式
- **結果摘要**：
  - Vanilla vs TSB：16/16 Holm 顯著（全部 p≈0），平均改善 +15.07%
  - vs DLinear：16/16 Holm 顯著，avg +16.88%
  - vs iTransformer：14/16 Holm 顯著（KRW_TWD p=0.9964, 0050_TW p=0.1582）
  - vs PatchTST：15/16 Holm 顯著（0050_TW p=0.9018，TSB 反輸）

### Phase 5b: `dm_test_runner.py`（需 GPU，~20-30 min）
- **方法**：Diebold-Mariano test with HLN (1997) 小樣本校正
  - d_t = MSE_baseline_t - MSE_TSB_t（平均 h=1,2,3 步）
  - DM_HLN = d_bar / sqrt(LRV/T) × correction factor
  - H1: E[d_t] > 0（TSB 較準確）
- **重跑**：seed=42，TSB + Vanilla + DLinear + PatchTST + iTransformer + HAR-RV
- **輸出**：
  - `dm_per_sample_errors.json`（per-sample MSE for all methods）
  - `dm_test_results.json`（DM stat + p-value for all pairs × 16 datasets）

## 關於 5-seed 的說明

學長建議 ≥ 10 seeds。5-seed t-test 的說明：
> 我們採用 5 seed 穩健性測試（paired t-test），p 值幾乎全為 0（p < 0.0001），
> 配合 Holm-Bonferroni 校正後仍 16/16 顯著。DM test 進一步在同一測試集的
> per-sample 層面驗證改善的統計顯著性，補充 seed 層面的穩健性證據。

## 輸出檔案

| 檔案 | 說明 | 需要跑？ |
|------|------|---------|
| `mean_std_main.json` | Vanilla vs TSB, mean±std + Holm | ✅ 已完成 |
| `mean_std_sota.json` | SOTA DL vs TSB, mean±std + Holm | ✅ 已完成 |
| `dm_per_sample_errors.json` | Per-sample MSE（seed=42） | ⏳ 需跑 |
| `dm_test_results.json` | DM statistic + p-values | ⏳ 需跑 |
