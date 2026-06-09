# Revision — 期刊投稿升級實驗

9 個 reviewer 等級的 revision 要點，對應此目錄下各 phase 的實驗。

---

## 對應總表

| 點 | 問題 | 對應 Phase | 狀態 |
|----|------|-----------|------|
| 1 | Persistence / HAR-RV / GARCH / EWMA 基線缺失 | phase1, phase2, phase3 | ✅ 已完成 |
| 2 | RevIN 比較不公平（架構 vs 正規化混淆） | phase4 | ✅ 已完成 |
| 3 | Baseline 欠調、SOTA 名單偏舊 | phase10 | ✅ 已完成 |
| 4 | 時間尺度穩健性（H × L 網格） | phase9, phase9b | ✅ 已完成（H 與 L 雙軸網格皆已跑完） |
| 5 | 統計檢定升級（DM test / Friedman+CD / Holm） | phase3, phase5 | ✅ 已完成 |
| 6 | 回測嚴謹度（Sortino / bootstrap / buy-and-hold） | phase7 | ✅ 已完成 |
| 7 | 消融補完（SNR gate 隔離 / param-matched） | phase6 | ⚠️ 部分完成（K 掃描、單核 vs Inception、loss 拆解仍缺） |
| 8 | 跨市場驗證（美股、加密） | — | ⚠️ 未規劃 |
| 9 | 計算成本完整性（FLOPs / memory） | phase8 | ✅ 已完成 |

---

## 各 Phase 詳細說明

---

### Phase 1 — Persistence Baseline ✅
**對應點 1（部分）**

**實驗內容：**
最簡單的 carry-forward 基線（σ_t 直接作為未來 3 天預測），驗證 Linear vanilla 是否有意義。

**關鍵結論：**
- Linear > Persistence：15/16（唯一例外 0050_TW 為資料品質問題）
- FX 類全勝（10/10），改善 9–12%
- Persistence MDA ≈ 0，顯示 TSB 的高 MDA（~0.72）是真實方向預測能力

**0050_TW 異常原因：**
2014-01-02 未複權股票分割（-75%，log_return = -1.39），導致 rolling vol 飆至 834%，StandardScaler scale 失真 3 倍，DL 模型喪失分辨能力。Persistence 不依賴 scaler 故不受影響，比較失去可比性。

**結果檔案：** `phase1_persistence/persistence_results.json`

---

### Phase 2 — Vol Baselines ✅
**對應點 1（完成）**

**實驗內容：**
四個波動率專用統計基線，EWMA / HAR-RV / GARCH(1,1) / GJR-GARCH，16 datasets。

**關鍵結論：**

| 基線 | TSB 勝出 | 平均改善 |
|------|---------|---------|
| EWMA（λ=0.94） | 16/16 | +37.1% |
| **HAR-RV**（gold standard） | **15/16** | **+14.0%** |
| GARCH(1,1) | 16/16 | +44.2% |
| GJR-GARCH | 16/16 | +44.2% |

GARCH 系列差的原因：GARCH 建模瞬時條件異方差，本研究目標為 7 日滾動波動率，定義不匹配。

**論文建議：** 主要結論的基準從「vs Vanilla Linear +15%」改為「vs HAR-RV gold standard +14%」，更具說服力。

**結果檔案：** `phase2_vol_baselines/vol_baselines_results.json`

---

### Phase 3 — Integration（全基線整合 + CD Diagram）✅
**對應點 1、5**

**實驗內容：**
整合 Phase 1+2+主實驗，7 個方法統一比較，重跑 Friedman + Nemenyi 統計檢定。

**關鍵結論：**

| 排名 | 方法 | 平均 Rank | 平均 RMSE |
|------|------|---------|---------|
| 1 | **TSB** | **1.12** | **3.762** |
| 2 | Linear | 2.25 | 4.570 |
| 3 | HAR-RV | 2.81 | 4.404 |
| 4 | Persistence | 3.88 | 4.733 |
| 5 | EWMA | 5.25 | 5.950 |
| 6 | GJR-GARCH | 6.31 | 7.629 |
| 7 | GARCH(1,1) | 6.38 | 7.580 |

Friedman χ²=86.76，p < 0.000001（顯著）
Nemenyi CD=2.25：TSB 顯著優於 Persistence、EWMA、GARCH、GJR（rank diff > CD）；TSB vs HAR-RV rank diff = 1.69 < CD → **統計上不顯著**（需誠實說明）

**結果檔案：** `phase3_integration/integration_results.json`、`cd_diagram_all_baselines.png`

---

### Phase 4 — RevIN 公平性修正 ✅
**對應點 2**

**實驗內容：**
PatchTST 加/不加 RevIN 的對比，驗證「正規化」vs「架構」是否混淆。

**關鍵結論（與學長假設相反）：**
RevIN 在所有 16 個 datasets 均有益於 PatchTST（移除後 RMSE 上升 6–84%）。

| TSB vs | 勝出 |
|--------|------|
| PatchTST w/ RevIN（原始） | 15/16 |
| PatchTST no RevIN | **16/16** |

**論文修正：** 原版論證「未用 RevIN 仍勝出」暗示 RevIN 不利，這是錯的。應改寫為：「RevIN 確實有益於 PatchTST，TSB 在兩種設定下均勝出，排除正規化方案差異作為替代解釋。」

**結果檔案：** `phase4_revin_fix/patchtst_no_revin_results.json`、`patchtst_revin_summary.csv`

---

### Phase 5 — 統計檢定升級 ✅
**對應點 5**

**5a — Mean ± Std + Holm-Bonferroni：**
- Vanilla vs TSB：16/16 Holm 顯著（全部 p≈0）
- vs DLinear：16/16 Holm 顯著
- vs PatchTST：15/16 Holm 顯著（0050_TW TSB 輸）
- vs iTransformer：14/16 Holm 顯著

**5b — Diebold-Mariano Test（seed=42，HLN小樣本校正）：**

| vs Baseline | DM 顯著勝 (p<0.05) | 備注 |
|-------------|-------------------|------|
| Vanilla | 16/16 | |
| DLinear | 16/16 | |
| PatchTST | 15/16 | 0050_TW TSB 輸（單 seed 問題） |
| iTransformer | 11/16 | 0050_TW TSB 輸（seed 偶然，5-seed TSB 勝 +22%） |
| HAR-RV | 11/16 | TWD FX 中低波動資料集差異不顯著 |

**注意：** iTransformer 在 0050_TW 的 DM 結果（stat=-3.98）是單 seed=42 的偶然，5-seed 實驗 TSB 明確勝出（7.51 vs 9.69）。iTransformer 在此 dataset CV=38.4%（其他 1-3%），極度不穩定。

**結果檔案：** `phase5_stat_upgrade/dm_test_results.json`、`mean_std_main.json`、`mean_std_sota.json`

---

### Phase 6 — 消融補完 ✅
**對應點 7**

**6a — Param-Matched Baseline（MLP-7K）：**

| Model | Params | Avg RMSE | Wins |
|-------|--------|----------|------|
| Vanilla Linear | 366 | 4.699 | 0/16 |
| MLP-7K | ~7k | 4.770 | 0/16 |
| **TSB** | ~7k | **3.776** | **16/16** |

MLP-7K 在 10/16 datasets 甚至比 Vanilla 差，TSB 對 MLP-7K 平均改善 +18.5%（最高 +38.6%）。增益來自**架構設計**，非參數量。

**6b — SNR Gate 隔離：**

| Config | Avg RMSE | Wins |
|--------|----------|------|
| No gate | 3.826 | 0/16 |
| Const-0.5 | 3.779 | 3/16 |
| Learned scalar | 3.785 | 3/16 |
| **SNR gate（TSB）** | **3.776** | **10/16** |

SNR-based gate 最佳，no gate 最差，說明 gate 本身必要，且 SNR 輸入提供額外資訊。

**結果檔案：** `phase6_ablation_ext/param_matched_results.json`、`snr_gate_ablation_results.json`

---

### Phase 7 — 回測嚴謹度 ✅
**對應點 6**

**新增：** Sortino ratio、Bootstrap Sharpe 顯著性檢定、Buy-and-Hold 對照。

| Method | Avg Sharpe | Avg Sortino |
|--------|-----------|-------------|
| Buy-and-Hold | 0.394 | 0.520 |
| Historical Vol | 0.399 | 0.580 |
| XGBoost | 0.403 | 0.576 |
| LightGBM | 0.401 | 0.573 |
| Linear | 0.371 | 0.522 |
| **Linear+TSB** | **0.414** | **0.580** |

**Bootstrap 顯著性：0/16**（需在論文誠實說明）
原因：~5 年日頻回測樣本不足以檢出小幅 Sharpe 差異。建議 framing 為補充分析，不作主要結論。

**結果檔案：** `phase7_backtest_ext/backtest_ext_results.json`

---

### Phase 8 — 計算成本補強 ✅
**對應點 9**

| Model | Params | FLOPs(M) | Latency(ms) | Peak Mem(MB) |
|-------|--------|----------|-------------|-------------|
| Vanilla-Linear | 366 | 0.000 | 0.23 | 9.2 |
| **TSB-Linear** | 7,156 | **0.044** | **1.35** | **9.4** |
| PatchTST | 402,823 | 0.407 | 1.28 | 12.4 |
| iTransformer | 269,571 | 0.137 | 0.40 | 12.5 |

TSB 參數量比 PatchTST 少 56×，FLOPs 少 9×，記憶體 overhead < 0.3MB。

**結果檔案：** `phase8_compute_ext/compute_ext_results.json`

---

### Phase 9 — 時間尺度穩健性 ✅ 已完成（2026-06-07）
**對應點 4**

**實際內容：** H ∈ {1, 3, 5, 10} × L=30 固定，3 seeds，Vanilla vs TSB。
**腳本：** `phase9_temporal/temporal_robustness.py`
**結果檔案：** `phase9_temporal/temporal_results.json`
**結論：** 短中期 H ≤ 5 全勝改善顯著（+12~17%），長 horizon H=10 優勢縮小但仍正向（+4.0%, 14/16）

---

### Phase 9b — Lookback 長度穩健性 ✅ 已完成（2026-06-08）
**對應點 4（Phase 9 的對稱補完）**

**實際內容：** L ∈ {30, 60, 90} × H=3 固定（論文主設定），3 seeds，Vanilla vs TSB。
**腳本：** `phase9b_lookback/lookback_robustness.py`
**結果檔案：** `phase9b_lookback/lookback_results.json`
**詳細文件：** `phase9b_lookback/README.md`
**結論：** 三組 lookback 設定皆 16/16 全勝，平均改善 +15.9%/+14.5%/+13.7%，遞減幅度僅 2.2pp
（遠比 phase9 horizon 軸的 13.1pp 平緩）→ TSB 對 lookback 長度高度穩健；論文主設定 L=30
落在最佳改善區間，並非偶然調出的最佳點。0050_TW 例外（改善隨 L 增長反而上升，可能因
更長視窗稀釋 2014 股票分割 artifact）

**雙軸穩健性合併敘事：** phase9（固定 L=30 掃 H）+ phase9b（固定 H=3 掃 L）共同證明
論文主設定 (L=30, H=3) 落在一個穩健的高效能區間內。

---

### Phase 10 — Baseline LR Tuning ✅ 已完成（2026-06-05/06）
**對應點 3**

**實際內容：** DLinear / PatchTST / iTransformer 各自在 16 個 datasets 上 grid search LR，產出 `best_lr.json`（每個 baseline x dataset 的最佳 LR）並重跑評估。
**腳本：** `phase10_lr_tuning/lr_grid_search.py`
**結果檔案：** `phase10_lr_tuning/best_lr.json`, `phase10_lr_tuning/lr_tuned_results.json`

---

### 點 7（消融補完）🔄 腳本已完成，執行中（2026-06-08）
`phase6_ablation_ext/` 已有 SNR gate ablation 與 param-matched baseline，新增三支腳本（語法/煙霧測試已驗證，使用者本人執行中）：
- 掃描專家數 K（1, 2, 3, 4）→ `expert_k_scan.py` → `expert_k_scan_results.json`
- 比較單一卷積核 vs Inception 多核 → `single_vs_inception_kernel.py` → `single_vs_inception_results.json`
- 逐項拆除方向性損失與兩項 aux loss → `loss_component_ablation.py` → `loss_component_results.json`

執行指令（皆支援 `--analyze` 重新分析，不需重跑）：
```bash
uv run python revision/phase6_ablation_ext/expert_k_scan.py
uv run python revision/phase6_ablation_ext/single_vs_inception_kernel.py
uv run python revision/phase6_ablation_ext/loss_component_ablation.py
```

---

### 點 8（跨市場驗證）⏸️ 暫緩，列入 future work（2026-06-07 使用者確認）
美股指數（GSPC, DJI, SOX）/ 個股（TSMC, AAPL, NVDA, TSLA, XOM）目前無對應 phase。
論文已於 5.2 節自承此限制；使用者決定本輪 revision 不跑，直接寫進論文 future work / 後續延伸方向，不需新建 phase11。

---

## 執行優先序

```
P1（接下來要跑）：
  phase6 補完  → 專家數 K 掃描（1,2,3,4）
              → 單一卷積核 vs Inception 多核
              → 方向性損失 + 2 個 aux loss 逐項拆除
  phase11（新建）→ 跨市場驗證：美股指數 GSPC/DJI/SOX + 個股 TSMC/AAPL/NVDA/TSLA/XOM

P2（整合進論文，已有結果）：
  phase1 + phase2 + phase3  → 補充統計基線 section
  phase4                    → 修正 4.4.3 節 RevIN 論證
  phase5                    → 補 DM test table + Holm 校正
  phase6                    → 補 SNR gate 消融表 + param-matched 段落
  phase7                    → 補 Sortino + bootstrap 說明
  phase8                    → 補 FLOPs + memory 數字
  phase9 + phase9b          → 補時間尺度穩健性 section（H × L 雙軸網格）
  phase10                   → 補 baseline LR tuning 協定說明
```

---

## 結果檔案索引

| Phase | 結果檔案 | 狀態 |
|-------|---------|------|
| phase1 | `persistence_results.json` | ✅ |
| phase2 | `vol_baselines_results.json` | ✅ |
| phase3 | `integration_results.json`, `cd_diagram_all_baselines.png` | ✅ |
| phase4 | `patchtst_no_revin_results.json`, `patchtst_revin_summary.csv` | ✅ |
| phase5 | `dm_test_results.json`, `mean_std_main.json`, `mean_std_sota.json` | ✅ |
| phase6 | `snr_gate_ablation_results.json`, `param_matched_results.json` | ✅ |
| phase6 (ext) | `expert_k_scan_results.json`, `single_vs_inception_results.json`, `loss_component_results.json` | 🔄 腳本完成，使用者執行中 |
| phase7 | `backtest_ext_results.json` | ✅ |
| phase8 | `compute_ext_results.json` | ✅ |
| phase9 | `temporal_results.json` | ✅ |
| phase9b | `lookback_results.json` | ✅ |
| phase10 | `best_lr.json`, `lr_tuned_results.json` | ✅ |
