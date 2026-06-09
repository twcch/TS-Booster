# Phase 3 — Integration：統整比較表 + CD Diagram

## 目的

將 Phase 1（Persistence）、Phase 2（EWMA / HAR-RV / GARCH / GJR-GARCH）、
以及原始主實驗（Linear vanilla / TSB）的結果整合為一張完整的比較表，
並重跑 Friedman + Nemenyi 統計檢定。

## 所有方法一覽

| 方法 | 類型 | 備注 |
|------|------|------|
| Persistence | 統計 | Naive carry-forward |
| EWMA | 統計 | λ=0.94，RiskMetrics |
| HAR-RV | 統計 | Corsi (2009)，gold standard |
| GARCH(1,1) | 統計 | 條件異方差 |
| GJR-GARCH | 統計 | 非對稱 GARCH |
| Linear | DL | Vanilla backbone，5-seed 均值 |
| TSB | DL | TS-Booster，5-seed 均值 |

## 主要結果

### 平均 RMSE Rank（1 = best）

| 排名 | 方法 | 平均 Rank |
|------|------|---------|
| 1 | **TSB** | **1.12** |
| 2 | Linear | 2.25 |
| 3 | HAR-RV | 2.81 |
| 4 | Persistence | 3.88 |
| 5 | EWMA | 5.25 |
| 6 | GJR-GARCH | 6.31 |
| 7 | GARCH(1,1) | 6.38 |

### 平均 RMSE（16 datasets）

| 方法 | 平均 RMSE | vs TSB |
|------|---------|--------|
| TSB | **3.7622** | — |
| Linear | 4.5699 | TSB −17.6% |
| HAR-RV | 4.4039 | TSB −14.6% |
| Persistence | 4.7332 | TSB −20.5% |
| EWMA | 5.9501 | TSB −36.8% |
| GJR-GARCH | 7.6285 | TSB −50.7% |
| GARCH(1,1) | 7.5801 | TSB −50.4% |

### MDA 觀察

- **Persistence MDA ≈ 0**：永遠預測「不變」，沒有方向預測能力
- **TSB MDA = 0.708**（最高），正確方向比率遙遙領先所有統計基線
- EWMA / HAR-RV / GARCH 的 MDA 約 0.57–0.59，僅略優於隨機（0.5）

## Friedman + Nemenyi 檢定

| 項目 | 數值 |
|------|------|
| Friedman χ²(6) | 86.76 |
| p-value | < 0.000001 |
| Nemenyi CD (α=0.05, k=7, n=16) | 2.25 |

### 顯著優於 TSB 的方法
**無**

### TSB 顯著優於的方法
| 方法 | rank diff | 顯著? |
|------|-----------|-------|
| Persistence | 2.75 | **SIG** |
| EWMA | 4.13 | **SIG** |
| GARCH(1,1) | 5.25 | **SIG** |
| GJR-GARCH | 5.19 | **SIG** |
| HAR-RV | 1.69 | n.s.（差距不足 CD=2.25） |
| Linear | 1.13 | n.s. |

### HAR-RV vs TSB 的說明

HAR-RV 是所有統計基線中最強的（rank 2.81），TSB 排名第一（rank 1.12），差距 1.69 < CD=2.25，
**統計上不顯著**。這表示 TSB 在 16 個 dataset 的整體排名上確實更好，但無法在 α=0.05 下宣稱顯著優於 HAR-RV。

這是一個誠實但可接受的結果：
- 勝率層面：TSB 15/16 勝出（可用 exact binomial test 補充）
- 幅度層面：平均改善 +13.97% RMSE（具體數字有意義）
- 方向預測：MDA 0.708 vs 0.587（差距明顯）
- 統計層面：Nemenyi 未達顯著，需誠實說明

## 論文修改建議

### 結論 framing 的更新

**原版：**
> TSB 相較 vanilla Linear 改善約 15%，16/16 datasets 顯著優於。

**建議版：**
> TSB 在所有方法中排名第一（avg rank 1.12），顯著優於 Persistence、EWMA、GARCH、
> GJR-GARCH（Friedman p < 0.001，Nemenyi 顯著），並在 15/16 datasets 上勝出
> HAR-RV（+13.97% RMSE）。TSB vs HAR-RV 的 Nemenyi 未達 α=0.05，
> 但勝率與 MDA 差距（0.708 vs 0.587）提供了補充證據。

### GARCH 表現差的說明

GARCH 系列（rank 6–7）表現遠差於預期，原因是預測目標為 7 日**滾動**波動率而非瞬時條件方差。
GARCH 的 h 步預測是均值回歸的條件方差，與高度平滑的滾動目標在定義上不匹配，應在論文中說明。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `integrate.py` | 整合腳本：載入 P1+P2+主實驗結果，輸出表格 + CD diagram |
| `integration_results.json` | 完整數值（avg_ranks、Friedman stats、per-dataset） |
| `cd_diagram_all_baselines.png` | 7-method CD diagram |
