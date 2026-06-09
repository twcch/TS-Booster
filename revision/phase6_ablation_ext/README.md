# Phase 6 — 消融補完

## 背景

學長指出兩項消融缺失：

### 6a: 參數量對齊對照組
> 應增加參數量對齊之對照組：為 Linear 配置一相同約 7K 參數之樸素 head，
> 以證明增益源自架構設計而非參數量之增加。此項直接支撐論文之核心論點，重要性甚高。

現況：
- Vanilla Linear：366 params
- TSB-Linear：7,156 params（多 6,790 params = +1855%）

若 TSB 贏的原因只是「多了參數」，那 +15% 的結論就站不住腳。

### 6b: SNR Gate 隔離
> 應單獨拆解 SNR Gate（分別替換為常數、可學純量及無 gate 之設定）；
> 目前其貢獻與 decomp 合併計入 +1.8% 中，未予明確區分。

## 腳本說明

### `param_matched_baseline.py`（需 GPU，~40-60 min）

**模型設定：**
- Vanilla Linear：366 params（原始基線）
- **MLP-7K**：7,171 params，2-hidden-layer MLP [60→64→48→3]，純增加容量
- TSB-Linear：7,156 params（架構設計：decomp + inception expert + SNR gate）

**關鍵假設：**
> IF TSB > MLP-7K：增益來自架構設計
> IF TSB ≈ MLP-7K：增益可能只是參數量的效果

### `snr_gate_ablation.py`（需 GPU，~60-80 min）

4 種 gate 變體，其餘配置完全相同（decomp + trend enrichment + expert）：

| Config | Gate 說明 | Params |
|--------|-----------|--------|
| No gate | gate=1.0（永遠加入 expert） | 7,154 |
| Const-0.5 | gate=0.5（永遠半衰減） | 7,154 |
| Learned | 單一可學純量，sigmoid 後得 gate | 7,155 |
| SNR gate (TSB) | 每 sample 的 SNR 比值驅動 gate | 7,156 |

**說明：**
- No gate ≈ +Expert 設定（decomp 版），是 SNR gate 的對照組
- Const-0.5 驗證 gate 值本身的重要性（固定 < 1 是否有益）
- Learned 驗證「可學 vs 輸入依賴」
- SNR gate 是 TSB 的 production 設定

## 輸出檔案

| 檔案 | 說明 | 需要跑？ |
|------|------|---------|
| `param_matched_results.json` | Vanilla / MLP-7K / TSB（5 seeds × 16 datasets） | ⏳ 需跑 |
| `snr_gate_ablation_results.json` | 4 gate configs（5 seeds × 16 datasets） | ⏳ 需跑 |
