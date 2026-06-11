# Phase 11: New SOTA Baselines (學長建議補強清單)

對應 `project_revision_9points.md` 點3「Baseline 名單偏舊」中，原本決定暫緩、
2026-06-10 重新決定要做的補強模型。分成兩部分：

## A. 新SOTA架構（已實作，待執行）

`src/models/sota_baselines_v2.py` 新增 5 個模型，介面與
`src/models/sota_baselines.py` 既有的 DLinear/PatchTST/iTransformer/TimeMixer
完全一致（`forward(x) -> (pred, pred, None, None, None)`），可直接套用
`train_baseline()` 訓練流程：

| 模型 | 來源 | 核心機制 | 參數量 (30→3, 2 channels) |
|------|------|----------|------|
| Autoformer | NeurIPS 2021 | 序列分解 + Auto-Correlation (FFT-based) | 21,444 |
| FEDformer | ICML 2022 | 頻域增強 Transformer (FEB-f / FEA-f) + MOE分解 | 58,428 |
| ETSformer | arXiv 2022 | 指數平滑 (level/growth/damping) + 頻域季節外推 | 6 |
| LightTS | arXiv 2022 | 連續採樣 + 間隔採樣 MLP (IEBlock) | 14,099 |
| TimesNet | ICLR 2023 | FFT週期偵測 + 1D→2D Inception卷積 | 73,491 |

> 所有架構皆已 forward+backward smoke test 通過，並用 USD_TWD 跑過 3-epoch
> 訓練驗證 pipeline 可正常運作。

### 執行方式

```bash
# 全部跑（5模型 x 16資料集 x 5 seeds = 400 runs，可中斷續跑）
uv run python revision/phase11_new_sota/run_new_sota.py

# 只跑特定模型
uv run python revision/phase11_new_sota/run_new_sota.py --models Autoformer TimesNet

# 只看分析（讀取已存的 JSON）
uv run python revision/phase11_new_sota/run_new_sota.py --analyze
```

結果存到 `revision/phase11_new_sota/new_sota_results.json`（不影響原本的
`experiments/comprehensive_results/sota_baselines/sota_baselines_results.json`）。
`analyze()` 會合併讀取兩個檔案，輸出：
- 16資料集 x 8方法（Linear, Linear+TSB, DLinear, PatchTST, iTransformer + 5新模型）RMSE表
- TSB vs 每個新模型的勝場數 + 顯著性檢定 + 平均差距%
- 三大類別（Taiwan Stocks / TWD FX / International FX）平均RMSE
- 平均排名、平均RMSE/MAE/MDA

### LR 選擇（**未做grid search**）

5模型 x 16資料集 x 5seeds 已經是 ~400 次訓練；若再做 LR grid search
（仿 phase10，3 seeds x 6 LR）會再乘上好幾倍計算量，使用者決定本輪先跳過。

採用各模型原始論文/官方repo的預設LR（適合短序列任務的版本）：

| 模型 | LR | 依據 |
|------|-----|------|
| Autoformer | 1e-4 | 官方repo ETT預設 |
| FEDformer | 1e-4 | 官方repo ETT預設 |
| ETSformer | 1e-3 | 官方repo預設 |
| LightTS | 1e-3 | 官方repo預設 |
| TimesNet | 1e-3 | 官方repo短期預測預設 |

**若某模型結果異常差**（例如RMSE遠高於Vanilla Linear），很可能是LR不適合
這個短窗金融任務 —— 可參考 `revision/phase10_lr_tuning/lr_grid_search.py`
的模式，針對該模型單獨補做 grid search（改 import 來源即可）。

### ETSformer 參數量極小 (6) 的說明

ETSformer 的核心是「level/growth 指數平滑遞迴 + 頻域季節性外推」，可學習
參數只有 alpha/beta/damping 三組 per-channel scalar（呼應 TSB 論文中
「decomp weights / SNR gate 也都是 global scalar」的設計哲學）。這是刻意
簡化但忠實呈現論文核心機制的版本，不是bug。如果跑出來RMSE異常好或異常差，
都值得在論文討論中提及「極簡可學習參數量」這個對比點。

---

## B. Foundation Models Zero-shot（已寫好腳本，需安裝額外依賴）

`revision/phase11_new_sota/foundation_zeroshot.py` — TimesFM / Chronos /
Moirai 三個預訓練模型的 zero-shot 比較（不訓練，只用預訓練權重在我們的
test set 上推論）。

### 安裝（擇一或全部，注意可能與 torch==2.7.1+cu118 有版本衝突，建議先在
獨立 venv 測試）

```bash
# Chronos (Amazon, T5-based)
uv pip install chronos-forecasting

# TimesFM (Google)
uv pip install timesfm

# Moirai (Salesforce)
uv pip install uni2ts gluonts
```

### 執行方式

```bash
uv run python revision/phase11_new_sota/foundation_zeroshot.py --model chronos
uv run python revision/phase11_new_sota/foundation_zeroshot.py --model timesfm
uv run python revision/phase11_new_sota/foundation_zeroshot.py --model moirai
uv run python revision/phase11_new_sota/foundation_zeroshot.py --model all
uv run python revision/phase11_new_sota/foundation_zeroshot.py --analyze
```

### 重要提醒

- **這些套件目前環境中都沒裝**，腳本是依照各自官方文件的 zero-shot API
  寫的（撰寫時間 2026-06），但這些 repo 更新很快，實際執行時某個模型的
  API 若有變動，**只需要修該模型對應的 `run_xxx()` function**，互不影響。
- 結果存到 `revision/phase11_new_sota/foundation_zeroshot_results.json`，
  `--analyze` 會跟 Linear / Linear+TSB 的5-seed平均RMSE一起列表比較。
- Zero-shot 只用 Volatility 單變量（context=30步原始尺度數值，預測3步），
  跟訓練式baseline的test split（後20%）一致，但**不做z-score正規化**
  （foundation model通常自己做normalization）。
- TimesFM v1 要求 context_len 是 32 的倍數，腳本已用「複製第一個值往前補
  到32步」的方式處理（`PAD_LEN=32`）。

---

## 與論文整合

這些是「reviewer點3新增模型」的延伸實驗，**目前定位為 supplementary**：
- 若A部分（5個訓練式SOTA）TSB依然全勝或大多勝 → 可以加強「vs更新的SOTA
  名單依然領先」的論述，放進主要SOTA表格或附錄表格皆可
- 若B部分（zero-shot foundation models）RMSE明顯劣於所有訓練式方法
  （預期如此，30→3短窗 + 金融波動率非foundation model強項）→ 可以寫成
  「zero-shot foundation models在這類短窗高頻金融任務上不具優勢，凸顯
  task-specific訓練的必要性」，呼應原本「future work」段落但改成有數據
  支持的討論
