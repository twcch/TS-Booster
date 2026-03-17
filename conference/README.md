# TSB-Lite: Simplified TS-Booster for Conference Paper

Simplified version of TS-Booster for conference submission.

## Architecture

```
Input x [B, 30, 2]
  ├── Backbone(x)          → y_base [B, 3]
  ├── MA_7(x)              → trend  [B, 30, 2]
  └── Expert([x, trend])   → y_expert [B, 3]

y_final = y_base + y_expert

Loss = MSE(y_final, y_true) + 0.3 * MSE(y_expert, y_true - y_base.detach())
```

### Simplifications vs Thesis (Full TS-Booster)

| Component | Thesis | Conference |
|---|---|---|
| Decomposition | Learnable multi-scale MA [7,13] | Fixed single MA (k=7) |
| Expert rounds | Multi-round (n=2) | Single round |
| Gate | SNR-based adaptive | None (direct addition) |
| Loss | Huber+Dir + aux + base | MSE + aux only |

## Quick Start

All commands run from the `conference/` directory:

```bash
cd conference

# 1. Main experiment: Vanilla vs TSB-Lite (16 datasets × 5 seeds)
uv run python experiments/main_experiment.py

# 2. Cross-backbone validation (4 backbones × 16 datasets × 5 seeds)
uv run python experiments/cross_backbone.py

# 3. SOTA baselines comparison (DLinear, PatchTST, iTransformer)
#    Requires main_experiment results first
uv run python experiments/sota_baselines.py

# 4. Compute cost analysis
uv run python experiments/compute_cost.py
```

### Analyze-only mode (from saved results)

```bash
uv run python experiments/main_experiment.py --analyze
uv run python experiments/cross_backbone.py --analyze
uv run python experiments/sota_baselines.py --analyze
```

## Output Structure

```
results/
├── main/
│   ├── results.json
│   └── figures/
│       ├── rmse_improvement.png
│       ├── mae_improvement.png
│       └── mda_comparison.png
├── cross_backbone/
│   ├── results.json
│   └── figures/
│       └── cross_backbone_heatmap.png
└── sota/
    ├── results.json
    └── figures/
        ├── tsb_vs_best_sota.png
        └── sota_average_rank.png
```

## Dependencies

Uses the same `pyproject.toml` as the parent project. Run with `uv run` from project root.

## Dataset

Datasets are stored in `../dataset/` (shared with thesis version).
