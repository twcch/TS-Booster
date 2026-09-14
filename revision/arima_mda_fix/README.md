# ARIMA MDA off-by-one fix (2026-08-03)

`benchmark_baselines.py` `run_arima()` used `last_obs = test_vol[i]` as the MDA
direction anchor — but `test_vol[i]` is the h=1 forecast target itself, so the
h=1 direction was always sign(0) and mechanically wrong, deflating ARIMA MDA to
~1/3. The correct anchor (the last observation before the forecast window) is
`vol[train_val_end + i - 1]`. RMSE/MAE were unaffected.

`recompute_parallel.py` re-runs ARIMA on the snapshot CSVs from commit 1a65142
(the tree state when `benchmark_baselines_results.json` was finalized) and
computes MDA with both anchors. Validation: the buggy MDA and RMSE must match
the stored JSON.

Results (`arima_mda_recompute.json`):
- 16-dataset Avg MDA: buggy 0.3568 (stored table: 0.3563), fixed **0.4727**
- 12/16 datasets reproduce the stored RMSE at 4-dp table precision exactly;
  deviations from optimizer convergence on this machine: EUR_TWD (2.0806 vs
  2.0898), 2454_TW (9.0533 vs 8.7451), 2317_TW (7.9323 vs 7.9330) — for these,
  the fixed MDA carries a small (~±0.01 per dataset) uncertainty.
- CAD_TWD auto_arima selects (0,1,0) without intercept: predictions are
  point-identical to persistence (like 0050_TW and 2603_TW).
- Caveat for the three degenerate (0,1,0) datasets (0050_TW, 2603_TW, CAD_TWD):
  their mda_fixed values (0.0355 / 0.0496 / 0.0431) are driven by ~1e-6-level
  floating-point residues between the pmdarima forecasts and the anchor, whose
  sign() is machine-unstable. A persistence-consistent treatment (pred_delta
  exactly 0) would give a 16-dataset average of ~0.4648 instead of 0.4727.
  Either convention is defensible; the paper uses the pipeline value 0.4727,
  so the overall uncertainty is a few units in the third decimal.

The snapshot `dataset/` CSVs are not stored here; restore them with
`git show 1a65142:dataset/<NAME>.csv` before re-running `recompute_parallel.py`.
For an authoritative regeneration, re-run `benchmark_baselines.py` (now fixed)
on the original experiment machine against the 2026-02/03 dataset snapshot.
