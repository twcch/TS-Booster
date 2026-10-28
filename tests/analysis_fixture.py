"""Synthetic campaign for the analysis tests.

write_campaign() replays the experiment definitions of run_batch_long_term_forecast_01 ... _08
on short synthetic series: every distinct configuration becomes one result directory in the
layout exp/common/runtime writes (result.json, metadata.json, pred/true/last/gate .npy), with
noise levels that make TS-Booster the best neural method, so the analysis modules run end to
end on exactly the configurations the campaign produced.
"""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.batch_runner import expand_models
from tools.data.build_volatility_dataset import volatility_frame

PLANS = [f"run_batch_long_term_forecast_0{number}" for number in range(1, 10)]
# run.py defaults for every key analysis.common.load_runs reads; the plan files override most of them.
BASE = {"loss": "MSE", "train_mse_weight": 0.0, "dir_weight": 0.5, "pred_offset": 0, "learning_rate": 1e-4,
        "train_epochs": 10, "enc_in": 7, "root_path": "./dataset/volatility/", "tsb_backbone": "linear",
        "tsb_aux_weight": 0.3, "tsb_base_weight": 0.1, "tsb_enrich": "trend", "tsb_decomp_fixed": False,
        "tsb_n_experts": 2, "tsb_expert_hidden": 32, "tsb_decomp_kernels": [7, 13], "tsb_expert_kernels": [3, 5, 7]}
PARAMS = {("TSBooster-vanilla", "linear"): 366, ("TSBooster", "linear"): 7156,
          ("TSBooster-vanilla", "lstm"): 50883, ("TSBooster", "lstm"): 57673,
          ("TSBooster-vanilla", "gru"): 38211, ("TSBooster", "gru"): 45001,
          ("TSBooster-vanilla", "transformer"): 67331, ("TSBooster", "transformer"): 74121,
          ("TSBooster-woexpert", "linear"): 734, ("TSBooster-wodecomp", "linear"): 6552,
          ("DLinear-paper", None): 372, ("PatchTST-paper", None): 402823, ("iTransformer-paper", None): 269571,
          ("DLinear", None): 372, ("PatchTST", None): 400899, ("iTransformer", None): 269571, ("MLP7K", None): 7171}
# Relative error scale per model: the full module beats its ablations, the aligned econometric models
# and the observed-window rule come close, the single-day conditional-volatility models are far off.
NOISE = {"TSBooster": 0.60, "TSBooster-constgate": 0.60, "TSBooster-learnedgate": 0.61, "TSBooster-nogate": 0.62,
         "TSBooster-wodecomp": 0.65, "TSBooster-woexpert": 0.98, "TSBooster-vanilla": 1.00, "MLP7K": 1.02,
         "DLinear-paper": 0.85, "PatchTST-paper": 0.90, "iTransformer-paper": 0.88,
         "DLinear": 0.86, "PatchTST": 0.90, "iTransformer": 0.87,
         "HAR-RV": 0.90, "ARIMA": 1.10, "XGBoost": 0.95, "LightGBM": 0.93,
         "EWMA": 1.80, "GARCH": 1.85, "GJR-GARCH": 1.85,
         "EWMA-aligned": 0.72, "GARCH-aligned": 0.70, "GJR-GARCH-aligned": 0.69, "ObservedWindowRule": 0.75}
LR_FACTOR = {1e-5: 2.0, 5e-5: 1.6, 1e-4: 1.4, 5e-4: 1.2, 1e-3: 1.0, 3e-3: 1.1}
GATES = {"TSBooster": "snr", "TSBooster-constgate": "const", "TSBooster-learnedgate": "learned"}
DECOMPOSED = {"TSBooster", "TSBooster-woexpert", "TSBooster-constgate", "TSBooster-learnedgate", "TSBooster-nogate"}
CLASSICAL = {"Persistence", "EWMA", "HAR-RV", "GARCH", "GJR-GARCH", "ARIMA", "XGBoost", "LightGBM",
             "EWMA-aligned", "GARCH-aligned", "GJR-GARCH-aligned", "ObservedWindowRule"}


def _rng(*parts):
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return np.random.default_rng(int(digest[:12], 16))


def write_prices(root, dataset, n_prices=407):
    """Geometric random walk -> dataset/raw/<ds>.csv and dataset/volatility/<ds>.csv; returns the volatility frame."""
    rng = _rng("prices", dataset)
    dates = pd.bdate_range("2010-01-04", periods=n_prices)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, n_prices)))
    prices = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": close})
    raw = Path(root) / "dataset" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw / f"{dataset}.csv", index=False)
    volatility = volatility_frame(prices)
    folder = Path(root) / "dataset" / "volatility"
    folder.mkdir(parents=True, exist_ok=True)
    volatility.to_csv(folder / f"{dataset}.csv", index=False)
    return volatility


def configurations(datasets, seeds, plans=PLANS):
    """Every distinct run configuration of the campaign plans, on the given datasets and seeds."""
    seen, configs = set(), []
    for name in plans:
        plan = importlib.import_module(name)
        for experiment in plan.EXPERIMENTS.values():
            windows = experiment.get("windows") or [[length, length // 2] for length in experiment["seq_lens"]]
            run_seeds = [0] if experiment.get("deterministic") else list(seeds)
            for model in expand_models(experiment["models"]):
                overrides = model if isinstance(model, dict) else {"model": model}
                for dataset in datasets:
                    for window in windows:
                        for horizon in experiment["pred_lens"]:
                            for seed in run_seeds:
                                config = {**BASE, **experiment["shared"], **overrides,
                                          "task_name": "long_term_forecast", "data": "custom",
                                          "data_path": f"{dataset}.csv", "seq_len": window[0],
                                          "label_len": window[1], "pred_len": horizon, "rand_seed": seed}
                                key = json.dumps(config, sort_keys=True)
                                if key not in seen:
                                    seen.add(key)
                                    configs.append(config)
    return configs


def example_config(dataset="AA_TWD", seed=42, **overrides):
    """The paper's TS-Booster+Linear configuration on one dataset and seed."""
    protocol = importlib.import_module(PLANS[0]).PROTOCOL
    return {**BASE, **protocol, "model": "TSBooster", "tsb_backbone": "linear", "task_name": "long_term_forecast",
            "data": "custom", "data_path": f"{dataset}.csv", "seq_len": 30, "label_len": 0, "pred_len": 3,
            "rand_seed": seed, **overrides}


def _metrics(pred, true, last):
    error = pred - true
    return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error))),
            "mda": float(np.mean(np.sign(pred - last[:, None, :]) == np.sign(true - last[:, None, :]))),
            "mse": float(np.mean(error ** 2)), "mape": float(np.mean(np.abs(error / true))),
            "mspe": float(np.mean((error / true) ** 2)),
            "r2": float(1 - np.sum(error ** 2) / np.sum((true - true.mean()) ** 2))}


def write_run(results_root, config, volatility, *, attempt="a", finished_at=None, scale=None,
              metric_space="original"):
    """One result directory for `config`; returns its path."""
    model, dataset, seed = config["model"], Path(config["data_path"]).stem, int(config["rand_seed"])
    horizon, offset = int(config["pred_len"]), int(config.get("pred_offset", 0))
    vol = volatility["Volatility"].to_numpy()
    n = len(vol)
    count = int(0.2 * n) - offset - horizon + 1
    rows = n - int(0.2 * n) - 1 + np.arange(count)
    last = vol[rows][:, None]
    true = np.stack([vol[rows + offset + h + 1] for h in range(horizon)], axis=1)[:, :, None]
    key = json.dumps(config, sort_keys=True)
    rng = _rng("run", key, attempt)
    if scale is None:
        scale = NOISE.get(model, 0.9) * LR_FACTOR.get(float(config.get("learning_rate", 1e-3)), 1.0)
    if model == "Persistence":
        pred = np.repeat(last[:, None, :], horizon, axis=1)
    else:
        pred = true + rng.normal(size=true.shape) * scale * 0.25 * vol.std()
    metrics = _metrics(pred, true, last)
    backbone = config.get("tsb_backbone") if model.startswith("TSBooster") else None
    params = PARAMS.get((model, backbone), 1000)
    details = {"metric_space": metric_space, "metrics_original": metrics,
               "metrics_normalized": {name: value / 10 for name, value in metrics.items()},
               "metric_report_original": {"per_horizon": [_metrics(pred[:, h:h + 1], true[:, h:h + 1], last)
                                                          for h in range(horizon)]},
               "params": params, "trainable_params": params, "epochs_trained": 40 + seed % 7,
               "best_epoch": 10 + seed % 7, "best_selection_metric": metrics["rmse"] ** 2}
    gate = None
    if model.startswith("TSBooster"):
        diagnostics = {"gate": GATES.get(model, "none")}
        if model in DECOMPOSED:
            w7 = 0.5 if config.get("tsb_decomp_fixed") else 0.45 + 0.2 * rng.random()
            diagnostics["decomp_mixture"] = {"7": w7, "13": 1 - w7}
        if model == "TSBooster":
            w, b = 0.3 + 0.4 * rng.random(), -0.5
            gate = 1 / (1 + np.exp(-(w * rng.uniform(0.5, 1.0, count) + b)))
            diagnostics.update(snr_gate_w=w, snr_gate_b=b)
        elif model == "TSBooster-constgate":
            gate = np.full(count, 0.5)
            diagnostics["const_gate"] = 0.5
        elif model == "TSBooster-learnedgate":
            gate = np.full(count, 0.56)
            diagnostics["learned_gate"] = 0.56
        if gate is not None:
            diagnostics.update(gate_mean=float(gate.mean()), gate_std=float(gate.std()),
                               gate_min=float(gate.min()), gate_max=float(gate.max()))
        details["model_diagnostics"] = diagnostics
    elif model in CLASSICAL:
        baseline = {"method": model}
        if model == "ARIMA":
            random_walk = _rng("arima", dataset).random() < 0.5
            baseline["selected"] = {"order": [0, 1, 0] if random_walk else [1, 1, 0],
                                    "with_intercept": not random_walk, "aic": 1.0, "params": []}
        elif model == "HAR-RV":
            baseline["coefficients"] = {str(h + 1): [0.2, 1.0, -0.2, 0.1] for h in range(horizon)}
        details["baseline"] = baseline
    name = (f"long_term_forecast_{dataset}_{config['seq_len']}_{config['label_len']}_{horizon}_{model}"
            f"_seed{seed}_{hashlib.sha256(key.encode()).hexdigest()[:8]}_{attempt}")
    run = Path(results_root) / name
    run.mkdir(parents=True, exist_ok=True)
    ids = {"experiment_id": name[:-2], "group_id": "group-" + name[:-2], "attempt_id": name}
    (run / "result.json").write_text(json.dumps({"status": "complete", **ids, "config": config,
                                                 "metrics": metrics, "details": details}))
    (run / "metadata.json").write_text(json.dumps({
        "status": "complete", **ids, "source_hash": "fixture",
        "finished_at": finished_at or f"2026-10-23T00:00:{seed % 60:02d}+00:00"}))
    np.save(run / "pred.npy", pred)
    np.save(run / "true.npy", true)
    np.save(run / "last.npy", last)
    if gate is not None:
        np.save(run / "gate.npy", gate.astype(np.float32))
    return run


def write_campaign(root, datasets=("AA_TWD", "BB_TWD"), seeds=(42, 123, 456), n_prices=407):
    """Write the whole synthetic campaign under root; returns {"runs": ..., "configs": ...}."""
    root = Path(root)
    volatility = {dataset: write_prices(root, dataset, n_prices) for dataset in datasets}
    configs = configurations(datasets, seeds)
    for config in configs:
        write_run(root / "results", config, volatility[Path(config["data_path"]).stem])
    distinct = {json.dumps({k: v for k, v in config.items() if k not in ("data_path", "rand_seed")}, sort_keys=True)
                for config in configs}
    return {"runs": len(configs), "configs": len(distinct)}
