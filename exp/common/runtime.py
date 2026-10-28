"""Device, run identity, checkpoint and artifact contracts."""

from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import uuid

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRIC_ORDER = ("mae", "mse", "rmse", "mape", "mspe", "r2")


def package_versions():
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "einops",
                    "transformers", "chronos-forecasting", "timesfm", "tirex-ts", "uni2ts",
                    "huggingface_hub", "datasets", "sktime", "mamba-ssm", "reformer-pytorch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


@dataclass
class TrainResult:
    model: torch.nn.Module
    train_time_s: float
    best_epoch: int | None = None


def resolve_device(args):
    """Use logical CUDA indices; never change CUDA_VISIBLE_DEVICES mid-process."""
    backend = getattr(args, "gpu_type", "auto")
    use_gpu = getattr(args, "use_gpu", True)
    if not use_gpu or backend == "cpu":
        device = torch.device("cpu")
    elif backend == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", getattr(args, "gpu", 0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif backend == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA 不可用；請使用 --gpu_type auto 或 --no_use_gpu。")
        device = torch.device("cuda", getattr(args, "gpu", 0))
    elif backend == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS 不可用；請使用 --gpu_type auto 或 --no_use_gpu。")
        device = torch.device("mps")
    else:
        raise ValueError(f"不支援的裝置：{backend}")
    if getattr(args, "use_multi_gpu", False):
        if device.type != "cuda":
            raise ValueError("--use_multi_gpu 僅支援 CUDA。")
        try:
            ids = [int(x.strip()) for x in args.devices.split(",")]
        except ValueError as error:
            raise ValueError("--devices 必須是以逗號分隔的 CUDA logical indices。") from error
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("--devices 不可為空或含重複索引。")
        if any(x < 0 or x >= torch.cuda.device_count() for x in ids):
            raise ValueError("--devices 超過目前可見 CUDA 裝置範圍。")
        args.device_ids = ids
        args.gpu = ids[0]
        device = torch.device("cuda", ids[0])
    if device.type == "cuda" and not 0 <= device.index < torch.cuda.device_count():
        raise ValueError("--gpu 超過目前可見 CUDA 裝置範圍。")
    args.device = args.runtime_device = device
    args.use_gpu = device.type != "cpu"
    args.gpu_type = device.type
    return device


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def seed_everything(seed, deterministic=False):
    if deterministic:
        # cuBLAS needs a fixed workspace for deterministic algorithms on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def precision_context(args, device):
    if not getattr(args, "use_amp", False):
        return nullcontext()
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise ValueError(f"目前 PyTorch 不支援 {device.type} autocast。")
    return torch.autocast(device_type=device.type)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_save_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_path(args, setting):
    selected = getattr(args, "checkpoint_run", None) or setting
    return Path(args.checkpoints) / selected / "checkpoint.pth"


def atomic_save_checkpoint(model, args, setting):
    path = Path(args.checkpoints) / setting / "checkpoint.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint.", dir=path.parent)
    os.close(handle)
    try:
        target = model.module if isinstance(model, torch.nn.DataParallel) else model
        torch.save(target.state_dict(), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(model, args, setting, device):
    path = checkpoint_path(args, setting)
    state = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    target.load_state_dict(state)
    return model


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(root=PROJECT_ROOT):
    root = Path(root)
    digest = hashlib.sha256()
    excluded = {".git", ".venv", "__pycache__", ".agents", ".codex", ".claude",
                "dataset", "data", "results", "checkpoints", "test_results", "tests"}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts) or not path.is_file():
            continue
        if path.suffix == ".py" or relative.parts[0] == "requirements":
            digest.update(str(relative).encode())
            digest.update(file_fingerprint(path).encode())
    return digest.hexdigest()


def data_fingerprint(args):
    root = Path(args.root_path)
    candidate = root / getattr(args, "data_path", "")
    if getattr(args, "data", None) in {"custom", "ETTh1", "ETTh2", "ETTm1", "ETTm2"}:
        paths = [candidate] if candidate.is_file() else []
    else:
        paths = sorted(p for p in root.rglob("*") if p.is_file() and
                       p.suffix in {".csv", ".ts", ".npy", ".npz"}) if root.exists() else []
    return {str(path.relative_to(root)): file_fingerprint(path) for path in paths} or {
        "remote_revision": getattr(args, "data_revision", None)}


def canonical_config(args):
    excluded = {"experiment_id", "attempt_id", "group_id", "manifest_path", "setting",
                "batch_key", "batch_attempt_id", "results_root", "test_results_root", "checkpoints",
                "checkpoint_run", "requested_config", "dry_run"}
    return _jsonable({key: value for key, value in vars(args).items() if key not in excluded})


def config_hash(config):
    raw = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def prepare_run(args, model, legacy_name, iteration=0):
    args.iteration = iteration
    config = canonical_config(args)
    config["iteration"] = iteration
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    effective = getattr(target, "effective_config", {})
    versions = package_versions()
    identity = {"config": config, "effective_config": effective,
                "source_hash": source_fingerprint(), "data": data_fingerprint(args),
                "versions": versions}
    checkpoint_source = None
    if not getattr(args, "is_training", True) and args.task_name != "zero_shot_forecast":
        selected = checkpoint_path(args, legacy_name).resolve()
        checkpoint_source = {"path": str(selected), "sha256": file_fingerprint(selected)}
        identity["checkpoint_source"] = checkpoint_source
    experiment_id = config_hash(identity)
    group = dict(identity)
    group["config"] = {k: v for k, v in config.items() if k not in {"rand_seed", "seed", "iteration"}}
    args.experiment_id, args.group_id = experiment_id, config_hash(group)
    args.attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    setting = f"{legacy_name}__cfg{experiment_id[:16]}__attempt{args.attempt_id}"
    args.setting = setting
    directory = Path(getattr(args, "results_root", "./results")) / setting
    args.manifest_path = str(directory / "metadata.json")
    record = {"schema_version": 1, "evaluation_version": 2, "status": "running",
              "experiment_id": experiment_id, "group_id": args.group_id,
              "attempt_id": args.attempt_id, "setting": setting, "legacy_setting": legacy_name,
              "config": config, "effective_config": effective, "source_hash": identity["source_hash"],
              "requested_config": getattr(args, "requested_config", config),
              "checkpoint_source": checkpoint_source,
              "data": identity["data"], "versions": versions,
              "python": sys.version, "device": str(args.device), "batch_key": getattr(args, "batch_key", None),
              "batch_attempt_id": getattr(args, "batch_attempt_id", None),
              "started_at": datetime.now(timezone.utc).isoformat()}
    atomic_write_json(directory / "config.json", config)
    atomic_write_json(directory / "requested_config.json", record["requested_config"])
    atomic_write_json(directory / "effective_config.json", effective)
    atomic_write_json(args.manifest_path, record)
    return setting


def mark_run_failed(args, error):
    if not getattr(args, "manifest_path", None):
        return
    path = Path(args.manifest_path)
    record = json.loads(path.read_text())
    record.update(status="failed", error_type=type(error).__name__, error=str(error))
    atomic_write_json(path, record)


def save_evaluation(args, setting, metrics, predictions, targets, extra=None, summary=None):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or predictions.size == 0:
        raise ValueError("Prediction/target shapes must match and contain samples.")
    # Imputation also emits dense arrays. Its mask limits scoring, rather than
    # granting an exception for NaN/Inf in observed prediction positions.
    for name, values in (("Prediction", predictions), ("Target", targets)):
        if values.dtype.kind not in "biuf":
            raise ValueError(f"{name} must be a real numeric array.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite, including positions outside an imputation mask.")
    details = dict(extra or {})
    mask = details.pop("mask", None)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != predictions.shape or mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all():
            raise ValueError("Imputation mask must be binary, finite, and match prediction/target shape.")
        if getattr(args, "task_name", None) == "imputation" and not np.any(mask == 0):
            raise ValueError("Imputation evaluation requires at least one masked target.")
    directory = Path(getattr(args, "results_root", "./results")) / setting
    directory.mkdir(parents=True, exist_ok=True)
    atomic_save_array(directory / "pred.npy", np.asarray(predictions))
    atomic_save_array(directory / "true.npy", np.asarray(targets))
    lowered = {str(key).lower(): value for key, value in metrics.items()}
    if all(key in lowered for key in METRIC_ORDER):
        legacy = np.asarray([lowered[key] for key in METRIC_ORDER])
    else:
        legacy = np.asarray([value for value in lowered.values() if isinstance(value, (float, int, np.number))])
    atomic_save_array(directory / "metrics.npy", legacy)
    additional = details.pop("additional_artifacts", [])
    if details.get("forecast_csv"):
        additional = list(additional) + [Path(details["forecast_csv"]).name]
    for name in additional:
        if not (directory / name).is_file():
            raise ValueError(f"Required extra artifact missing: {name}")
    if mask is not None:
        atomic_save_array(directory / "mask.npy", np.asarray(mask))
    record = {"schema_version": 1, "evaluation_version": 2, "status": "complete",
              "experiment_id": getattr(args, "experiment_id", setting),
              "group_id": getattr(args, "group_id", setting),
              "attempt_id": getattr(args, "attempt_id", setting), "setting": setting,
              "config": canonical_config(args), "metrics": metrics, "summary": summary,
              "metric_states": {key: "nonfinite" if isinstance(value, (float, np.floating)) and
                                 not math.isfinite(value) else "ok" for key, value in metrics.items()},
              "details": details,
              "artifacts": ["pred.npy", "true.npy", "metrics.npy"] + (["mask.npy"] if mask is not None else []) + list(additional)}
    atomic_write_json(directory / "result.json", record)
    if getattr(args, "manifest_path", None):
        path = Path(args.manifest_path)
        metadata = json.loads(path.read_text())
        metadata.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                        artifacts=record["artifacts"] + ["result.json"])
        atomic_write_json(path, metadata)
    return record
