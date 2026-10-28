"""Small, reproducible CPU measurements; no training or provider downloads.

Example:
    python tools/benchmark.py --output benchmark_results/cpu_baseline.json

Model cost follows the paper's compute study (Table 8): parameters and batch-32
forward latency of each backbone with and without TS-Booster and of the deep
comparators, at seq_len=30, pred_len=3, two channels.
The output is created exclusively. Choose a new path for subsequent runs.
Multiworker measurements use spawn and may require permission for local IPC.
"""

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


def summary(values):
    median = statistics.median(values)
    return {"values": values, "median": median, "min": min(values), "max": max(values),
            "median_absolute_deviation": statistics.median(abs(v - median) for v in values)}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MODEL_FILES = {
    "TSBooster": "models/my_models/tsbooster/TSBooster.py",
    "TSBooster-vanilla": "models/my_models/tsbooster/ablation_variant/TSBooster-vanilla.py",
    "MLP7K": "models/my_models/tsbooster/baselines/MLP7K.py",
    "DLinear-paper": "models/my_models/tsbooster/baselines/DLinear-paper.py",
    "PatchTST-paper": "models/my_models/tsbooster/baselines/PatchTST-paper.py",
    "iTransformer-paper": "models/my_models/tsbooster/baselines/iTransformer-paper.py",
}
BACKBONES = ("linear", "lstm", "gru", "transformer")
# NAME:BACKBONE for the TS-Booster family, NAME otherwise. The paper's compute study (Table 8)
# compares each backbone with and without TS-Booster and lists the deep comparators.
DEFAULT_MODELS = [f"{name}:{backbone}" for backbone in BACKBONES for name in ("TSBooster-vanilla", "TSBooster")] + [
    "MLP7K", "DLinear-paper", "PatchTST-paper", "iTransformer-paper"]


def load_model(name):
    spec = importlib.util.spec_from_file_location("benchmark_" + name.replace("-", "_"), ROOT / MODEL_FILES[name])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def model_config(args, backbone):
    return SimpleNamespace(task_name="long_term_forecast", seq_len=args.seq_len, pred_len=args.pred_len,
                           enc_in=args.channels, c_out=1, features="MS", target_channel_index=-1,
                           d_model=args.d_model, e_layers=2, n_heads=4, d_ff=args.d_model * 2, dropout=0.3,
                           tsb_backbone=backbone, tsb_decomp_kernels=[7, 13], tsb_expert_kernels=[3, 5, 7],
                           tsb_expert_hidden=32, tsb_n_experts=2, tsb_expert_type="cnn", tsb_enrich="trend",
                           tsb_aux_weight=0.3, tsb_base_weight=0.1)


def resolve_device(name):
    """torch.device for --device, failing early when the backend is not available."""
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requested but MPS is not available")
    return torch.device(name)


def synchronize(device):
    """Wait for queued kernels so wall-clock timing covers the whole forward pass."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_models(args):
    """Eval-mode forward latency per batch on --device; models alternate order between rounds."""
    device = resolve_device(args.device)
    models = {}
    for entry in args.models:
        name, _, backbone = entry.partition(":")
        if name not in MODEL_FILES or backbone not in ("", *BACKBONES):
            raise ValueError(f"Unknown benchmark model {entry!r}; use NAME or NAME:BACKBONE with NAME in "
                             f"{sorted(MODEL_FILES)}")
        torch.manual_seed(args.seed)
        models[entry] = load_model(name)(model_config(args, backbone or "linear")).eval().to(device)
    x = torch.randn(args.model_batch_size, args.seq_len, args.channels,
                    generator=torch.Generator().manual_seed(args.seed)).to(device)
    times = {entry: [] for entry in models}
    with torch.no_grad():
        for model in models.values():
            for _ in range(args.warmup):
                model(x, None, None, None)
        synchronize(device)
        for index in range(args.rounds):
            order = list(models.items())
            for entry, model in order if index % 2 == 0 else reversed(order):
                synchronize(device)
                started = time.perf_counter()
                for _ in range(args.forwards):
                    model(x, None, None, None)
                synchronize(device)
                times[entry].append((time.perf_counter() - started) * 1000 / args.forwards)
    report = {}
    for entry, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        report[entry] = {"params": params, "trainable_params": sum(p.numel() for p in model.parameters()
                                                                   if p.requires_grad),
                         "batch_forward_ms": summary(times[entry]),
                         "effective_config": getattr(model, "effective_config", None)}
    return {"inputs": f"seeded synthetic tensors on {args.device}",
            "configuration": {"batch": args.model_batch_size, "seq_len": args.seq_len, "pred_len": args.pred_len,
                              "channels": args.channels, "host_d_model": args.d_model, "dtype": "float32",
                              "device": args.device, "seed": args.seed, "rounds": args.rounds,
                              "forwards_per_round": args.forwards, "warmup_per_model": args.warmup},
            "models": report,
            "limits": ["Inference timing only; training cost is recorded per run in result.json",
                       "Synthetic inputs with untrained weights",
                       f"Single {args.device} configuration; do not extrapolate to other devices or shapes",
                       "Alternating order reduces drift; concurrent host load remains possible"]}


def worker_init(_worker_id):
    torch.set_num_threads(1)


def loader_dataset(args):
    started = time.perf_counter()
    if args.loader_source == "csv":
        from data_provider.data_loader import Dataset_Custom
        path = Path(args.csv).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV missing: {path}; provide --csv or use --loader-source memory")
        config = SimpleNamespace(augmentation_ratio=0, allow_remote_data=False, seed=args.seed)
        dataset = Dataset_Custom(config, str(path.parent), data_path=path.name, flag="train",
            size=[args.seq_len, args.seq_len // 2, args.pred_len], features="MS", target=args.target,
            scale=True, timeenc=1, freq="b")
        source = {"kind": "real Dataset_Custom", "csv": str(path.relative_to(ROOT))
                  if path.is_relative_to(ROOT) else str(path), "csv_sha256": fingerprint(path),
                  "windows": len(dataset), "features": dataset.feature_names,
                  "split": "train", "features_mode": "MS", "target": args.target,
                  "freq": "b", "timeenc": 1, "augmentation_ratio": 0}
    else:
        generator = torch.Generator().manual_seed(args.seed)
        dataset = TensorDataset(torch.randn(args.memory_samples, args.seq_len, args.channels, generator=generator),
            torch.randn(args.memory_samples, args.seq_len // 2 + args.pred_len, args.channels, generator=generator))
        source = {"kind": "in-memory TensorDataset fixture", "windows": len(dataset),
                  "channels": args.channels, "seed": args.seed,
                  "limit": "Does not measure Dataset_Custom CSV preparation or NumPy window slicing"}
    source["preparation_seconds"] = time.perf_counter() - started
    return dataset, source


def loader_round(dataset, args, workers):
    options = dict(batch_size=args.loader_batch_size, num_workers=workers, shuffle=False,
                   drop_last=False, pin_memory=False, worker_init_fn=worker_init,
                   generator=torch.Generator().manual_seed(args.seed))
    if workers:
        options.update(multiprocessing_context="spawn", persistent_workers=False,
                       prefetch_factor=2, timeout=60)
    started = time.perf_counter()
    loader = DataLoader(dataset, **options)
    iterator = None
    batches, samples, checksum, first_seconds = 0, 0, 0.0, None
    try:
        iterator = iter(loader)
        for index in range((len(dataset) + args.loader_batch_size - 1) // args.loader_batch_size):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if index == 0:
                first_seconds = time.perf_counter() - started
            batches += 1
            samples += batch[0].shape[0]
            checksum += sum(float(tensor.double().sum()) for tensor in batch)
            if index == 0:
                steady_started = time.perf_counter()
        consumed = time.perf_counter()
        if batches < 2:
            raise ValueError("At least two data batches are required for startup/steady separation")
    finally:
        if iterator is not None:
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
    finished = time.perf_counter()
    return {"first_batch_seconds": first_seconds,
            "remaining_batch_seconds": consumed - steady_started,
            "end_to_end_seconds": finished - started,
            "shutdown_seconds": finished - consumed,
            "remaining_batch_ms": (consumed - steady_started) * 1000 / (batches - 1),
            "batches": batches, "samples": samples, "checksum": checksum}


def benchmark_loader(args):
    dataset, source = loader_dataset(args)
    # Consume every scheduled sample so worker queues drain before teardown.
    # Truncating a live larger iterator can measure queue shutdown timeouts.
    selected = min(len(dataset), args.loader_batches * args.loader_batch_size)
    dataset = Subset(dataset, range(selected))
    source["benchmark_windows"] = selected
    source["selection"] = "first consecutive windows, fully consumed each round"
    results, reference = {}, None
    # Rotate the worker order between rounds to avoid always measuring one mode first.
    for round_index in range(args.rounds):
        shift = round_index % len(args.workers)
        order = args.workers[shift:] + args.workers[:shift]
        for workers in order:
            print(f"DataLoader workers={workers}, round={round_index + 1}/{args.rounds}", flush=True)
            result = loader_round(dataset, args, workers)
            if reference is None:
                reference = (result["samples"], result["batches"], result["checksum"])
            elif (result["samples"], result["batches"], result["checksum"]) != reference:
                raise AssertionError("DataLoader modes produced different data/order")
            results.setdefault(str(workers), []).append(result)
    measurements = {}
    for workers, rounds in results.items():
        measurements[workers] = {"rounds": rounds,
            "summary": {key: summary([row[key] for row in rounds]) for key in
                        ("first_batch_seconds", "remaining_batch_ms", "end_to_end_seconds", "shutdown_seconds")}}
    return {"source": source, "configuration": {"batch_size": args.loader_batch_size,
                "max_batches": args.loader_batches, "rounds": args.rounds, "workers": args.workers,
                "multiprocessing_context": "spawn for workers>0", "persistent_workers": False,
                "prefetch_factor": 2, "shuffle": False, "pin_memory": False,
                "consumer": "sum each batch tensor as a deterministic data checksum"},
            "measurements": measurements, "data_order_parity": True,
            "limits": ["Fresh workers in each round; startup and teardown are explicitly included",
                       "Remaining-batch timing may include prefetch warmup and includes checksum cost",
                       "No model/GPU transfer; results do not determine training concurrency",
                       "CSV preparation is measured once separately; no full CSV-read A/B"]}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=ROOT / "benchmark_results/cpu_baseline.json")
    result.add_argument("--rounds", type=int, default=3)
    result.add_argument("--forwards", type=int, default=40)
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu",
                        help="device of the model forward benchmark (the paper's Table 8 used a GPU)")
    result.add_argument("--seed", type=int, default=5)
    result.add_argument("--seq-len", type=int, default=30)
    result.add_argument("--pred-len", type=int, default=3)
    result.add_argument("--channels", type=int, default=2)
    result.add_argument("--d-model", type=int, default=64, help="host width of the LSTM/GRU/Transformer")
    result.add_argument("--model-batch-size", type=int, default=32)
    result.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="NAME or NAME:BACKBONE entries")
    result.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    result.add_argument("--loader-source", choices=("csv", "memory"), default="memory")
    result.add_argument("--csv", type=Path, default=None, help="CSV for --loader-source csv")
    result.add_argument("--target", default="Volatility", help="MS target column for --loader-source csv")
    result.add_argument("--loader-batch-size", type=int, default=32)
    result.add_argument("--loader-batches", type=int, default=16)
    result.add_argument("--memory-samples", type=int, default=1024)
    result.add_argument("--models-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    positive = ("rounds", "forwards", "threads", "seq_len", "pred_len", "channels", "d_model",
                "model_batch_size", "loader_batch_size", "memory_samples")
    if any(getattr(args, key) <= 0 for key in positive) or args.warmup < 0 or args.seq_len < 2 or args.loader_batches < 2:
        raise ValueError("Benchmark dimensions/counts must be positive, seq_len and loader_batches at least two")
    if args.d_model % 4 or not args.workers or len(set(args.workers)) != len(args.workers) or min(args.workers) < 0:
        raise ValueError("d_model must be divisible by four; workers must be unique nonnegative integers")
    if args.loader_source == "csv" and args.csv is None:
        raise ValueError("--loader-source csv requires --csv")
    if args.output.exists():
        raise FileExistsError(f"Benchmark output already exists; select a new --output path: {args.output}")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    sources = [ROOT / path for path in ("tools/benchmark.py", *MODEL_FILES.values(), "layers/TSBooster_layers.py",
        "layers/TSBooster_baselines.py", "layers/model_contracts.py", "data_provider/data_loader.py",
        "data_provider/preparation.py", "utils/timefeatures.py")]
    before = {str(path.relative_to(ROOT)): fingerprint(path) for path in sources}
    try:
        report = {"schema_version": 1, "measured_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                  "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                      "machine": platform.machine(), "logical_cpus": os.cpu_count(), "device": args.device,
                      "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
                      "torch_threads": args.threads, "torch_interop_threads": torch.get_num_interop_threads()},
                  "source_sha256": before, "command_arguments": vars(args) | {"output": str(args.output), "csv": str(args.csv)}}
        print(f"Model parameters and alternating {args.device} forward latency", flush=True)
        report["models"] = benchmark_models(args)
        if not args.models_only:
            report["dataloader"] = benchmark_loader(args)
        report["source_unchanged_during_measurement"] = all(fingerprint(ROOT / name) == checksum for name, checksum in before.items())
        report["deferred"] = [{"item": "SDPA", "reason": "No GPU kernel/profile evidence; retain existing attention math"},
            {"item": "shared prepared-data cache", "reason": "This measurement has no repeated-CSV preparation baseline or invalidation experiment"},
            {"item": "provider batching/cache", "reason": "No weights downloaded; provider grouping and sampling require separate integration measurements"}]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Saved: {args.output}", flush=True)
        return report
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    main()
