"""Shared batch sweeps with version-aware resume and verified artifacts."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import itertools
import json
import os
from pathlib import Path
import shlex
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np

from exp.common.runtime import (PROJECT_ROOT, atomic_write_json, config_hash,
                                data_fingerprint, file_fingerprint, package_versions, source_fingerprint)


def available_cpus():
    """CPUs this process may use: its affinity, capped by a cgroup CPU quota.

    Containers (e.g. Vast.ai) report every host CPU in os.cpu_count() while the
    quota allows only a share of them.
    """
    try:
        count = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS, Windows
        count = os.cpu_count() or 1
    try:  # cgroup v2: "<quota|max> <period>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        limit = None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        try:  # cgroup v1: quota -1 means unlimited
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            limit = quota / int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text()) if quota > 0 else None
        except (OSError, ValueError):
            limit = None
    return max(1, min(count, int(limit)) if limit else count)


RUNS_PER_GPU = 32  # each run holds a CUDA context of about 0.5 GB


def default_parallel(gpus=None):
    """One run per available CPU, at most RUNS_PER_GPU per GPU.

    Short TS-Booster runs at one thread each, on RTX 4090 instances:
    - 15.36-CPU quota, 32 runs: 14.9 / 24.8 / 28.7 / 32.8 / 28.6 runs per minute with
      4 / 8 / 12 / 16 / 20 parallel runs; past the quota, throttling costs more than
      the extra runs gain, and two threads per run did not help;
    - 35-CPU quota, 64 runs: 35.8 / 38.7 / 40.4 with 16 / 24 / 32 (16 runs with two
      threads: 37.4); the GPU stays near saturation and 32 runs use 16 GB of its memory.
    """
    return max(1, min(available_cpus(), RUNS_PER_GPU * max(1, len(gpus or []))))


def _parser_for(script):
    """run.py's parser, or the build_parser() of another entrypoint (e.g. the classical baselines)."""
    if script == "run.py":
        from run import build_parser
    else:
        build_parser = importlib.import_module(Path(script).with_suffix("").as_posix().replace("/", ".")).build_parser
    return build_parser()


def expand_models(models):
    """Model entries: a name, a dict of options, or a dict whose "grid" maps options to
    lists of values; a grid expands into every combination (several hyperparameter
    sets for one model). Plain list values stay single values (e.g. kernel sets)."""
    expanded = []
    for entry in models:
        if isinstance(entry, str) or "grid" not in entry:
            expanded.append(entry)
            continue
        entry = dict(entry)
        grid = entry.pop("grid")
        if not isinstance(grid, dict) or any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError(f"grid of {entry.get('model')!r} must map options to nonempty lists of values")
        keys = list(grid)
        for values in itertools.product(*(grid[key] for key in keys)):
            expanded.append({**entry, **dict(zip(keys, values))})
    return expanded


def build_commands(config, python=sys.executable):
    """Pair context/label windows rather than taking their Cartesian product.

    config["script"] selects the entrypoint (default run.py); options are checked
    against that entrypoint's parser. Model grids are expanded (expand_models).
    """
    shared = dict(config["shared"])
    script = config.get("script", "run.py")
    actions = _parser_for(script)._actions
    option_actions = {action.option_strings[0][2:]: action for action in actions if action.option_strings}
    windows = config.get("windows")
    if windows is None:
        windows = [[length, length // 2] for length in config["seq_lens"]]
    commands = []
    for data, window, horizon, seed, model in itertools.product(
            config["data_names"], windows, config["pred_lens"], config["rand_seeds"],
            expand_models(config["models"])):
        if len(window) != 2 or not 0 <= window[1] <= window[0]:
            raise ValueError("Each window must be [seq_len, label_len] with 0 <= label_len <= seq_len.")
        options = dict(shared)
        options.update(data_path=f"{data}.csv", seq_len=window[0], label_len=window[1],
                       pred_len=horizon, rand_seed=seed,
                       model_id=f"{data}_{window[0]}_{window[1]}_{horizon}")
        options.update({"model": model} if isinstance(model, str) else model)
        argv = [str(python), "-u", str(PROJECT_ROOT / script)]
        for name, value in sorted(options.items()):
            if name not in option_actions:
                raise ValueError(f"Unknown run.py option: {name}")
            action = option_actions[name]
            if isinstance(value, bool):
                candidates = [item for item in actions if item.dest == action.dest
                              and isinstance(item, (argparse._StoreTrueAction, argparse._StoreFalseAction))
                              and item.const is value]
                if candidates:
                    argv.append(candidates[0].option_strings[0])
                elif action.default != value:
                    raise ValueError(f"No CLI flag can encode {name}={value}")
            elif value is not None:
                argv.append(f"--{name}")
                argv.extend(map(str, value if isinstance(value, list) else [value]))
        commands.append(argv)
    return commands


def _options(argv):
    options = {}
    key = None
    for value in argv:
        if value.startswith("--"):
            key = value[2:]
            options[key] = True
        elif key is not None:
            previous = options[key]
            options[key] = value if previous is True else ([previous, value] if isinstance(previous, str) else previous + [value])
    return options


def command_key(argv, source_hash=None, versions=None):
    options = _options(argv)
    root = Path(options.get("root_path", "./data/ETT/"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    data_args = SimpleNamespace(root_path=root, data=options.get("data", "ETTh1"),
                                data_path=options.get("data_path", "ETTh1.csv"),
                                data_revision=options.get("data_revision"))
    executable = Path(shutil.which(argv[0]) or argv[0]).absolute()
    # The script is the first non-flag argument after the interpreter (e.g. after -u).
    entrypoint = next((value for value in argv[1:] if not value.startswith("-")), None)
    if entrypoint is not None and Path(entrypoint).is_relative_to(PROJECT_ROOT):
        entrypoint = str(Path(entrypoint).relative_to(PROJECT_ROOT))
    identity = {"executable": str(executable), "entrypoint": entrypoint, "options": options,
                "source_hash": source_hash or source_fingerprint(), "versions": versions or {},
                "data_hash": data_fingerprint(data_args)}
    if options.get("checkpoint_run"):
        weights = Path(options.get("checkpoints", "./checkpoints/"))
        if not weights.is_absolute():
            weights = PROJECT_ROOT / weights
        weights = weights / str(options["checkpoint_run"]) / "checkpoint.pth"
        identity["checkpoint_hash"] = file_fingerprint(weights) if weights.is_file() else "missing"
    return config_hash(identity)


def executable_versions(executable):
    """Fingerprint the child's environment, including distinct venv symlinks."""
    path = Path(shutil.which(executable) or executable).absolute()
    versions = package_versions()
    if path == Path(sys.executable).absolute():
        return versions
    script = ('import importlib.metadata as m,json,sys; '
              'installed={d.metadata["Name"].lower().replace("_","-"):d.version for d in m.distributions()}; '
              'print(json.dumps({p:(sys.version.split()[0] if p=="python" else '
              'installed.get(p.lower().replace("_","-"))) for p in json.loads(sys.argv[1])}))')
    result = subprocess.run([str(path), '-c', script, json.dumps(list(versions))],
                            check=True, capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT)
    return json.loads(result.stdout)


@contextmanager
def manifest_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ".lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as error:
        stream.close()
        raise RuntimeError(f"另一個 runner 正使用 {directory}；請使用不同 --run_dir。") from error
    try:
        yield
    finally:
        if os.name == "nt":
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def verify_artifacts(directory):
    directory = Path(directory)
    try:
        metadata = json.loads((directory / "metadata.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        if metadata["status"] != "complete" or result["status"] != "complete":
            return False
        if any(not result.get(key) or metadata.get(key) != result.get(key)
               for key in ("experiment_id", "attempt_id", "group_id", "setting")):
            return False
        required = set(result["artifacts"]) | {"pred.npy", "true.npy", "metrics.npy"}
        for filename in required:
            path = directory / filename
            if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
                return False
            if path.suffix == ".npy":
                np.load(path, mmap_mode="r", allow_pickle=False)
        prediction = np.load(directory / "pred.npy", mmap_mode="r", allow_pickle=False)
        target = np.load(directory / "true.npy", mmap_mode="r", allow_pickle=False)
        return (prediction.shape == target.shape and prediction.size > 0 and
                np.isfinite(prediction).all() and np.isfinite(target).all())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_commands(commands, run_dir, max_parallel=1, dry_run=False, timeout=None, *, threads_per_job=None,
                 gpus=None):
    """Run commands as up to max_parallel concurrent child processes.

    threads_per_job caps each child's CPU threads (OMP/MKL) so parallel runs do not
    oversubscribe the cores; gpus lists CUDA devices, and each run starts on the one
    with the fewest active runs (via CUDA_VISIBLE_DEVICES, outside the run identity).
    """
    if max_parallel <= 0:
        raise ValueError("max_parallel 必須大於零。")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout 必須大於零。")
    if threads_per_job is not None and threads_per_job <= 0:
        raise ValueError("threads_per_job 必須大於零。")
    gpus = [str(gpu) for gpu in gpus or []]
    if dry_run:
        print(json.dumps({"experiments": len(commands), "max_parallel": max_parallel,
                          "threads_per_job": threads_per_job, "gpus": gpus or None}))
        for command in commands:
            print(shlex.join(command))
        return 0
    environments = {executable: executable_versions(executable) for executable in {argv[0] for argv in commands}}
    code = source_fingerprint()
    directory = Path(run_dir)
    guard, processes, stop = threading.Lock(), {}, threading.Event()
    gpu_load = {gpu: 0 for gpu in gpus}
    base_environment = os.environ.copy()
    if threads_per_job:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            base_environment[name] = str(threads_per_job)
    with manifest_lock(directory):
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text()) if path.exists() else {"schema_version": 1, "experiments": {}}
        entries = manifest["experiments"]
        pending, skipped = [], 0
        seen = set()
        for argv in commands:
            key = command_key(argv, code, environments[argv[0]])
            if key in seen:
                raise ValueError("Sweep contains duplicate commands; use --itr for repeated experiments.")
            seen.add(key)
            previous = entries.get(key, {})
            artifacts = previous.get("result_directories", [])
            if previous.get("status") == "complete" and artifacts and all(verify_artifacts(p) for p in artifacts):
                skipped += 1
            else:
                pending.append((key, argv))

        def update(key, **fields):
            with guard:
                entries.setdefault(key, {}).update(fields)
                atomic_write_json(path, manifest)

        def run_one(key, argv):
            if stop.is_set():
                return 1
            attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
            log = directory / "logs" / key / f"{attempt}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            options = _options(argv)
            started, started_wall = time.perf_counter(), time.time()
            child = list(argv) + ["--batch_key", key, "--batch_attempt_id", attempt]
            environment, device = base_environment, None
            update(key, status="running", command=argv, attempt=attempt, log=str(log),
                   attempts=entries.get(key, {}).get("attempts", 0) + 1)
            try:
                with log.open("w", encoding="utf-8") as output:
                    output.write(shlex.join(child) + "\n")
                    output.flush()
                    with guard:
                        if stop.is_set():
                            update_needed = True
                        else:
                            update_needed = False
                            if gpu_load:
                                device = min(gpu_load, key=gpu_load.get)
                                gpu_load[device] += 1
                                environment = dict(base_environment, CUDA_VISIBLE_DEVICES=device)
                            process = subprocess.Popen(child, cwd=PROJECT_ROOT, stdout=output,
                                                       stderr=subprocess.STDOUT, start_new_session=os.name != "nt",
                                                       env=environment)
                            processes[key] = process
                    if update_needed:
                        update(key, status="cancelled")
                        return 1
                    try:
                        returncode = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        _stop_process(process)
                        returncode = 124
                    finally:
                        with guard:
                            processes.pop(key, None)
                            if device is not None:
                                gpu_load[device] -= 1
                results = []
                root = Path(str(options.get("results_root", "./results")))
                if not root.is_absolute():
                    root = PROJECT_ROOT / root
                for record_path in root.glob("*/metadata.json"):
                    try:
                        # Only folders written during this attempt can belong to it.
                        if record_path.stat().st_mtime < started_wall - 5:
                            continue
                        record = json.loads(record_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    if record.get("batch_key") == key and record.get("batch_attempt_id") == attempt:
                        results.append(str(record_path.parent.resolve()))
                expected = int(options.get("itr", 1)) if int(options.get("is_training", 1)) else 1
                complete = returncode == 0 and len(results) == expected and all(verify_artifacts(p) for p in results)
                update(key, status="complete" if complete else "failed", returncode=returncode,
                       elapsed_s=time.perf_counter() - started, result_directories=results,
                       error=None if complete else "子程序失敗或必要產物未完成驗證。")
                return 0 if complete else 1
            except Exception as error:
                update(key, status="failed", error=f"{type(error).__name__}: {error}")
                return 1

        failures, finished, began = 0, 0, time.perf_counter()
        print(f"{len(pending)} runs to do ({skipped} already complete), {max_parallel} at a time, "
              f"CPU threads per run: {threads_per_job or 'default'}; logs in {directory / 'logs'}", flush=True)
        executor = ThreadPoolExecutor(max_workers=max_parallel)
        futures = {executor.submit(run_one, key, argv): (key, argv) for key, argv in pending}
        try:
            for future in as_completed(futures):
                failures += future.result() != 0
                finished += 1
                key, argv = futures[future]
                with guard:
                    entry, running = dict(entries.get(key, {})), len(processes)
                print(_progress_line(finished, len(pending), entry, argv, running, failures,
                                     time.perf_counter() - began), flush=True)
        except KeyboardInterrupt:
            stop.set()
            with guard:
                active = list(processes.values())
            for process in active:
                _stop_process(process)
            failures += 1
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        print(json.dumps({"total": len(commands), "executed": len(pending),
                          "skipped": skipped, "failed": failures}, ensure_ascii=False))
        return 1 if failures else 0


def _duration(seconds):
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def _describe(argv):
    """model[:backbone] data pl<H>[ po<offset>] seed <s>, from a run's argv."""
    options = _options(argv)
    model = str(options.get("model", "?"))
    if model.startswith("TSBooster") and options.get("tsb_backbone"):
        model += f":{options['tsb_backbone']}"
    parts = [model, Path(str(options.get("data_path", options.get("data", "?")))).stem,
             f"pl{options.get('pred_len', '?')}"]
    if options.get("pred_offset") not in (None, "0"):
        parts.append(f"po{options['pred_offset']}")
    return " ".join(parts + [f"seed {options.get('rand_seed', '?')}"])


def _progress_line(finished, total, entry, argv, running, failures, elapsed):
    """One line per finished run: count, status, run, its time, and the sweep's ETA."""
    status = entry.get("status", "?")
    width = len(str(total))
    line = (f"[{finished:>{width}}/{total}] {status:<8} {_describe(argv)} ({_duration(entry.get('elapsed_s', 0))})"
            f" | running {running} | failed {failures} | elapsed {_duration(elapsed)}")
    if finished < total:
        line += f" | ETA {_duration(elapsed / finished * (total - finished))}"
    if status != "complete" and entry.get("log"):
        line += f" | log {entry['log']}"
    return line


def _stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.terminate()
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _parse_value(text):
    """--set values: JSON when possible (2, 0.5, true, [5, 11]), otherwise the raw string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def count_runs(config):
    windows = config.get("windows") or config.get("seq_lens") or []
    return (len(config["data_names"]) * len(windows) * len(config["pred_lens"])
            * len(config["rand_seeds"]) * len(expand_models(config["models"])))


def apply_selection(config, args):
    """Narrow a sweep (datasets, seeds, models) and apply CLI overrides, on a copy.

    A sweep with "deterministic": true runs its methods once; --seeds leaves it alone.
    """
    config = json.loads(json.dumps(config))
    if getattr(args, "datasets", None):
        config["data_names"] = [name for name in config["data_names"] if name in args.datasets]
    if getattr(args, "seeds", None) and not config.get("deterministic"):
        config["rand_seeds"] = [seed for seed in config["rand_seeds"] if seed in args.seeds]
    if getattr(args, "models", None):
        config["models"] = [entry for entry in config["models"]
                            if (entry if isinstance(entry, str) else entry["model"]) in args.models]
    for item in getattr(args, "set", None) or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        value = _parse_value(value)
        config["shared"][key] = value
        for entry in config["models"]:  # an explicit override also wins over per-model values and grids
            if isinstance(entry, dict):
                if key in entry:
                    entry[key] = value
                entry.get("grid", {}).pop(key, None)
                if entry.get("grid") == {}:
                    entry.pop("grid")
    for key in ("num_workers", "gpu_type", "use_dtw", "inverse", "no_plots"):
        value = getattr(args, key)
        if value is not None:
            config["shared"][key] = value
    config["shared"].setdefault("num_workers", 0)
    config["shared"].setdefault("gpu_type", "auto")
    return config


def result_directories(commands, run_dir):
    """Result folders of completed commands, looked up in the run manifest."""
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return []
    entries = json.loads(path.read_text())["experiments"]
    environments = {executable: executable_versions(executable) for executable in {argv[0] for argv in commands}}
    code, directories = source_fingerprint(), []
    for argv in commands:
        entry = entries.get(command_key(argv, code, environments[argv[0]]), {})
        if entry.get("status") == "complete":
            directories.extend(entry.get("result_directories", []))
    return directories


def write_report(directories, output, scale="primary"):
    """Excel (raw + summary sheets) or CSV (plus <name>_summary.csv) for the given runs."""
    import pandas as pd
    from tools.parse_result_to_xlsx import COLUMNS, parse_results, summarize
    frames = [parse_results(Path(directory), scale) for directory in directories]
    frames = [frame.dropna(axis=1, how="all") for frame in frames if not frame.empty]
    if not frames:
        print("No complete runs to report.")
        return 1
    # All-empty columns (e.g. backbone without TS-Booster runs) come back after the concat.
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.reindex(columns=COLUMNS + [name for name in frame if name not in COLUMNS])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        frame.to_csv(output, index=False)
        summarize(frame).to_csv(output.with_name(output.stem + "_summary.csv"), index=False)
    else:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="raw", index=False)
            summarize(frame).to_excel(writer, sheet_name="summary", index=False)
    print(f"Report: {len(frame)} runs -> {output}")
    return 0


def missing_data(commands):
    """Local CSVs that custom-data commands read but that do not exist."""
    missing = set()
    for argv in commands:
        options = _options(argv)
        if options.get("data") != "custom" or not isinstance(options.get("data_path"), str):
            continue
        root = Path(str(options.get("root_path", ".")))
        path = (root if root.is_absolute() else PROJECT_ROOT / root) / options["data_path"]
        if not path.is_file():
            missing.add(str(path))
    return sorted(missing)


def sweep_main(defaults=None, argv=None, max_parallel=1, experiments=None, default_experiments=None,
               descriptions=None, run_dir_name=None, threads_per_job=None):
    """Run one sweep, or named experiments that share one manifest (identical runs train once).

    experiments maps names to sweep configs (an optional "description" shows in --list);
    --experiment selects them (default: default_experiments, else all), --config runs a
    JSON sweep instead. Missing data files stop the sweep before anything runs.
    """
    if experiments and default_experiments is None:
        default_experiments = list(experiments)
    parser = argparse.ArgumentParser(description="TS-Booster batch sweep")
    parser.add_argument("--config", type=Path, help="JSON sweep file (overrides --experiment)")
    if experiments:
        names = ", ".join(experiments)
        parser.add_argument("--experiment", nargs="+", metavar="NAME",
                            help=f"experiments to run, or 'all' (default: {' '.join(default_experiments)}); {names}")
        parser.add_argument("--list", action="store_true", help="list experiments and their run counts")
    parser.add_argument("--datasets", nargs="+", help="run only these data names")
    parser.add_argument("--seeds", nargs="+", type=int, help="run only these seeds")
    parser.add_argument("--models", nargs="+", help="run only these model names")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override a run option for every run, e.g. --set train_epochs=2")
    parser.add_argument("--print_config", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=max_parallel,
                        help="concurrent runs (default for None: one per available CPU, at most 32 per GPU)")
    parser.add_argument("--threads_per_job", type=int, default=threads_per_job,
                        help="CPU threads per run (default for None: available CPUs, after any cgroup quota, "
                             "/ max_parallel)")
    parser.add_argument("--gpus", nargs="+", help="CUDA device ids to spread runs over, e.g. --gpus 0 1")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--gpu_type", choices=["auto", "cpu", "cuda", "mps"], default=None)
    parser.add_argument("--use_dtw", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--inverse", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no_plots", action="store_true", default=None)
    parser.add_argument("--plots", dest="no_plots", action="store_false")
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--report", type=Path, help="after the sweep, write an .xlsx or .csv report of its runs")
    parser.add_argument("--report_scale", choices=["primary", "normalized", "original"], default="primary")
    args = parser.parse_args(argv)
    if args.config:
        configs = {args.config.stem: json.loads(args.config.read_text())}
    elif experiments:
        selected = list(experiments) if args.experiment == ["all"] else (args.experiment or list(default_experiments))
        unknown = [name for name in selected if name not in experiments]
        if unknown:
            parser.error(f"Unknown experiments {unknown}; choose from {list(experiments)} or 'all'")
        configs = {name: experiments[name] for name in selected}
    else:
        configs = {"default": defaults}
    try:
        configs = {name: apply_selection(config, args) for name, config in configs.items()}
    except ValueError as error:
        parser.error(str(error))
    if getattr(args, "list", False):
        for name, config in experiments.items():
            note = (descriptions or {}).get(name) or config.get("description", "")
            print(f"{name:24s} {count_runs(apply_selection(config, args)):6d} runs  {note}")
        return 0
    if args.print_config:
        printable = next(iter(configs.values())) if len(configs) == 1 else configs
        print(json.dumps(printable, indent=2, ensure_ascii=False))
        return 0
    commands, seen = [], set()
    for name, config in configs.items():
        built = build_commands(config) if count_runs(config) else []
        fresh = [argv for argv in built if tuple(argv) not in seen]
        seen.update(map(tuple, fresh))
        commands.extend(fresh)
        if len(configs) > 1:
            print(f"{name}: {len(built)} runs ({len(built) - len(fresh)} shared with earlier experiments)")
    if not commands:
        parser.error("The selection contains no runs.")
    first = next(iter(configs.values()))
    directory = args.run_dir or PROJECT_ROOT / "batch_runs" / (run_dir_name or first["shared"]["task_name"])
    if args.max_parallel is None:
        args.max_parallel = default_parallel(args.gpus)
    if args.max_parallel <= 0:
        parser.error("--max_parallel must be positive")
    threads = args.threads_per_job or max(1, available_cpus() // args.max_parallel)
    missing = missing_data(commands)
    if missing:
        print(f"{len(missing)} data files are missing, e.g. {missing[0]}; build them with "
              "tools/data/build_volatility_dataset.py (see README, Data).", flush=True)
        if not args.dry_run:
            return 2
    code = run_commands(commands, directory, args.max_parallel, args.dry_run, args.timeout,
                        threads_per_job=threads, gpus=args.gpus)
    if args.report and not args.dry_run:
        write_report(result_directories(commands, directory), args.report, args.report_scale)
    return code
