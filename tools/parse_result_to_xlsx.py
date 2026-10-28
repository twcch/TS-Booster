"""Export structured results, or explicitly unverified legacy text, to Excel or CSV.

    python tools/parse_result_to_xlsx.py results --output report.xlsx --scale original
    python tools/parse_result_to_xlsx.py results --output report.csv   # also writes report_summary.csv
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
_MODELS = sorted({p.stem for p in (ROOT / "models").rglob("*.py")
                  if not p.stem.startswith("_") and p.stem != "capabilities"}, key=len, reverse=True)
HEADER_RE = re.compile(
    r"^long_term_forecast_(?P<model_id>.+?)_(?P<model>" + "|".join(map(re.escape, _MODELS)) + r")"
    r"_(?P<data_type>[^_]+)_ft(?P<features>[^_]+)_sl(?P<seql_lens>\d+)_ll\d+_pl(?P<pred_lens>\d+)_"
    r".*?_seed(?P<seed>\d+)_(?P<repeat>\d+)(?:__cfg.*)?\s*$"
)
METRIC_RE = re.compile(r"(?P<key>[a-z0-9_]+):\s*(?P<val>[+-]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?))", re.I)
METRICS = ["MSE", "MAE", "RMSE", "MAPE", "R2"]
GROUP_KEYS = ["data", "model", "backbone", "des", "seql_lens", "pred_lens", "pred_offset", "config_group",
              "metric_scale"]
# Per-run cost columns (result.json summary); averaged over seeds in the summary sheet.
COSTS = ["train_time_s", "inference_time_s", "inference_ms_per_sample", "epochs_trained", "best_epoch",
         "model_size_mb", "process_peak_rss_mb"]
COLUMNS = ["seed", "seql_lens", "pred_lens", "pred_offset", "model", "backbone", "des", "data", *METRICS,
           "params", "trainable_params", *COSTS, "gpu_mem_peak_mb", "config_group", "metric_scale",
    "run_id", "metadata_verified", "finished_at"]


def _frame(rows):
    frame = pd.DataFrame(rows)
    for name in COLUMNS:
        if name not in frame:
            frame[name] = pd.Series(dtype="object")
    return frame[COLUMNS + [name for name in frame if name not in COLUMNS]]


def parse(txt_path: Path) -> pd.DataFrame:
    """Legacy text cannot recover omitted hyperparameters or data versions."""
    lines = [line.strip() for line in Path(txt_path).read_text().splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        match = HEADER_RE.match(line)
        if match is None:
            continue
        if index + 1 >= len(lines) or HEADER_RE.match(lines[index + 1]):
            warnings.warn(f"Header without metrics: {line}")
            continue
        values = {item.group("key").lower(): float(item.group("val"))
                  for item in METRIC_RE.finditer(lines[index + 1])}
        if not {"mse", "mae", "rmse"}.issubset(values):
            warnings.warn(f"Incomplete legacy metrics: {line}")
            continue
        group = re.sub(r"_seed\d+_\d+(?:__cfg.*)?$", "", line)
        rows.append({"seed": int(match["seed"]), "seql_lens": int(match["seql_lens"]),
                     "pred_lens": int(match["pred_lens"]), "model": match["model"],
                     "data": re.sub(r"_\d+_\d+_\d+$", "", match["model_id"]),
                     **{name: values.get(name.lower(), np.nan) for name in METRICS},
                     "params": values.get("params"), "trainable_params": values.get("trainable_params"),
                     "gpu_mem_peak_mb": values.get("gpu_mem_peak_mb"),
                     "config_group": "legacy:" + hashlib.sha256(group.encode()).hexdigest(),
                     "metric_scale": "legacy_unknown", "run_id": line,
                     "metadata_verified": False})
    return _frame(rows)


def _result_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / "result.json").is_file():
        return [path / "result.json"]
    return sorted(path.glob("*/result.json"))


def _complete_record(filename):
    """A complete status alone does not prove that metadata belongs to a result."""
    metadata_path = filename.parent / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        result = json.loads(filename.read_text())
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        warnings.warn(f"Unreadable result/metadata pair {filename}: {error}")
        return None
    if result.get("status") != "complete" or metadata.get("status") != "complete":
        return None
    identifiers = ("experiment_id", "group_id", "attempt_id")
    if any(not result.get(key) or result.get(key) != metadata.get(key) for key in identifiers):
        warnings.warn(f"Result/metadata identity mismatch: {filename}")
        return None
    return result, metadata


def parse_results(path: Path, scale="primary") -> pd.DataFrame:
    if scale not in {"primary", "normalized", "original"}:
        raise ValueError("scale must be primary, normalized, or original")
    rows = []
    for filename in _result_files(path):
        pair = _complete_record(filename)
        if pair is None:
            continue
        result, metadata = pair
        config, details = result["config"], result.get("details", {})
        if scale == "primary":
            metrics = result["metrics"]
            actual_scale = details.get("metric_space", "task_defined")
        else:
            metrics = details.get(f"metrics_{scale}")
            if metrics is None:
                warnings.warn(f"No {scale} metrics in {filename}")
                continue
            actual_scale = scale
        values = {name.upper(): value for name, value in metrics.items()
                  if value is None or isinstance(value, (int, float))}
        summary = result.get("summary") or {}
        costs = {name: summary.get(name) for name in COSTS}
        costs["train_time_s"] = costs["train_time_s"] or details.get("train_time_s")
        costs["inference_time_s"] = costs["inference_time_s"] or details.get("forward_time_s")
        rows.append({"seed": config["rand_seed"], "seql_lens": config["seq_len"],
                     "pred_lens": config["pred_len"], "pred_offset": config.get("pred_offset", 0),
                     "model": config["model"],
                     "backbone": config.get("tsb_backbone") if str(config["model"]).startswith("TSBooster") else None,
                     "des": config.get("des"),
                     "data": Path(config["data_path"]).stem if config.get("data") == "custom" else config["data"],
                     **values, "params": details.get("params"),
                     "trainable_params": details.get("trainable_params"), **costs,
                     "gpu_mem_peak_mb": details.get("eval_peak_allocated_mb"),
                     "config_group": result["group_id"], "metric_scale": actual_scale,
                     "run_id": result["attempt_id"], "metadata_verified": True,
                     "finished_at": metadata.get("finished_at"),
                     "task": config["task_name"]})
    return _frame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    # Old callers without extra columns are supported but cannot prove config equivalence.
    frame = frame.copy()
    for key, default in (("config_group", "legacy_unspecified"), ("metric_scale", "legacy_unknown")):
        if key not in frame:
            frame[key] = default
    attempts = frame.groupby(GROUP_KEYS, dropna=False).size()
    # Folder names and copy order do not establish which evaluation finished last.
    # Legacy text lacks timestamps, so its original record order remains the fallback.
    if "finished_at" in frame:
        frame["_completion_order"] = pd.to_datetime(frame["finished_at"], utc=True,
                                                    errors="coerce", format="mixed")
        frame = frame.sort_values("_completion_order", kind="stable", na_position="first")
    duplicates = frame.duplicated(GROUP_KEYS + ["seed"], keep="last")
    if duplicates.any():
        warnings.warn(f"{int(duplicates.sum())} repeated seed/config attempts; summary uses the latest attempt.")
    selected = frame.loc[~duplicates].copy()
    for name in COSTS:
        if name in selected:
            selected[name] = pd.to_numeric(selected[name], errors="coerce")
    grouped = selected.groupby(GROUP_KEYS, sort=False, dropna=False)
    names = [name for name in selected if name in METRICS or name.isupper()]
    summary = grouped[names].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["n_seeds"] = grouped["seed"].nunique()
    summary["n_attempts"] = attempts
    for name in ("params", "trainable_params"):
        if name in selected:
            summary[name] = grouped[name].first()
    for name in COSTS:
        if name in selected:
            summary[f"{name}_mean"], summary[f"{name}_std"] = grouped[name].mean(), grouped[name].std()
    if "gpu_mem_peak_mb" in selected:
        summary["gpu_mem_peak_mb_mean"] = grouped["gpu_mem_peak_mb"].mean()
        summary["gpu_mem_peak_mb_std"] = grouped["gpu_mem_peak_mb"].std()
    return summary.reset_index().sort_values(GROUP_KEYS).reset_index(drop=True)


def export_legacy(path, destination):
    """Regenerate compatible text from complete attempts; never append in workers."""
    if Path(path).suffix == ".txt":
        raise ValueError("Legacy text export requires structured result JSON input")
    lines = []
    for filename in _result_files(path):
        pair = _complete_record(filename)
        if pair is None:
            continue
        record, metadata = pair
        lines.extend([metadata["legacy_setting"], ", ".join(
            f"{key}:{'nan' if value is None else value}" for key, value in record["metrics"].items()), ""])
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scale", choices=["primary", "normalized", "original"], default="primary")
    parser.add_argument("--legacy_text_export", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"Input not found: {args.input}")
    frame = parse(args.input) if args.input.suffix == ".txt" else parse_results(args.input, args.scale)
    if frame.empty:
        parser.error("No complete experiment records were found.")
    output = args.output or (args.input.with_suffix(".xlsx") if args.input.is_file() else args.input / "report.xlsx")
    if args.legacy_text_export and args.input.suffix == ".txt":
        parser.error("--legacy_text_export requires structured result JSON input.")
    if args.legacy_text_export and output.resolve() == args.legacy_text_export.resolve():
        parser.error("Excel and legacy text outputs must use different paths.")
    summary_csv = output.with_name(output.stem + "_summary.csv") if output.suffix == ".csv" else None
    for destination in [output, summary_csv, args.legacy_text_export]:
        if destination and destination.exists() and not args.overwrite:
            parser.error(f"Output exists; use --overwrite explicitly: {destination}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv:
        frame.to_csv(output, index=False)
        summarize(frame).to_csv(summary_csv, index=False)
    else:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="raw", index=False)
            summarize(frame).to_excel(writer, sheet_name="summary", index=False)
    if args.legacy_text_export:
        export_legacy(args.input, args.legacy_text_export)
    print(f"Parsed {len(frame)} attempts -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
