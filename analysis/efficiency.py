"""Table 8 (registered parameter counts; latency pending a GPU benchmark) and Fig. 6 (parameter-matched
comparison) of spec §5.5.

    python analysis/efficiency.py [--results results] [--output analysis/output/efficiency]
"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import (BACKBONE_LABEL, BACKBONES, DATASETS, PALETTE, PAPER_DEEP, SEEDS, TSLIB_DEEP,  # noqa: E402
                             dataset_means, deep, load_runs, require, tsb, write_figure, write_json, write_table)
from analysis.stats import compare, overall, reductions  # noqa: E402

PENDING = "pending GPU benchmark"
LATER = {"train_epochs": 120}
MLP = {"model": "MLP7K", "train_epochs": 120, "seq_len": 30, "pred_len": 3, "pred_offset": 0}
MATCHED = {"Vanilla Linear": tsb("TSBooster-vanilla", **LATER), "MLP-7K": MLP, "TS-Booster+Linear": tsb(**LATER)}
# article_v2 Table 8, §4.4 and Fig. 6.
PAPER = {"table8": {"Linear": (366, 7156, 0.23, 1.35), "LSTM": (50883, 57673, 0.36, 1.38), "GRU": (38211, 45001, 0.39, 1.45),
                    "Transformer": (67331, 74121, 0.78, 1.98), "DLinear": (372, 0.20), "PatchTST": (402823, 1.28),
                    "iTransformer": (269571, 0.40)},
         "added_parameters": 6790, "module_parameters_by_lookback": {30: 6790, 60: 12190, 90: 17590},
         "fig6_rmse": {"Vanilla Linear": 4.6992, "MLP-7K": 4.7698, "TS-Booster+Linear": 3.7761},
         "fig6_mda": {"Vanilla Linear": 0.6203, "MLP-7K": 0.6878, "TS-Booster+Linear": 0.7078},
         "fig6_reduction_vs_vanilla_pct": 19.6, "mlp_parameters": 7171}


def _params(frame, filters, datasets, seeds, label, allow_missing):
    rows = require(frame, filters, datasets, seeds, label, allow_missing)
    values = rows["params"].dropna().unique()
    if len(values) != 1:
        raise ValueError(f"{label}: the parameter count varies across runs: {sorted(values)}")
    return int(values[0])


def find_benchmark(data_root):
    """The newest GPU run of tools/benchmark.py in benchmark_results/ next to dataset/, or None."""
    folder = Path(data_root).parent / "benchmark_results"
    candidates = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("environment", {}).get("device") == "cuda" and "models" in data.get("models", {}):
            candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None
    _, path, data = max(candidates, key=lambda item: item[0])
    configuration = data["models"].get("configuration", {})
    return {"path": str(path), "device_name": data["environment"].get("device_name", "GPU"),
            "batch": configuration.get("batch"), "forwards": configuration.get("forwards_per_round"),
            "rounds": configuration.get("rounds"),
            "latency_ms": {entry: report["batch_forward_ms"]["median"] for entry, report in data["models"]["models"].items()}}


def run(frame, out_dir, *, datasets=DATASETS, seeds=SEEDS, allow_missing=False, bootstrap=5000,
        data_root=ROOT / "dataset"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    benchmark = find_benchmark(data_root)

    def latency(entry):
        if benchmark is None:
            return PENDING
        return benchmark["latency_ms"].get(entry, "not measured")

    # Table 8: registered parameters and, when a GPU benchmark exists, median batch latency.
    rows, added = [], {}
    for backbone in BACKBONES:
        vanilla = _params(frame, tsb("TSBooster-vanilla", backbone), datasets, seeds, f"Vanilla {backbone}", allow_missing)
        full = _params(frame, tsb("TSBooster", backbone), datasets, seeds, f"TS-Booster {backbone}", allow_missing)
        added[backbone] = full - vanilla
        rows.append({"Model": BACKBONE_LABEL[backbone], "Vanilla params": vanilla, "+TS-Booster params": full,
                     "Added params": full - vanilla, "Vanilla latency (ms)": latency(f"TSBooster-vanilla:{backbone}"),
                     "+TS-Booster latency (ms)": latency(f"TSBooster:{backbone}")})
    for model in PAPER_DEEP:
        rows.append({"Model": f"{model[:-6]} (paper impl.)", "Vanilla params": _params(frame, deep(model, 1e-3), datasets, seeds, model, allow_missing),
                     "+TS-Booster params": "—", "Added params": "—", "Vanilla latency (ms)": latency(model), "+TS-Booster latency (ms)": "—"})
    for model in TSLIB_DEEP:
        rows.append({"Model": f"{model} (TSLib)", "Vanilla params": _params(frame, deep(model, 1e-3), datasets, seeds, model, allow_missing),
                     "+TS-Booster params": "—", "Added params": "—", "Vanilla latency (ms)": latency(model), "+TS-Booster latency (ms)": "—"})
    rows.append({"Model": "MLP-7K", "Vanilla params": _params(frame, MLP, datasets, seeds, "MLP-7K", allow_missing),
                 "+TS-Booster params": "—", "Added params": "—", "Vanilla latency (ms)": latency("MLP7K"), "+TS-Booster latency (ms)": "—"})
    by_lookback = {}
    for length in (30, 60, 90):
        by_lookback[length] = (_params(frame, tsb(seq_len=length, **LATER), datasets, seeds[:3], f"TS-Booster L={length}", allow_missing)
                               - _params(frame, tsb("TSBooster-vanilla", seq_len=length, **LATER), datasets, seeds[:3], f"Vanilla L={length}", allow_missing))
    if benchmark is None:
        latency_note = ("batch inference latency needs the isolated GPU benchmark (tools/benchmark.py --device cuda), "
                        "since the campaign timings ran 32 jobs per GPU")
    else:
        latency_note = (f"latency is the median over {benchmark['rounds']} rounds of {benchmark['forwards']} forward passes "
                        f"with batch {benchmark['batch']} on {benchmark['device_name']} ({Path(benchmark['path']).name})")
    write_table(pd.DataFrame(rows), out / "table8_params", caption=f"Registered parameter counts; {latency_note}.",
                note=f"Module parameters by look-back: {', '.join(f'L={k}: {v:,}' for k, v in by_lookback.items())}.")

    # Fig. 6: parameter-matched comparison (later batch).
    means = {arm: {metric: dataset_means(frame, filters, datasets, seeds, metric, arm, allow_missing) for metric in ("rmse", "mda")}
             for arm, filters in MATCHED.items()}
    rmse = pd.DataFrame({arm: means[arm]["rmse"] for arm in MATCHED})
    mda = pd.DataFrame({arm: means[arm]["mda"] for arm in MATCHED})
    best_rmse, best_mda = rmse.idxmin(axis=1), mda.idxmax(axis=1)
    pairs = {"MLP-7K vs Vanilla Linear": ("Vanilla Linear", "MLP-7K"), "TS-Booster+Linear vs MLP-7K": ("MLP-7K", "TS-Booster+Linear"),
             "TS-Booster+Linear vs Vanilla Linear": ("Vanilla Linear", "TS-Booster+Linear")}
    tests = {}
    for name, (base, cand) in pairs.items():
        table = compare(frame, MATCHED[base], MATCHED[cand], datasets, seeds, allow_missing=allow_missing, labels=(base, cand))
        tests[name] = {**reductions(rmse[base], rmse[cand]), **overall(table["reduction_pct"]),
                       "significant": int(table["significant"].sum()), "n": len(table)}
    matched_rows = []
    for arm in MATCHED:
        reduction = reductions(rmse["Vanilla Linear"], rmse[arm])["reduction_of_mean_pct"]
        matched_rows.append({"Model": arm, "Params": _params(frame, MATCHED[arm], datasets, seeds, arm, allow_missing),
                             "Mean RMSE": float(rmse[arm].mean()), "Best RMSE on": f"{int((best_rmse == arm).sum())}/{len(rmse)}",
                             "Mean MDA": float(mda[arm].mean()), "Best MDA on": f"{int((best_mda == arm).sum())}/{len(mda)}",
                             "Reduction vs Vanilla %": reduction,
                             "Significant vs Vanilla": "—" if arm == "Vanilla Linear" else
                             f"{tests[f'{arm} vs Vanilla Linear']['significant']}/{tests[f'{arm} vs Vanilla Linear']['n']}"})
    matched = pd.DataFrame(matched_rows)
    write_table(matched, out / "table_param_matched", bold=["Mean RMSE"], pct_columns=["Reduction vs Vanilla %"],
                caption="Parameter-matched comparison (later batch, 120 epochs): Vanilla Linear, an MLP of about 7,000 "
                        "parameters and TS-Booster+Linear.")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, column, best_column, title in ((axes[0], "Mean RMSE", "Best RMSE on", "(a) RMSE (lower is better)"),
                                           (axes[1], "Mean MDA", "Best MDA on", "(b) MDA (higher is better)")):
        values = matched[column].tolist()
        bars = ax.bar(range(len(matched)), values, color=[PALETTE[1], PALETTE[2], PALETTE[0]])
        for bar, value, best in zip(bars, values, matched[best_column]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}\nbest on {best}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(matched)), [f"{row.Model}\n({row.Params:,} params)" for row in matched.itertuples()], fontsize=8)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    write_figure(fig, out / "fig6")

    summary = {"added_parameters": added, "module_parameters_by_lookback": by_lookback,
               "latency_source": None if benchmark is None else {key: benchmark[key] for key in ("path", "device_name", "batch", "forwards", "rounds")},
               "latency_ms": None if benchmark is None else {row["Model"]: (row["Vanilla latency (ms)"], row["+TS-Booster latency (ms)"]) for row in rows},
               "param_matched": {"means": {arm: {"rmse": float(rmse[arm].mean()), "mda": float(mda[arm].mean())} for arm in MATCHED},
                                 "best_on": {arm: {"rmse": f"{int((best_rmse == arm).sum())}/{len(rmse)}", "mda": f"{int((best_mda == arm).sum())}/{len(mda)}"}
                                             for arm in MATCHED},
                                 "reduction_vs_vanilla_pct": dict(zip(matched["Model"], matched["Reduction vs Vanilla %"])),
                                 "mlp_vs_vanilla_p": tests["MLP-7K vs Vanilla Linear"]["wilcoxon_p"],
                                 "tsb_vs_mlp_p": tests["TS-Booster+Linear vs MLP-7K"]["wilcoxon_p"], "tests": tests}}
    write_json({**summary, "paper": PAPER, "table8": rows}, out / "efficiency.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output" / "efficiency")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    summary = run(load_runs(args.results), args.output, allow_missing=args.allow_missing)
    print("added parameters:", summary["added_parameters"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
