"""Choose one value of a hyperparameter per configuration by validation score, as the paper
does for its deep comparators' learning rates, and write a results root with that choice.

    python tools/select_by_validation.py results --output results_lr_selected \\
        --vary learning_rate --models DLinear-paper PatchTST-paper iTransformer-paper

Runs of a --models model that differ only in --vary and the seed are one choice; the value
with the lowest mean validation score over --search_seeds (the paper: 42, 123, 456) wins,
ties going to the smallest value. The score is the validation RMSE (the square root of the
stored best validation loss, which ranks like the original-scale RMSE) for runs trained
with MSE, otherwise the stored loss. The output holds the chosen value's runs for every
seed and, unchanged, all runs of other models, so tools/parse_result_to_xlsx.py and
tools/stats/significance.py read it directly; selection.json records every score. Files
are hard-linked where the file system allows it, otherwise copied.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parse_result_to_xlsx import _complete_record, _result_files  # noqa: E402

# Settings that may differ between otherwise identical runs.
RUNTIME_KEYS = {"rand_seed", "seed", "iteration", "gpu", "gpu_type", "use_gpu", "use_multi_gpu", "devices",
                "device", "device_ids", "num_workers", "pin_memory", "persistent_workers", "no_plots"}
SEARCH_SEEDS = [42, 123, 456]


def validation_score(result):
    value = result.get("details", {}).get("best_selection_metric")
    if value is None:
        return None
    return math.sqrt(value) if str(result["config"].get("loss", "")).upper() == "MSE" else float(value)


def select(results, vary, models, search_seeds=SEARCH_SEEDS):
    """Return (selections, directories to keep)."""
    groups = defaultdict(lambda: defaultdict(dict))  # choice -> value -> seed -> (finished_at, score, path)
    labels, keep = {}, []
    for filename in _result_files(results):
        pair = _complete_record(filename)
        if pair is None:
            continue
        result, metadata = pair
        config = result["config"]
        if config["model"] not in models:
            keep.append(filename.parent)
            continue
        missing = [name for name in vary if name not in config]
        if missing:
            raise ValueError(f"{filename.parent.name} has no {missing} in its config")
        choice = json.dumps({"config": {k: v for k, v in config.items() if k not in RUNTIME_KEYS | set(vary)},
                             "source_hash": metadata.get("source_hash"), "data": metadata.get("data")},
                            sort_keys=True)
        value = json.dumps([config[name] for name in vary])
        attempt = (str(metadata.get("finished_at")), validation_score(result), filename.parent)
        seeds = groups[choice][value]
        if config["rand_seed"] not in seeds or attempt[0] > seeds[config["rand_seed"]][0]:
            seeds[config["rand_seed"]] = attempt
        dataset = Path(config["data_path"]).stem if config.get("data") == "custom" else config["data"]
        labels[choice] = {"model": config["model"], "data": dataset, "seq_len": config.get("seq_len"),
                          "pred_len": config.get("pred_len"), "pred_offset": config.get("pred_offset", 0),
                          "des": config.get("des")}
    selections = []
    for choice, candidates in groups.items():
        means, incomplete = {}, []
        for value, seeds in candidates.items():
            scores = [seeds[seed][1] for seed in search_seeds if seed in seeds and seeds[seed][1] is not None]
            if len(scores) == len(search_seeds):
                means[value] = sum(scores) / len(scores)
            else:
                incomplete.append(json.loads(value))
        if not means:
            warnings.warn(f"No candidate of {labels[choice]} has all search seeds {search_seeds}; skipped")
            continue
        chosen = min(sorted(means, key=json.loads), key=means.get)
        selections.append({**labels[choice], "chosen": dict(zip(vary, json.loads(chosen))),
                           "mean_validation_score": {"+".join(map(str, json.loads(v))): s for v, s in
                                                     sorted(means.items(), key=lambda item: json.loads(item[0]))},
                           "incomplete": incomplete, "seeds": sorted(candidates[chosen]),
                           "runs": [path for _, _, path in candidates[chosen].values()]})
    selections.sort(key=lambda row: (row["model"], row["data"], str(row["pred_len"]), str(row["seq_len"])))
    return selections, keep


def link_tree(source, destination):
    for path in source.rglob("*"):
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="results root")
    parser.add_argument("--output", type=Path, required=True, help="new results root (must not exist)")
    parser.add_argument("--vary", nargs="+", default=["learning_rate"], help="config keys chosen by validation")
    parser.add_argument("--models", nargs="+", required=True, help="models whose runs are selected")
    parser.add_argument("--search_seeds", nargs="+", type=int, default=SEARCH_SEEDS)
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error(f"Input not found: {args.input}")
    if args.output.exists():
        parser.error(f"Output exists: {args.output}")
    selections, keep = select(args.input, args.vary, set(args.models), args.search_seeds)
    if not selections:
        parser.error("Nothing to select: no complete runs of --models with all --search_seeds")
    for directory in keep + [path for row in selections for path in row["runs"]]:
        link_tree(directory, args.output / directory.name)
    record = {"input": str(args.input), "vary": args.vary, "search_seeds": args.search_seeds,
              "score": "validation RMSE (sqrt of best_selection_metric) for MSE runs, else best_selection_metric",
              "selections": [{**row, "runs": [path.name for path in row["runs"]]} for row in selections]}
    (args.output / "selection.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    for row in selections:
        chosen = ", ".join(f"{name}={value}" for name, value in row["chosen"].items())
        flag = f"  (incomplete: {row['incomplete']})" if row["incomplete"] else ""
        print(f"{row['model']:<20} {row['data']:<12} pl{row['pred_len']}: {chosen}{flag}")
    print(f"{len(selections)} choices, {sum(len(row['runs']) for row in selections)} selected runs and "
          f"{len(keep)} other runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
