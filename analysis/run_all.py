"""Run every analysis module in order and write SUMMARY.md and summary.json (spec §5.10).

    python analysis/run_all.py                       # everything, into analysis/output/
    python analysis/run_all.py --only backtest       # one module
    python analysis/run_all.py --bootstrap 500       # quicker Ledoit-Wolf tests

Each module reads the loaded runs once, writes its tables (csv/md/tex), figures (png/pdf) and a json
of everything it computed under analysis/output/<module>/, and returns the numbers the paper quotes.
Missing runs stop the command with the list of what is missing (exit code 2) unless --allow-missing.
"""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.common import DATASETS, SEEDS, MissingRuns, load_runs, write_json  # noqa: E402

MODULES = ["main_results", "comparators", "ablation", "sensitivity", "efficiency", "gate", "target_information",
           "backtest", "case_studies"]
TITLES = {"main_results": "Tables 3-4, Fig. 2: backbones alone and with TS-Booster",
          "comparators": "Tables 5-7, Figs. 3-5: statistical, machine-learning and deep comparators",
          "ablation": "Figs. 8-9: component and gate ablations, architecture x objective (R1.1/R2.1/R2.6)",
          "sensitivity": "Fig. 7, §4.5: horizon, look-back, settings, loss terms, kernels, decomposition controls",
          "efficiency": "Table 8, Fig. 6: parameters and the parameter-matched comparison",
          "gate": "Fig. 10, §4.7: decomposition weights and gate behaviour (R1.2/R2.2)",
          "target_information": "Known returns, the future-only target and observed-window inputs (R1.3/R1.4/R2.4)",
          "backtest": "Fig. 11: inverse-volatility backtest with Ledoit-Wolf tests (R1.7)",
          "case_studies": "0050, KRW/TWD and the dataset summary (R1.5/R1.11)"}


def _text(value):
    if isinstance(value, bool) or value is None:
        return "—" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:.4f}" if value == 0 or abs(value) >= 1e-3 else f"{value:.2e}"
    if isinstance(value, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return ", ".join(_text(item) for item in value)
        text = json.dumps(value, default=str)
        return text if len(text) <= 160 else text[:157] + "..."
    return str(value).replace("|", "\\|")


def _rows(value, prefix=""):
    """Flatten nested summaries into (dotted key, text) rows."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _rows(item, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, _text(value)


def summarize(results, args):
    lines = ["# Analysis summary", "",
             f"Results: `{args.results}`; {len(args.datasets)} datasets; seeds {args.seeds}; bootstrap {args.bootstrap}.",
             "Each module's folder holds the tables (csv, md, tex), figures (png, pdf) and a json of every number.", ""]
    for name, (summary, paper, seconds) in results.items():
        lines += [f"## {name}", "", f"_{TITLES[name]}_ — {seconds:.1f} s", "", "| Quantity | Value |", "|---|---|"]
        lines += [f"| {key} | {text} |" for key, text in _rows(summary)]
        if paper:
            lines += ["", "### Paper", "", "Values quoted in article_v2 for the same quantities.", "",
                      "| Quantity | Value |", "|---|---|"]
            lines += [f"| {key} | {text} |" for key, text in _rows(paper)]
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis" / "output")
    parser.add_argument("--data_root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--only", nargs="+", choices=MODULES, metavar="MODULE", help=f"a subset of {MODULES}")
    parser.add_argument("--allow-missing", action="store_true", help="warn about missing runs and drop their datasets")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Ledoit-Wolf bootstrap resamples")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    frame = load_runs(args.results)
    print(f"loaded {len(frame)} runs in {time.perf_counter() - started:.1f} s")
    results = {}
    for name in [module for module in MODULES if not args.only or module in args.only]:
        module = importlib.import_module(f"analysis.{name}")
        started = time.perf_counter()
        try:
            summary = module.run(frame, args.output / name, datasets=args.datasets, seeds=args.seeds,
                                 allow_missing=args.allow_missing, bootstrap=args.bootstrap, data_root=args.data_root)
        except MissingRuns as error:
            print(f"{name}: {error}", file=sys.stderr)
            return 2
        seconds = time.perf_counter() - started
        print(f"{name}: {seconds:.1f} s")
        results[name] = (summary, getattr(module, "PAPER", {}), seconds)
    args.output.mkdir(parents=True, exist_ok=True)
    # A partial run (--only) updates its modules' sections and keeps the others from the last summary.
    previous = args.output / "summary.json"
    merged = {name: (entry["summary"], entry.get("paper", {}), entry.get("seconds", float("nan")))
              for name, entry in (json.loads(previous.read_text()).items() if previous.is_file() else [])}
    merged.update(results)
    merged = {name: merged[name] for name in MODULES if name in merged}
    (args.output / "SUMMARY.md").write_text(summarize(merged, args), encoding="utf-8")
    write_json({name: {"summary": summary, "paper": paper, "seconds": seconds} for name, (summary, paper, seconds) in merged.items()},
               args.output / "summary.json")
    print(f"wrote {args.output / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
