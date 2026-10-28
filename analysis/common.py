"""Shared layer of the analysis (spec §3): the campaign's runs as one DataFrame, method selectors,
dataset order and labels, the test-origin mapping, and the csv/md/tex and png/pdf writers.

Every module of analysis/ works on the frame `load_runs()` returns: one row per complete run with
its selector columns (model, backbone, loss, the tsb_* settings, window, horizon, learning rate,
epochs), its original-scale metrics, per-horizon metrics, training details and diagnostics.
"""

import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.parse_result_to_xlsx import _complete_record, _result_files  # noqa: E402

# Paper order (Table 4): seven TWD crosses, three international pairs, six Taiwanese equities.
DATASETS = ["USD_TWD", "EUR_TWD", "JPY_TWD", "CNY_TWD", "THB_TWD", "KRW_TWD", "CAD_TWD",
            "EUR_USD", "GBP_USD", "USD_JPY",
            "0050_TW", "2317_TW", "2330_TW", "2454_TW", "2603_TW", "2881_TW"]
DISPLAY = {name: name[:-3] if name.endswith("_TW") else name.replace("_", "/") for name in DATASETS}
GROUPS = {"TWD FX": DATASETS[:7], "International FX": DATASETS[7:10], "Equities": DATASETS[10:]}
SEEDS = [42, 123, 456, 789, 2024]
SEEDS_3 = SEEDS[:3]
METRICS = ("rmse", "mae", "mda", "mse", "mape", "mspe", "r2")
HORIZON_METRICS = ("rmse", "mae", "mda")
BACKBONES = ["linear", "lstm", "gru", "transformer"]
BACKBONE_LABEL = {"linear": "Linear", "lstm": "LSTM", "gru": "GRU", "transformer": "Transformer"}
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 3e-3]
STATISTICAL = ["Persistence", "EWMA", "HAR-RV", "GARCH", "GJR-GARCH", "ARIMA"]
ALIGNED = ["EWMA-aligned", "GARCH-aligned", "GJR-GARCH-aligned"]
TREES = ["XGBoost", "LightGBM"]
PAPER_DEEP = ["DLinear-paper", "PatchTST-paper", "iTransformer-paper"]
TSLIB_DEEP = ["DLinear", "PatchTST", "iTransformer"]
DAGGER = {"EWMA", "GARCH", "GJR-GARCH"}  # single-day conditional volatility, not the rolling target
# The paper's TS-Booster protocol (Table 2); every TS-Booster-family selector starts from it.
DEFAULT = dict(loss="HYBRID", train_mse_weight=0.0, aux=0.3, base=0.1, enrich="trend", decomp_fixed=False,
               n_experts=2, expert_hidden=32, decomp_kernels=(7, 13), expert_kernels=(3, 5, 7), dir_weight=0.5,
               seq_len=30, pred_len=3, pred_offset=0, enc_in=2, learning_rate=1e-3, train_epochs=100)
SELECTOR_COLUMNS = ["dataset", "model", "backbone", *DEFAULT, "obs_features"]
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]  # Okabe-Ito


class MissingRuns(ValueError):
    """A module's required (method, dataset, seed) runs are absent or duplicated in results/."""


# === Method selectors ===


def tsb(variant="TSBooster", backbone="linear", **overrides):
    """A TS-Booster-family model under the paper's protocol, with per-experiment overrides."""
    return {"model": variant, "backbone": backbone, **DEFAULT, **overrides}


def classical(model, pred_offset=0):
    """A statistical or machine-learning comparator (tools/baselines/classical_baselines.py)."""
    return {"model": model, "seq_len": 30, "pred_len": 3, "pred_offset": pred_offset}


def deep(model, learning_rate=None, **overrides):
    """A deep comparator trained with MSE + Adam; learning_rate None matches the whole grid."""
    filters = {"model": model, "loss": "MSE", "train_mse_weight": 0.0, "enc_in": 2, "seq_len": 30, "pred_len": 3,
               "pred_offset": 0}
    if learning_rate is not None:
        filters["learning_rate"] = learning_rate
    return {**filters, **overrides}


# === Loading ===


def _tuple(value):
    return tuple(value) if isinstance(value, (list, tuple)) else value


def _row(result, metadata, path):
    config, details = result["config"], result.get("details", {})
    if details.get("metric_space") != "original":
        raise ValueError(f"{path.name}: metrics are in {details.get('metric_space')!r} space, not original")
    model = str(config["model"])
    row = {"dataset": Path(config["data_path"]).stem if config.get("data") == "custom" else config["data"],
           "model": model, "backbone": config.get("tsb_backbone") if model.startswith("TSBooster") else None,
           "seed": int(config["rand_seed"]), "loss": config.get("loss"),
           "train_mse_weight": float(config.get("train_mse_weight", 0.0)),
           "aux": float(config.get("tsb_aux_weight", 0.3)), "base": float(config.get("tsb_base_weight", 0.1)),
           "enrich": config.get("tsb_enrich", "trend"), "decomp_fixed": bool(config.get("tsb_decomp_fixed", False)),
           "n_experts": int(config.get("tsb_n_experts", 2)), "expert_hidden": int(config.get("tsb_expert_hidden", 32)),
           "decomp_kernels": _tuple(config.get("tsb_decomp_kernels", (7, 13))),
           "expert_kernels": _tuple(config.get("tsb_expert_kernels", (3, 5, 7))),
           "dir_weight": float(config.get("dir_weight", 0.5)), "seq_len": int(config["seq_len"]),
           "pred_len": int(config["pred_len"]), "pred_offset": int(config.get("pred_offset", 0)),
           "enc_in": int(config.get("enc_in", 0)), "learning_rate": float(config.get("learning_rate", 0.0)),
           "train_epochs": int(config.get("train_epochs", 0)),
           "obs_features": str(config.get("root_path", "")).rstrip("/").endswith("volatility_obs")}
    for metric in METRICS:
        row[metric] = result["metrics"].get(metric)
        row[f"{metric}_norm"] = details.get("metrics_normalized", {}).get(metric)
    for h, report in enumerate(details.get("metric_report_original", {}).get("per_horizon", []), start=1):
        for metric in HORIZON_METRICS:
            row[f"{metric}_h{h}"] = report.get(metric)
    diagnostics = details.get("model_diagnostics") or {}
    mixture = diagnostics.get("decomp_mixture")
    row.update(params=details.get("params"), epochs_trained=details.get("epochs_trained"),
               best_epoch=details.get("best_epoch"), best_val=details.get("best_selection_metric"),
               gate_mean=diagnostics.get("gate_mean"), gate_std=diagnostics.get("gate_std"),
               gate_min=diagnostics.get("gate_min"), gate_max=diagnostics.get("gate_max"),
               snr_w=diagnostics.get("snr_gate_w"), snr_b=diagnostics.get("snr_gate_b"),
               decomp_w={int(k): float(v) for k, v in mixture.items()} if mixture else None,
               baseline=details.get("baseline"), path=path, finished_at=metadata.get("finished_at"))
    return row


def load_runs(results_root=ROOT / "results"):
    """One row per complete run (result and metadata both complete and consistent); the latest
    attempt per configuration and seed wins."""
    rows = []
    for filename in _result_files(results_root):
        pair = _complete_record(filename)
        if pair is None:
            continue
        result, metadata = pair
        rows.append(_row(result, metadata, filename.parent))
    if not rows:
        raise ValueError(f"No complete runs under {results_root}")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("finished_at", kind="stable").drop_duplicates(SELECTOR_COLUMNS + ["seed"], keep="last")
    return frame.reset_index(drop=True)


def select(frame, **filters):
    """Rows matching every filter: equality, lists as tuples, None for null, floats within rounding."""
    mask = pd.Series(True, index=frame.index)
    for key, value in filters.items():
        column = frame[key]
        if value is None:
            mask &= column.isna()
        elif isinstance(value, (list, tuple)):
            mask &= column.apply(lambda item: isinstance(item, tuple) and item == tuple(value))
        elif isinstance(value, float):
            numbers = pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)
            mask &= pd.Series(np.isclose(numbers, value, rtol=1e-9, atol=1e-15), index=frame.index)
        else:
            mask &= column == value
    return frame[mask]


def require(frame, filters, datasets, seeds, label, allow_missing=False):
    """The runs of one method on the given datasets and seeds, exactly one per (dataset, seed).

    Missing or duplicated runs raise MissingRuns naming each of them; with allow_missing the
    incomplete datasets are dropped with a warning instead (duplicates always raise).
    """
    rows = select(frame, **filters)
    rows = rows[rows["dataset"].isin(datasets) & rows["seed"].isin(seeds)]
    counts = rows.groupby(["dataset", "seed"]).size()
    missing = [(dataset, seed) for dataset in datasets for seed in seeds if counts.get((dataset, seed), 0) == 0]
    duplicated = [(dataset, seed) for (dataset, seed), count in counts.items() if count > 1]
    if missing or duplicated:
        message = f"{label}: missing {missing}; duplicated {duplicated}"
        if duplicated or not allow_missing:
            raise MissingRuns(message)
        warnings.warn(message)
        incomplete = {dataset for dataset, _ in missing}
        rows = rows[~rows["dataset"].isin(incomplete)]
    order = {dataset: index for index, dataset in enumerate(datasets)}
    return rows.assign(_order=rows["dataset"].map(order)).sort_values(["_order", "seed"]).drop(columns="_order")


def seed_table(frame, filters, metric="rmse"):
    """{dataset: {seed: value}} for one method."""
    table = {}
    for row in select(frame, **filters).itertuples():
        table.setdefault(row.dataset, {})[int(row.seed)] = float(getattr(row, metric))
    return table


def dataset_means(frame, filters, datasets, seeds, metric="rmse", label=None, allow_missing=False):
    """Seed means of one metric per dataset, as a Series in dataset order."""
    rows = require(frame, filters, datasets, seeds, label or str(filters), allow_missing)
    means = rows.groupby("dataset")[metric].mean()
    return means.reindex([dataset for dataset in datasets if dataset in means.index])


def choose_learning_rate(frame, filters, datasets, search_seeds=SEEDS_3, candidates=LEARNING_RATES):
    """The paper's Table 6 rule: per dataset, the rate with the lowest mean validation score over the
    search seeds (validation RMSE for MSE-trained runs, the stored validation loss otherwise), ties
    going to the smaller rate. Columns: dataset, chosen, score_<rate> for every candidate."""
    rows = select(frame, **{key: value for key, value in filters.items() if key != "learning_rate"})
    rows = rows[rows["dataset"].isin(datasets) & rows["seed"].isin(search_seeds)]
    out = []
    for dataset in datasets:
        scores = {}
        for rate in candidates:
            subset = rows[(rows["dataset"] == dataset) & np.isclose(rows["learning_rate"].astype(float), rate)]
            if set(subset["seed"]) != set(search_seeds) or subset["best_val"].isna().any():
                warnings.warn(f"{filters.get('model')} {dataset} learning rate {rate:g}: search seeds incomplete; skipped")
                continue
            values = subset["best_val"].astype(float)
            if (subset["loss"] == "MSE").all():
                values = np.sqrt(values)
            scores[rate] = float(values.mean())
        if not scores:
            raise MissingRuns(f"{filters.get('model')} {dataset}: no learning rate has all search seeds {search_seeds}")
        out.append({"dataset": dataset, "chosen": min(sorted(scores), key=scores.get),
                    **{f"score_{rate:g}": scores.get(rate, math.nan) for rate in candidates}})
    return pd.DataFrame(out)


# === Data ===


def load_volatility(dataset, data_root):
    return pd.read_csv(Path(data_root) / "volatility" / f"{dataset}.csv", float_precision="round_trip")


def test_origins(dataset, n_samples, data_root):
    """Row indices and dates of the forecast origins of the n_samples test samples.

    Sample i's origin is row n - int(0.2 n) - 1 + i of dataset/volatility/<dataset>.csv (the
    Dataset_Custom test border); verified against last.npy and true.npy to float32 precision.
    """
    frame = load_volatility(dataset, data_root)
    n = len(frame)
    rows = n - int(0.2 * n) - 1 + np.arange(int(n_samples))
    if n_samples and rows[-1] >= n - 1:
        raise ValueError(f"{dataset}: {n_samples} test samples do not fit {n} rows")
    return rows, frame["date"].iloc[rows].tolist()


def next_day_return(dataset, rows, data_root):
    """The log return realized on the trading day after each origin row."""
    return load_volatility(dataset, data_root)["log_return"].to_numpy()[np.asarray(rows) + 1]


# === Labels and output ===


def display(name):
    if name in DISPLAY:
        return DISPLAY[name]
    return f"{name}†" if name in DAGGER else str(name)


def _is_nan(value):
    try:
        return value is None or math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def fmt_p(p):
    return "n/a" if _is_nan(p) else f"{float(p):.2e}"


def tex_p(p):
    if _is_nan(p):
        return "n/a"
    mantissa, exponent = f"{float(p):.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


# Symbols the tables use, as pdfLaTeX math so the .tex bodies stay ASCII (σ* before σ).
TEX_SYMBOLS = [("σ*", r"$\sigma^{*}$"), ("σ", r"$\sigma$"), ("Δ", r"$\Delta$"), ("λ", r"$\lambda$"), ("α", r"$\alpha$"),
               ("β", r"$\beta$"), ("χ²", r"$\chi^{2}$"), ("χ", r"$\chi$"), ("‖", r"$\|$"), ("−", r"$-$"), ("≤", r"$\le$"),
               ("≥", r"$\ge$"), ("×", r"$\times$"), ("±", r"$\pm$"), ("≈", r"$\approx$"), ("†", r"$^{\dagger}$"),
               ("–", "--"), ("—", "---"), ("…", r"\ldots"), ("<", r"$<$"), (">", r"$>$")]


def _escape(text):
    out = str(text)
    for char in ("&", "%", "#", "_"):
        out = out.replace(char, "\\" + char)
    for symbol, command in TEX_SYMBOLS:
        out = out.replace(symbol, command)
    return "{}" + out if out.startswith("[") else out  # a leading [ would be read as an optional argument


def _leading_number(value):
    """The number a cell starts with: 1.5282 for 1.5282, '1.5282 ± 0.0066' or '1.5282 (3)'."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)
    try:
        return float(str(value).split("±")[0].split("(")[0])
    except ValueError:
        return math.nan


def _format_cell(value, column, p_columns, pct_columns, decimals, tex):
    if column in p_columns:
        return tex_p(value) if tex else fmt_p(value)
    if column in pct_columns:
        return "n/a" if _is_nan(value) else f"{float(value):+.2f}%"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if math.isnan(value) else f"{float(value):.{decimals}f}"
    return "" if value is None else str(value)


def _bold_columns(frame, record, bold, lower_is_better):
    numbers = {column: _leading_number(record[column]) for column in bold if column in frame.columns}
    numbers = {column: number for column, number in numbers.items() if not math.isnan(number)}
    if not numbers:
        return set()
    best = min(numbers.values()) if lower_is_better else max(numbers.values())
    return {column for column, number in numbers.items() if number == best}


def _bold_cells(frame, bold, lower_is_better, higher_rows):
    """{row position: columns to bold}: across the `bold` columns within each row, or down the
    column when a single column is given (the best row of that column)."""
    bold = [column for column in bold if column in frame.columns]
    if len(bold) == 1:
        numbers = [_leading_number(value) for value in frame[bold[0]]]
        finite = [number for number in numbers if not math.isnan(number)]
        if not finite:
            return {}
        best = min(finite) if lower_is_better else max(finite)
        return {position: {bold[0]} for position, number in enumerate(numbers) if number == best}
    return {position: _bold_columns(frame, record, bold, lower_is_better != (position in set(higher_rows)))
            for position, record in frame.iterrows()}


def ordered(index, datasets):
    """The datasets of `index` in the order of `datasets`."""
    present = set(index)
    return [dataset for dataset in datasets if dataset in present]


def write_table(frame, stem, *, caption="", bold=(), lower_is_better=True, p_columns=(), pct_columns=(),
                decimals=4, note="", higher_rows=()):
    """Write stem.csv (full precision), stem.md and stem.tex (booktabs body, best of `bold` in bold).

    With several `bold` columns the best value of each row is bold; with one, the best row of that
    column. `higher_rows` lists the row positions (e.g. an MDA average) where the larger value is best.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True)
    frame.to_csv(stem.with_suffix(".csv"), index=False)
    columns = list(frame.columns)
    md_rows, tex_rows = [], []
    bold_cells = _bold_cells(frame, bold, lower_is_better, higher_rows)
    for position, record in frame.iterrows():
        strong = bold_cells.get(position, set())
        md_cells, tex_cells = [], []
        for column in columns:
            md = _format_cell(record[column], column, p_columns, pct_columns, decimals, tex=False)
            tex = _format_cell(record[column], column, p_columns, pct_columns, decimals, tex=True)
            if column not in p_columns:
                tex = _escape(tex)
            if column in strong:
                md, tex = f"**{md}**", f"\\textbf{{{tex}}}"
            md_cells.append(md.replace("|", "\\|"))
            tex_cells.append(tex)
        md_rows.append("| " + " | ".join(md_cells) + " |")
        tex_rows.append(" & ".join(tex_cells) + r" \\")
    markdown = [f"**{caption}**", ""] if caption else []
    markdown += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns), *md_rows]
    if note:
        markdown += ["", note]
    stem.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    latex = [f"% {caption}"] if caption else []
    if note:
        latex.append(f"% {note}")
    latex += [r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}", r"\toprule",
              " & ".join(_escape(column) for column in columns) + r" \\", r"\midrule", *tex_rows,
              r"\bottomrule", r"\end{tabular}"]
    stem.with_suffix(".tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_figure(fig, stem):
    """Save stem.png (300 dpi) and stem.pdf, then close the figure."""
    import matplotlib.pyplot as plt
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_jsonable) + "\n", encoding="utf-8")
