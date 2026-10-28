"""Revision controls: objective matching, future-only targets, decomposition controls,
observed-window features, reference forecasts and the batch runner's sweep features."""

import contextlib
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import run
from data_provider.data_loader import Dataset_Custom
from exp.exp_basic import Exp_Basic, LazyModelDict
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from models.capabilities import MODEL_CAPABILITIES
from tests.test_tsbooster import ORIGINAL, ROOT, config, count, load_model
from tools import select_by_validation
from tools.baselines import classical_baselines
from tools.batch_runner import apply_selection, build_commands, count_runs, expand_models, write_report
from tools.data.build_volatility_dataset import volatility_frame
from tools.stats import significance


PLAN = sorted(ROOT.glob("run_batch_long_term_forecast_[0-9][0-9].py"))


def plan_modules():
    """The setup check and the numbered experiment files, in order."""
    return [importlib.import_module(path.stem) for path in [ROOT / "run_batch_long_term_forecast.py", *PLAN]]


def importable(name):
    """False when missing or when its native library cannot load (e.g. libomp for xgboost)."""
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def load_paper_script(path):
    spec = importlib.util.spec_from_file_location("paper_" + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    saved = list(sys.path)
    try:
        with warnings.catch_warnings():  # the scripts silence warnings globally on import
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved  # and append the paper repository to sys.path
    return module


def prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d"), "Close": close})


def write_volatility(directory, observed=0, n=300):
    frame = volatility_frame(prices(n), observed_window_features=observed)
    frame.to_csv(Path(directory) / "vol.csv", index=False)
    return frame


def cli(directory, *extra):
    return ["--model_id", "ctl", "--data", "custom", "--root_path", str(directory), "--data_path", "vol.csv",
            "--features", "MS", "--target", "Volatility", "--freq", "b", "--seq_len", "16", "--label_len", "0",
            "--pred_len", "3", "--inverse", "--no_plots", "--no_use_gpu", "--results_root",
            str(Path(directory) / "results"), "--checkpoints", str(Path(directory) / "weights"), *extra]


class ObjectiveMatching(unittest.TestCase):
    def test_train_mse_weight_is_added_in_training_only(self):
        class Last(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.tensor(0.3))

            def forward(self, x, *_):
                return x[:, -1:, :].expand(-1, 2, -1) + self.bias

        experiment = Exp_Long_Term_Forecast.__new__(Exp_Long_Term_Forecast)
        experiment.args = SimpleNamespace(features="MS", label_len=1, pred_len=2, loss="MSE", train_mse_weight=0.1)
        experiment.device, experiment.model = torch.device("cpu"), Last()
        x = torch.randn(4, 5, 2)
        y = torch.randn(4, 3, 2)
        batch = (x, y, torch.zeros(4, 5, 3), torch.zeros(4, 3, 3))
        criterion = experiment._select_criterion()
        validation, _ = experiment._batch_loss(batch, criterion)
        training, _ = experiment._batch_loss(batch, criterion, training=True)
        torch.testing.assert_close(training, 1.1 * validation)


class FutureOnlyTarget(unittest.TestCase):
    def test_windows_skip_the_offset_steps(self):
        with TemporaryDirectory() as directory:
            frame = write_volatility(directory)
            args = SimpleNamespace(augmentation_ratio=0, allow_remote_data=False, seed=2, pred_offset=6)
            shifted = Dataset_Custom(args, directory, flag="test", size=[16, 0, 3], features="MS",
                                     data_path="vol.csv", target="Volatility", timeenc=1, freq="b")
            plain = Dataset_Custom(SimpleNamespace(augmentation_ratio=0, allow_remote_data=False, seed=2), directory,
                                   flag="test", size=[16, 0, 3], features="MS", data_path="vol.csv",
                                   target="Volatility", timeenc=1, freq="b")
        self.assertEqual(len(shifted), len(plain) - 6)
        x, y, _, y_mark = shifted[0]
        self.assertEqual(y.shape[0], 6 + 3)
        n = len(frame)
        start = n - int(0.2 * n)  # first test target row without the offset
        raw = shifted.inverse_transform_target(y[-3:, -1:])[:, 0]
        np.testing.assert_allclose(raw, frame.Volatility.to_numpy()[start + 6:start + 9], rtol=1e-10)

    def test_cli_trains_on_steps_seven_to_nine(self):
        with TemporaryDirectory() as directory:
            frame = write_volatility(directory)
            argv = cli(directory, "--task_name", "long_term_forecast", "--is_training", "1",
                       "--model", "TSBooster", "--enc_in", "2", "--dec_in", "2", "--c_out", "1",
                       "--pred_offset", "6", "--train_epochs", "1", "--batch_size", "16")
            self.assertEqual(run.main(argv), 0)
            result, = (Path(directory) / "results").iterdir()
            self.assertIn("_pl3_po6_", result.name)
            truth = np.load(result / "true.npy")
            n = len(frame)
            np.testing.assert_allclose(truth[0, :, 0], frame.Volatility.to_numpy()[n - int(0.2 * n) + 6:][:3],
                                       rtol=1e-5)
            self.assertEqual(json.loads((result / "result.json").read_text())["config"]["pred_offset"], 6)

    def test_offset_is_rejected_where_unsupported(self):
        base = ["--task_name", "long_term_forecast", "--is_training", "1", "--model_id", "x", "--data", "custom",
                "--features", "MS", "--enc_in", "2", "--dec_in", "2", "--c_out", "1", "--seq_len", "16",
                "--label_len", "8", "--pred_len", "3", "--pred_offset", "6"]
        for model, task in (("Transformer", "long_term_forecast"), ("Chronos", "zero_shot_forecast")):
            argv = [value if value not in ("long_term_forecast",) else task for value in base] + ["--model", model]
            with self.subTest(model=model), self.assertRaises(ValueError):
                run.validate_args(run.build_parser().parse_args(argv))


class DecompositionControls(unittest.TestCase):
    def test_width_matched_inputs_keep_the_parameter_count(self):
        for enrich in ("raw", "zeros"):
            model = load_model("TSBooster")(config(tsb_enrich=enrich))
            self.assertEqual(count(model), 7156)
            self.assertEqual(model(torch.randn(4, 30, 2), None, None, None).shape, (4, 3, 1))
        self.assertIn("width control", load_model("TSBooster")(config(tsb_enrich="raw")).effective_config["expert_input"])

    def test_fixed_decomposition_stays_equal_after_training(self):
        model = load_model("TSBooster")(config(tsb_decomp_fixed=True))
        self.assertEqual(count(model), 7154)
        self.assertIn("booster.decomp.weights", dict(model.named_buffers()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        model(torch.randn(8, 30, 2), None, None, None).square().mean().backward()
        optimizer.step()
        self.assertEqual(model.diagnostics()["decomp_mixture"], {"7": 0.5, "13": 0.5})
        self.assertIn("fixed equal weights", model.effective_config["decomp"])


class ObservedWindowFeatures(unittest.TestCase):
    def test_features_are_the_eq19_known_return_std(self):
        frame = prices()
        base = volatility_frame(frame)
        built = volatility_frame(frame, observed_window_features=3)
        self.assertEqual(list(built), ["date", "log_return", "obs_vol_h1", "obs_vol_h2", "obs_vol_h3", "Volatility"])
        self.assertEqual(len(built), len(base))
        returns = np.log(frame.Close / frame.Close.shift(1)).fillna(0.0).to_numpy()
        row = 50  # built row r corresponds to price row r + 6
        for h in (1, 2, 3):
            known = returns[row + 6 - (6 - h):row + 7]
            self.assertAlmostEqual(built[f"obs_vol_h{h}"][row], np.std(known, ddof=1) * np.sqrt(252) * 100)
        with self.assertRaises(ValueError):
            volatility_frame(frame, observed_window_features=6)


def fit(directory, *methods, seed="0", **options):
    """Run classical baselines on <directory>/vol.csv; returns {method: run directory}."""
    extra = [value for key, value in options.items() for value in (f"--{key}", str(value))]
    for method in methods:
        argv = cli(directory, "--task_name", "long_term_forecast", "--is_training", "1", "--model", method,
                   "--rand_seed", seed, *extra)
        assert classical_baselines.main(argv) == 0
    return {json.loads((p / "result.json").read_text())["config"]["model"]: p
            for p in (Path(directory) / "results").iterdir() if (p / "result.json").is_file()}


class ClassicalBaselines(unittest.TestCase):
    def test_persistence_rule_and_one_sample_comparison(self):
        with TemporaryDirectory() as directory:
            frame = write_volatility(directory)
            runs = fit(directory, "Persistence", "ObservedWindowRule")
            prediction, last = np.load(runs["Persistence"] / "pred.npy"), np.load(runs["Persistence"] / "last.npy")
            np.testing.assert_allclose(prediction[:, :, 0], np.repeat(last, 3, axis=1), rtol=1e-6)
            n = len(frame)
            first_input = frame.log_return.to_numpy()[n - int(0.2 * n) - 16:n - int(0.2 * n)]
            expected = [np.std(first_input[-(7 - h):], ddof=1) * np.sqrt(252) * 100 for h in (1, 2, 3)]
            np.testing.assert_allclose(np.load(runs["ObservedWindowRule"] / "pred.npy")[0, :, 0], expected,
                                       rtol=1e-5)
            rule = json.loads((runs["ObservedWindowRule"] / "result.json").read_text())
            self.assertEqual(rule["details"]["baseline"]["known_returns_per_step"], [6, 5, 4])
            self.assertEqual(rule["summary"]["params"], 0)
            self.assertEqual((rule["config"]["task_name"], rule["config"]["is_training"]), ("long_term_forecast", 1))
            # Seeds of a candidate against the deterministic persistence score.
            truth = np.load(runs["Persistence"] / "true.npy")
            for seed in (42, 123, 456):
                path = Path(directory) / "results" / f"candidate-{seed}"
                path.mkdir()
                ids = {"experiment_id": f"c{seed}", "group_id": "c", "attempt_id": f"c{seed}"}
                persistence_rmse = json.loads((runs["Persistence"] / "result.json").read_text())["metrics"]["rmse"]
                (path / "result.json").write_text(json.dumps({
                    "status": "complete", **ids, "metrics": {"rmse": persistence_rmse * (0.8 + seed / 1e4)},
                    "config": {"model": "TSBooster", "data": "custom", "data_path": "vol.csv", "rand_seed": seed,
                               "seq_len": 16, "pred_len": 3, "task_name": "long_term_forecast"},
                    "details": {"metric_space": "original"}}))
                (path / "metadata.json").write_text(json.dumps({"status": "complete", **ids}))
                np.save(path / "pred.npy", truth)
                np.save(path / "true.npy", truth)
            output = Path(directory) / "test.json"
            significance.main(["--results", str(Path(directory) / "results"), "--output", str(output), "paired",
                               "--baseline", "Persistence", "--candidate", "TSBooster"])
            row, = json.loads(output.read_text())["per_dataset"]
            self.assertEqual(row["test"], "one_sample")
            self.assertGreater(row["mean_gap_pct"], 15)

    def test_ewma_and_har_rv_follow_their_definitions(self):
        with TemporaryDirectory() as directory:
            frame = write_volatility(directory)
            runs = fit(directory, "EWMA", "HAR-RV")
            returns, target = frame.log_return.to_numpy(), frame.Volatility.to_numpy()
            n = len(frame)
            origin, n_train = n - int(0.2 * n) - 1, int(0.7 * n)  # last observed step of the first test window
            variance = returns[0] ** 2
            for value in returns[1:origin + 1]:
                variance = 0.94 * variance + (1 - 0.94) * value ** 2
            np.testing.assert_allclose(np.load(runs["EWMA"] / "pred.npy")[0, :, 0], np.sqrt(variance * 252) * 100,
                                       rtol=1e-5)

            def regressors(t):
                return [1.0, target[t], target[t - 4:t + 1].mean(), target[t - 21:t + 1].mean()]
            origins = np.arange(22, n_train)
            design = np.array([regressors(t) for t in origins])
            expected = [np.dot(regressors(origin), np.linalg.lstsq(design, target[origins + h], rcond=None)[0])
                        for h in (1, 2, 3)]
            np.testing.assert_allclose(np.load(runs["HAR-RV"] / "pred.npy")[0, :, 0], expected, rtol=1e-5)
            har = json.loads((runs["HAR-RV"] / "result.json").read_text())
            self.assertEqual(har["summary"]["params"], 12)
            self.assertEqual(har["details"]["baseline"]["training_origins"], [22, n_train - 1])

    def test_aligned_models_forecast_the_rolling_target(self):
        with TemporaryDirectory() as directory:
            frame = write_volatility(directory)
            runs = fit(directory, "EWMA-aligned")
            returns = frame.log_return.to_numpy()
            n = len(frame)
            origin = n - int(0.2 * n) - 1
            variance = returns[0] ** 2
            for value in returns[1:origin + 1]:
                variance = 0.94 * variance + (1 - 0.94) * value ** 2
            expected = []
            for h in (1, 2, 3):  # the 7-day target window holds 7 - h known returns and h future ones
                known = returns[origin + h - 6:origin + 1]
                second = (np.square(known).sum() + h * variance - (known.sum() ** 2 + h * variance) / 7) / 6
                expected.append(np.sqrt(second * 252) * 100)
            np.testing.assert_allclose(np.load(runs["EWMA-aligned"] / "pred.npy")[0, :, 0], expected, rtol=1e-5)
        with TemporaryDirectory() as directory:  # a window of future returns only: back to the flat EWMA forecast
            write_volatility(directory)
            methods = ["EWMA", "EWMA-aligned"] + (["GARCH-aligned"] if importable("arch") else [])
            runs = fit(directory, *methods, pred_offset=6)
            np.testing.assert_allclose(np.load(runs["EWMA-aligned"] / "pred.npy"), np.load(runs["EWMA"] / "pred.npy"),
                                       rtol=1e-5)
            if "GARCH-aligned" in runs:
                garch = np.load(runs["GARCH-aligned"] / "pred.npy")
                self.assertTrue(np.isfinite(garch).all() and (garch > 0).all())

    def test_only_fitted_long_term_runs_are_accepted(self):
        with TemporaryDirectory() as directory:
            write_volatility(directory)
            for flags in (("--task_name", "zero_shot_forecast", "--is_training", "0"),
                          ("--task_name", "long_term_forecast", "--is_training", "0")):
                with self.subTest(flags=flags), self.assertRaises(SystemExit), \
                        contextlib.redirect_stderr(io.StringIO()):
                    classical_baselines.main(cli(directory, *flags, "--model", "Persistence"))

    @unittest.skipUnless((ORIGINAL / "revision/phase2_vol_baselines/vol_baselines.py").is_file()
                         and importable("arch") and importable("pmdarima"), "paper code, arch or pmdarima missing")
    def test_methods_reproduce_the_paper_code(self):
        rng = np.random.default_rng(0)
        close = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, 406)))  # 400 rows: both codes split identically
        prices_ = pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=406).strftime("%Y-%m-%d"),
                                "Close": close})
        vol = load_paper_script(ORIGINAL / "revision/phase2_vol_baselines/vol_baselines.py")
        bench = load_paper_script(ORIGINAL / "experiments/benchmark_baselines.py")
        with TemporaryDirectory() as directory:
            (Path(directory) / "dataset").mkdir()
            prices_.to_csv(Path(directory) / "dataset" / "vol.csv", index=False)
            # As tools/data/build_volatility_dataset.py: built from the price CSV the paper code reads.
            volatility_frame(pd.read_csv(Path(directory) / "dataset" / "vol.csv")).to_csv(
                Path(directory) / "vol.csv", index=False)
            with contextlib.chdir(directory), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data, windows = vol.load_dataset("vol"), bench.extract_data("vol")
                paper = {"Persistence": vol.compute_metrics(np.tile(data["last_obs"][:, None], (1, 3)),
                                                            data["y_test"], data["last_obs"]),
                         "EWMA": vol.run_ewma(data), "HAR-RV": vol.run_har_rv(data),
                         "GARCH": vol.run_garch(data, 1, 1, 0), "GJR-GARCH": vol.run_garch(data, 1, 1, 1),
                         "ARIMA": bench.run_arima(windows)}
                for method, runner in (("XGBoost", bench.run_xgboost), ("LightGBM", bench.run_lightgbm)):
                    if importable(method.lower()):
                        paper[method] = runner(windows, seed=42)
            runs = fit(directory, *paper, seed="42", seq_len=30)
            for method, expected in paper.items():
                with self.subTest(method=method):
                    actual = json.loads((runs[method] / "result.json").read_text())["details"]["metrics_original"]
                    for name in ("rmse", "mae", "mda"):
                        self.assertTrue(math.isclose(actual[name], expected[name], rel_tol=1e-12), name)


class BatchRunner(unittest.TestCase):
    def test_grid_expands_and_set_overrides_it(self):
        models = [{"model": "TSBooster", "grid": {"learning_rate": [0.001, 0.0005], "tsb_decomp_kernels": [[7, 13], [5, 11]]}},
                  "TSBooster-vanilla"]
        self.assertEqual(len(expand_models(models)), 5)
        sweep = {"data_names": ["vol"], "windows": [[16, 0]], "pred_lens": [3], "rand_seeds": [1, 2],
                 "models": models, "shared": {"task_name": "long_term_forecast"}}
        self.assertEqual(count_runs(sweep), 10)
        selected = apply_selection(sweep, SimpleNamespace(set=["learning_rate=0.01"], datasets=None, seeds=[2],
                                                          models=["TSBooster"], num_workers=None, gpu_type=None,
                                                          use_dtw=None, inverse=None, no_plots=None))
        self.assertEqual(count_runs(selected), 2)
        self.assertEqual(selected["shared"]["learning_rate"], 0.01)
        with self.assertRaises(ValueError):
            expand_models([{"model": "TSBooster", "grid": {"learning_rate": 0.001}}])
        # Deterministic comparators run once (seed 0); --seeds narrows only seeded sweeps.
        options = dict(set=None, datasets=["USD_TWD"], seeds=[42], models=None, num_workers=None, gpu_type=None,
                       use_dtw=None, inverse=None, no_plots=None)
        classical = {"script": "tools/baselines/classical_baselines.py", "data_names": ["USD_TWD"],
                     "windows": [[30, 0]], "pred_lens": [3], "models": ["HAR-RV"],
                     "shared": {"task_name": "long_term_forecast"}}
        once = apply_selection(dict(classical, rand_seeds=[0], deterministic=True), SimpleNamespace(**options))
        seeded = apply_selection(dict(classical, rand_seeds=[42, 123]), SimpleNamespace(**options))
        self.assertEqual((once["rand_seeds"], seeded["rand_seeds"]), ([0], [42]))

    def test_classical_entrypoint_and_report(self):
        with TemporaryDirectory() as directory:
            write_volatility(directory)
            statistical = importlib.import_module("run_batch_long_term_forecast_02").EXPERIMENTS["statistical"]
            sweep = dict(statistical, data_names=["vol"], windows=[[16, 0]], models=["Persistence"])
            sweep["shared"] = dict(sweep["shared"], root_path=directory, results_root=str(Path(directory) / "results"))
            command, = build_commands(sweep)
            self.assertTrue(command[2].endswith("tools/baselines/classical_baselines.py"))
            self.assertEqual(classical_baselines.main(command[3:] + ["--no_use_gpu"]), 0)
            report = Path(directory) / "report.csv"
            self.assertEqual(write_report(list((Path(directory) / "results").iterdir()), report), 0)
            summary = pd.read_csv(Path(directory) / "report_summary.csv")
            self.assertEqual(summary.loc[0, "model"], "Persistence")
            self.assertIn("MDA_mean", summary)
            self.assertIn("backbone", summary)

    def test_plan_files_share_one_protocol_and_manifest(self):
        self.assertEqual([path.stem[-2:] for path in PLAN], [f"{i:02d}" for i in range(1, len(PLAN) + 1)])
        self.assertTrue(5 <= len(PLAN) <= 10)
        first, *rest = plan_modules()[1:]
        names = list(first.EXPERIMENTS)
        for module in rest:
            self.assertEqual(module.PROTOCOL, first.PROTOCOL, module.__name__)
            self.assertEqual(module.DATASETS, first.DATASETS, module.__name__)
            self.assertEqual((module.RUN_DIR, module.RESULTS_ROOT, module.CHECKPOINTS),
                             (first.RUN_DIR, first.RESULTS_ROOT, first.CHECKPOINTS), module.__name__)
            names.extend(module.EXPERIMENTS)
        self.assertEqual(len(names), len(set(names)))

    def test_every_batch_run_is_a_valid_long_term_forecast(self):
        commands, seen = [], set()
        for module in plan_modules():
            for name, config in module.EXPERIMENTS.items():
                if config.get("script") and config["rand_seeds"] == [0]:
                    self.assertTrue(config.get("deterministic"), f"{module.__name__}: {name}")
                fresh = [argv for argv in build_commands(config) if tuple(argv) not in seen]
                seen.update(map(tuple, fresh))
                commands.extend(fresh)
        parsers = {"run.py": run.build_parser(),
                   "tools/baselines/classical_baselines.py": classical_baselines.build_parser()}
        registry = LazyModelDict(Exp_Basic._scan_models_directory(None))
        built = set()
        for argv in commands:
            script = Path(argv[2]).relative_to(ROOT).as_posix()
            args = parsers[script].parse_args(argv[3:])
            self.assertEqual((args.task_name, args.is_training), ("long_term_forecast", 1), argv)
            if script != "run.py":
                self.assertIn(args.model, classical_baselines.METHODS)
                continue
            run.validate_args(args)
            self.assertIn(args.model, MODEL_CAPABILITIES)
            # Build and run every distinct model configuration once.
            shape = {k: v for k, v in vars(args).items()
                     if k not in {"rand_seed", "data_path", "model_id", "learning_rate", "train_epochs", "des"}}
            key = json.dumps(shape, sort_keys=True, default=str)
            if key in built:
                continue
            built.add(key)
            with contextlib.redirect_stdout(io.StringIO()):
                model = registry[args.model](args).eval()
            with torch.no_grad():
                output = model(torch.randn(2, args.seq_len, args.enc_in), None, None, None)
            self.assertEqual(tuple(output.shape[:2]), (2, args.pred_len), args.model)
        self.assertGreater(len(built), 50)


class LearningRateSelection(unittest.TestCase):
    def test_lowest_mean_validation_rmse_wins_and_every_seed_follows(self):
        with TemporaryDirectory() as directory:
            results = Path(directory) / "results"

            def write(name, metric, **config):
                path = results / name
                path.mkdir(parents=True)
                ids = {"experiment_id": name, "group_id": name, "attempt_id": name}
                config = {"model": "DLinear-paper", "data": "custom", "loss": "MSE", "seq_len": 30, "pred_len": 3,
                          **config}
                (path / "result.json").write_text(json.dumps({"status": "complete", **ids, "config": config,
                                                              "metrics": {"rmse": 1.0},
                                                              "details": {"best_selection_metric": metric}}))
                (path / "metadata.json").write_text(json.dumps({"status": "complete", **ids, "source_hash": "s",
                                                                "data": {}, "finished_at": "2026-10-22T00:00:00"}))

            # Validation MSE per seed. On "a" the mean RMSE prefers 1e-3 (0.233 < 0.25) although the mean
            # MSE prefers 1e-4; on "b" the scores tie and the smaller learning rate wins.
            scores = {("a", 0.001): [0.01, 0.01, 0.25], ("a", 0.0001): [0.0625] * 3,
                      ("b", 0.001): [0.16] * 3, ("b", 0.0001): [0.16] * 3}
            for (data, rate), values in scores.items():
                for seed, value in zip((42, 123, 456), values):
                    write(f"{data}-{rate}-{seed}", value, data_path=f"{data}.csv", learning_rate=rate, rand_seed=seed)
            write("a-0.001-789", 9.0, data_path="a.csv", learning_rate=0.001, rand_seed=789)
            write("a-0.0001-789", 0.0, data_path="a.csv", learning_rate=0.0001, rand_seed=789)
            write("tsb", 0.5, model="TSBooster", data_path="a.csv", learning_rate=0.001, rand_seed=42)
            output = Path(directory) / "selected"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(select_by_validation.main([str(results), "--output", str(output),
                                                            "--models", "DLinear-paper"]), 0)
            record = json.loads((output / "selection.json").read_text())
            self.assertEqual({row["data"]: row["chosen"]["learning_rate"] for row in record["selections"]},
                             {"a": 0.001, "b": 0.0001})
            self.assertEqual(sorted(p.name for p in output.iterdir() if p.is_dir()),
                             sorted(["a-0.001-42", "a-0.001-123", "a-0.001-456", "a-0.001-789",
                                     "b-0.0001-42", "b-0.0001-123", "b-0.0001-456", "tsb"]))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                select_by_validation.main([str(results), "--output", str(output), "--models", "DLinear-paper"])


if __name__ == "__main__":
    unittest.main()
