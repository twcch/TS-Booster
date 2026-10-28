"""Tests of the analysis layer on the synthetic campaign of tests/analysis_fixture.py."""

import json
import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tests import analysis_fixture as fixture
from analysis.common import (SEEDS_3, MissingRuns, classical, load_runs, load_volatility, next_day_return, require,
                             select, test_origins, tsb, write_table)
from analysis.stats import (compare, dm_compare, holm_adjusted, holm_bonferroni, inverse_volatility,
                            ledoit_wolf_pvalue, rank_test, reductions, seed_test, sharpe)
from analysis.common import BACKBONES, choose_learning_rate, deep
from analysis import (ablation, backtest, case_studies, comparators, efficiency, gate, main_results, run_all,
                      sensitivity, target_information)

CAMPAIGN = {}


def setUpModule():
    root = Path(tempfile.mkdtemp(prefix="analysis-"))
    written = fixture.write_campaign(root)
    CAMPAIGN.update(root=root, written=written, datasets=["AA_TWD", "BB_TWD"],
                    frame=load_runs(root / "results"), out=root / "out")


def tearDownModule():
    shutil.rmtree(CAMPAIGN["root"], ignore_errors=True)


class Common(unittest.TestCase):
    def setUp(self):
        self.root, self.frame = CAMPAIGN["root"], CAMPAIGN["frame"]
        self.written, self.datasets = CAMPAIGN["written"], CAMPAIGN["datasets"]

    def test_load_runs_has_one_row_per_run(self):
        self.assertEqual(len(self.frame), self.written["runs"])
        for column in ("dataset", "model", "backbone", "seed", "loss", "train_mse_weight", "aux", "base", "enrich",
                       "decomp_fixed", "n_experts", "expert_hidden", "decomp_kernels", "expert_kernels", "dir_weight",
                       "seq_len", "pred_len", "pred_offset", "enc_in", "learning_rate", "train_epochs", "obs_features",
                       "rmse", "mda", "rmse_norm", "rmse_h1", "mda_h3", "params", "epochs_trained", "best_val",
                       "gate_mean", "snr_w", "decomp_w", "baseline", "path"):
            self.assertIn(column, self.frame.columns)
        linear = select(self.frame, **tsb())
        self.assertEqual(len(linear), 2 * 3)
        self.assertEqual(linear["decomp_kernels"].iloc[0], (7, 13))
        self.assertTrue(select(self.frame, model="Persistence")["backbone"].isna().all())

    def test_require_lists_missing_runs(self):
        with self.assertRaises(MissingRuns) as caught:
            require(self.frame, tsb(), ["AA_TWD", "ZZ_TWD"], SEEDS_3, "TS-Booster+Linear")
        self.assertIn("ZZ_TWD", str(caught.exception))
        with self.assertWarns(UserWarning):
            kept = require(self.frame, tsb(), ["AA_TWD", "ZZ_TWD"], SEEDS_3, "x", allow_missing=True)
        self.assertEqual(set(kept["dataset"]), {"AA_TWD"})

    def test_duplicate_attempts_keep_the_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volatility = fixture.write_prices(root, "AA_TWD")
            config = fixture.example_config()
            fixture.write_run(root / "results", config, volatility, attempt="a", finished_at="2026-10-23T01:00:00")
            later = fixture.write_run(root / "results", config, volatility, attempt="b", scale=2.0,
                                      finished_at="2026-10-23T02:00:00")
            frame = load_runs(root / "results")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame["rmse"].iloc[0], json.loads((later / "result.json").read_text())["metrics"]["rmse"])

    def test_normalized_metric_space_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            volatility = fixture.write_prices(root, "AA_TWD")
            run = fixture.write_run(root / "results", fixture.example_config(), volatility, metric_space="normalized")
            with self.assertRaises(ValueError) as caught:
                load_runs(root / "results")
            self.assertIn(run.name, str(caught.exception))

    def test_origins_match_last_and_next_day_return(self):
        row = select(self.frame, **tsb(), dataset="AA_TWD", seed=42).iloc[0]
        last = np.load(row["path"] / "last.npy")[:, 0]
        rows, dates = test_origins("AA_TWD", len(last), self.root / "dataset")
        volatility = load_volatility("AA_TWD", self.root / "dataset")
        np.testing.assert_allclose(volatility["Volatility"].to_numpy()[rows], last, rtol=1e-6)
        self.assertEqual(dates[0], volatility["date"].iloc[rows[0]])
        np.testing.assert_array_equal(next_day_return("AA_TWD", rows, self.root / "dataset"),
                                      volatility["log_return"].to_numpy()[rows + 1])

    def test_write_table_higher_rows_flip_the_bold_direction(self):
        frame = pd.DataFrame({"Row": ["Avg. RMSE", "Avg. MDA"], "A": [1.0, 1.0], "B": [2.0, 2.0]})
        stem = self.root / "tables" / "higher_rows"
        write_table(frame, stem, bold=["A", "B"], higher_rows=[1])
        lines = [line for line in stem.with_suffix(".tex").read_text().splitlines() if line.startswith("Avg.")]
        self.assertIn(r"\textbf{1.0000}", lines[0])
        self.assertIn(r"\textbf{2.0000}", lines[1])

    def test_write_table_single_bold_column_marks_the_best_row(self):
        frame = pd.DataFrame({"Model": ["a", "b", "c"], "Mean RMSE": [2.0, 1.0, 3.0]})
        stem = self.root / "tables" / "single_bold"
        write_table(frame, stem, bold=["Mean RMSE"])
        tex = stem.with_suffix(".tex").read_text()
        self.assertEqual(tex.count(r"\textbf{"), 1)
        self.assertIn(r"\textbf{1.0000}", tex)

    def test_write_table_maps_symbols_for_tex(self):
        frame = pd.DataFrame({"Variant": ["[x ‖ trend]", "Δ SR σ* λ_dir α β −0.5 p < 0.05 a > b"], "Value": [1.0, 2.0]})
        stem = self.root / "tables" / "symbols"
        write_table(frame, stem)
        tex = stem.with_suffix(".tex").read_text()
        for expected in (r"{}[x $\|$ trend]", r"$\Delta$", r"$\sigma^{*}$", r"$\lambda$\_dir", r"$\alpha$", r"$\beta$",
                         r"$-$0.5", r"$<$", r"$>$"):
            self.assertIn(expected, tex)
        body = tex.split(r"\midrule")[1]
        self.assertTrue(body.isascii(), body)

    def test_choose_learning_rate(self):
        chosen = choose_learning_rate(self.frame, deep("DLinear-paper"), self.datasets)
        self.assertEqual(list(chosen["chosen"]), [0.001, 0.001])
        self.assertEqual(len([column for column in chosen.columns if column.startswith("score_")]), 6)

    def test_write_table_escapes_tex(self):
        frame = pd.DataFrame({"Method": ["GJR-GARCH†", "a_b"], "RMSE": [1.5, 1.25], "Gain %": [1.0, 2.0],
                              "p": [5.97e-7, float("nan")]})
        stem = self.root / "tables" / "escape"
        write_table(frame, stem, bold=["RMSE"], p_columns=["p"], pct_columns=["Gain %"])
        tex = stem.with_suffix(".tex").read_text()
        self.assertIn(r"\textbf{1.2500}", tex)
        self.assertIn(r"$^{\dagger}$", tex)
        self.assertIn(r"a\_b", tex)
        self.assertIn(r"$5.97\times10^{-7}$", tex)
        self.assertIn("n/a", tex)
        self.assertIn(r"+2.00\%", tex)
        for suffix in (".csv", ".md"):
            self.assertTrue(stem.with_suffix(suffix).is_file(), suffix)


class Stats(unittest.TestCase):
    def setUp(self):
        self.frame, self.datasets = CAMPAIGN["frame"], CAMPAIGN["datasets"]

    def test_holm_adjusted_agrees_with_rejections(self):
        p = [0.001, 0.04, 0.012, 0.3]
        np.testing.assert_allclose(holm_adjusted(p), [0.004, 0.08, 0.036, 0.3])
        self.assertEqual(list(holm_adjusted(p) <= 0.05), holm_bonferroni(p))

    def test_holm_adjusted_ignores_nan(self):
        adjusted = holm_adjusted([0.01, float("nan"), 0.04])
        np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.04])
        self.assertTrue(np.isnan(adjusted[1]))

    def test_seed_test_constant_difference_is_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the degenerate case must not reach scipy's precision warning
            degenerate = seed_test(np.array([1.0, 2.0, 3.0]), np.array([0.9, 1.9, 2.9]), "rmse")
            regular = seed_test(np.array([1.0, 2.0, 3.0]), np.array([0.9, 1.7, 2.6]), "rmse")
        self.assertTrue(np.isnan(degenerate["p_one_sided"]))
        self.assertLess(regular["p_one_sided"], 0.05)

    def test_reductions_two_aggregations(self):
        out = reductions(pd.Series([2.0, 4.0]), pd.Series([1.0, 3.0]))
        self.assertAlmostEqual(out["mean_of_reductions_pct"], 37.5)
        self.assertAlmostEqual(out["reduction_of_mean_pct"], 100 * (3 - 2) / 3)

    def test_compare_uses_one_sample_for_a_deterministic_baseline(self):
        table = compare(self.frame, classical("Persistence"), tsb(), self.datasets, SEEDS_3, base_seeds=[0])
        self.assertEqual(set(table["test"]), {"one_sample"})
        self.assertEqual(len(table), 2)
        paired = compare(self.frame, tsb("TSBooster-vanilla"), tsb(), self.datasets, SEEDS_3)
        self.assertEqual(set(paired["test"]), {"paired"})
        self.assertTrue((paired["reduction_pct"] > 0).all())
        self.assertTrue((paired["p_holm"] >= paired["p"]).all())

    def test_dm_compare_pairs_every_seed(self):
        table = dm_compare(self.frame, tsb("TSBooster-vanilla"), tsb(), ["AA_TWD"], SEEDS_3)
        self.assertEqual(sorted(table["seed"]), SEEDS_3)
        self.assertTrue((table["dm"] > 0).all())

    def test_rank_test_orders_methods(self):
        means = pd.DataFrame({"A": [1.0, 1.0], "B": [2.0, 3.0], "C": [3.0, 2.0]}, index=["x", "y"])
        out = rank_test(means)
        self.assertEqual(out["average_ranks"]["A"], 1.0)
        self.assertEqual(out["average_ranks"]["B"], 2.5)
        self.assertEqual(out["k"], 3)

    def test_inverse_volatility_hand_example(self):
        out = inverse_volatility(np.array([30.0, 5.0, 100.0]), np.array([0.01, -0.02, 0.0]), cost_bps=10)
        np.testing.assert_allclose(out["weights"], [0.5, 3.0, 0.2])
        expected = np.array([0.5, 3.0, 0.2]) * np.expm1([0.01, -0.02, 0.0]) - 0.001 * np.array([0.5, 2.5, 2.8])
        np.testing.assert_allclose(out["returns"], expected)
        self.assertAlmostEqual(out["nav"], float(np.prod(1 + expected)))
        self.assertAlmostEqual(sharpe(np.array([0.01, 0.02, 0.03])), np.sqrt(252) * 0.02 / np.std([0.01, 0.02, 0.03]))

    def test_inverse_volatility_rejects_nan(self):
        with self.assertRaises(ValueError):
            inverse_volatility(np.array([30.0, np.nan]), np.array([0.01, 0.02]))

    def test_ledoit_wolf_direction_and_reproducibility(self):
        rng = np.random.default_rng(3)
        r2 = rng.normal(0, 0.01, 300)
        r1 = r2 + 0.003
        self.assertLess(ledoit_wolf_pvalue(r1, r2, n_boot=400)["p"], 0.05)
        self.assertGreater(ledoit_wolf_pvalue(r2, r1, n_boot=400)["p"], 0.5)
        same = ledoit_wolf_pvalue(r2, r2, n_boot=400)
        self.assertEqual((same["t_obs"], same["p"]), (0.0, 1.0))  # identical series: no evidence either way
        null = [ledoit_wolf_pvalue(rng.normal(0, 0.01, 200), rng.normal(0, 0.01, 200), n_boot=200, seed=k)["p"]
                for k in range(20)]
        self.assertLessEqual(sum(p < 0.05 for p in null), 4)
        self.assertTrue(0.3 < np.mean(null) < 0.7)
        self.assertEqual(ledoit_wolf_pvalue(r1, r2, n_boot=400, seed=1), ledoit_wolf_pvalue(r1, r2, n_boot=400, seed=1))

    def test_ledoit_wolf_short_series_is_nan(self):
        self.assertTrue(np.isnan(ledoit_wolf_pvalue(np.ones(20) * 0.01, np.zeros(20), n_boot=50)["p"]))


class Modules(unittest.TestCase):
    def setUp(self):
        self.root, self.frame, self.out = CAMPAIGN["root"], CAMPAIGN["frame"], CAMPAIGN["out"]
        self.datasets = CAMPAIGN["datasets"]

    def test_main_results(self):
        out = self.out / "main_results"
        summary = main_results.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("table3.tex", "table4.md", "cross_backbone_reductions.csv", "fig2_heatmap.png", "fig2_bars.pdf",
                     "main_results.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(summary["positive_combinations"], 4 * len(self.datasets))
        self.assertEqual(set(summary["reduction_by_backbone_pct"]), set(BACKBONES))

    def test_comparators(self):
        out = self.out / "comparators"
        summary = comparators.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("table5.tex", "table6.tex", "table6_tslib.md", "table6_matched.md", "table6_selected_lr.csv",
                     "table7.tex", "table_dm.csv", "fig3_cd_statistical.png", "fig4_cd_ml.pdf", "fig5_cd_deep.png",
                     "arima_orders.md", "reference_rule.md", "comparators.json"):
            self.assertTrue((out / name).is_file(), name)
        table7 = pd.read_csv(out / "table7.csv")
        self.assertIn("Persistence", set(table7["Comparator"]))
        self.assertEqual(summary["rank_tests"]["statistical"]["k"], 7)
        self.assertGreater(summary["vs_har_rv"]["reduction_of_mean_pct"], 0)

    def test_comparators_keep_the_dataset_order(self):
        out = self.out / "comparators_reversed"
        comparators.run(self.frame, out, datasets=list(reversed(self.datasets)), seeds=SEEDS_3)
        for name in ("table5", "table5_mda", "table6", "table6_tslib"):
            table = pd.read_csv(out / f"{name}.csv")
            self.assertEqual(list(table["Dataset"].iloc[:2]), list(reversed(self.datasets)), name)
        gain = pd.read_csv(self.out / "gate" / "table_gate_vs_gain.csv") if (self.out / "gate" / "table_gate_vs_gain.csv").is_file() else None
        if gain is not None:
            self.assertEqual(list(gain["Dataset"]), self.datasets)

    def test_ablation(self):
        out = self.out / "ablation"
        summary = ablation.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("fig8.png", "table_ablation.tex", "table_objective_factorial.tex", "table_objective_contrasts.md",
                     "table_incremental.tex", "fig9.pdf", "table_gate_ablation.md", "ablation.json"):
            self.assertTrue((out / name).is_file(), name)
        means = summary["fig8_means"]
        self.assertLess(means["Full"], means["+Expert"])
        self.assertLess(means["+Expert"], means["Vanilla"])
        self.assertEqual(set(summary["factorial_contrasts"]), {"a", "b", "c", "d"})

    def test_sensitivity(self):
        out = self.out / "sensitivity"
        summary = sensitivity.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("fig7_horizon.png", "fig7_lookback.pdf", "table_horizon.tex", "table_lookback.md",
                     "table_sensitivity.tex", "table_loss_components.tex", "table_expert_kernels.tex",
                     "table_decomposition_controls.tex", "table_decomp_backbones.tex", "sensitivity.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(set(summary["horizon"]), {1, 3, 5, 10})
        self.assertEqual(set(summary["lookback"]), {30, 60, 90})
        self.assertEqual(set(summary["expert_kernels_change_pct"]), {"[3]", "[5]", "[7]", "[3, 7]", "[3, 5, 7, 9]"})

    def test_efficiency(self):
        out = self.out / "efficiency"
        summary = efficiency.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3, data_root=self.root / "dataset")
        for name in ("table8_params.tex", "fig6.png", "table_param_matched.tex", "efficiency.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(summary["added_parameters"]["linear"], 7156 - 366)
        self.assertIn("pending GPU benchmark", (out / "table8_params.md").read_text())
        self.assertIsNone(summary["latency_source"])

    def test_efficiency_fills_latency_from_a_benchmark_next_to_the_data(self):
        benchmark = self.root / "benchmark_results" / "gpu_test.json"
        benchmark.parent.mkdir(exist_ok=True)
        models = {name: {"params": 1, "batch_forward_ms": {"median": value}} for name, value in
                  (("TSBooster-vanilla:linear", 0.075), ("TSBooster:linear", 1.436), ("DLinear-paper", 0.379), ("MLP7K", 0.151))}
        benchmark.write_text(json.dumps({"environment": {"device": "cuda", "device_name": "Test GPU"},
                                         "models": {"configuration": {"batch": 32, "forwards_per_round": 200, "rounds": 3},
                                                    "models": models}}))
        try:
            out = self.out / "efficiency_latency"
            summary = efficiency.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3, data_root=self.root / "dataset")
            table = pd.read_csv(out / "table8_params.csv").set_index("Model")
            self.assertEqual(summary["latency_source"]["device_name"], "Test GPU")
            self.assertAlmostEqual(float(table.loc["Linear", "+TS-Booster latency (ms)"]), 1.436)
            self.assertAlmostEqual(float(table.loc["DLinear (paper impl.)", "Vanilla latency (ms)"]), 0.379)
            self.assertEqual(table.loc["LSTM", "Vanilla latency (ms)"], "not measured")
            self.assertIn("Test GPU", (out / "table8_params.md").read_text())
        finally:
            benchmark.unlink()

    def test_gate(self):
        out = self.out / "gate"
        summary = gate.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("fig10_decomp_weights.png", "fig10_gate.pdf", "fig10_gate_seeds.png", "table_gate_parameters.csv",
                     "table_gate_variance.md", "fig_gate_distribution.png", "table_gate_vs_gain.md", "gate.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertAlmostEqual(sum(summary["variance_share"].values()), 1.0, places=6)
        self.assertEqual(set(summary["variance_share"]), {"within_run", "between_seeds", "between_datasets"})
        self.assertGreater(summary["variance_share"]["between_seeds"], 0.0)
        self.assertEqual(len(pd.read_csv(out / "table_gate_parameters.csv")), len(self.datasets) * 3)

    def test_target_information(self):
        out = self.out / "target_information"
        summary = target_information.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3)
        for name in ("table_per_horizon.tex", "fig_known_share.png", "table_future_backbones.tex",
                     "table_future_comparators.tex", "table_future_significance.tex", "table_future_deep.tex",
                     "table_observed_features.tex", "target_information.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(set(summary["per_horizon_reduction_pct"]), {1, 2, 3})
        self.assertEqual(set(summary["future_backbones"]), set(BACKBONES))
        self.assertEqual(set(summary["future_deep_reduction_of_mean_pct"]), {"DLinear", "PatchTST", "iTransformer"})
        significance = pd.read_csv(out / "table_future_significance.csv")
        self.assertIn("DLinear", set(significance["Comparator"]))

    def test_backtest(self):
        out = self.out / "backtest"
        summary = backtest.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3, bootstrap=200,
                               data_root=self.root / "dataset")
        for name in ("fig11.png", "fig11_seeds.pdf", "table_backtest.tex", "table_backtest_seeds.tex",
                     "table_backtest_methods.tex", "table_backtest_sensitivity.md", "backtest.json"):
            self.assertTrue((out / name).is_file(), name)
        table = pd.read_csv(out / "table_backtest.csv")
        self.assertEqual(len(table), len(self.datasets))
        self.assertEqual(len(summary["sensitivity"]), 9)
        self.assertTrue(all(0 <= p <= 1 for p in table["p"].dropna()))
        saved = json.loads((out / "backtest.json").read_text())
        self.assertIn("t_obs", saved["ledoit_wolf"]["42"][self.datasets[0]])
        self.assertIn("Persistence", saved["sharpe_by_dataset"])

    def test_backtest_rejects_shifted_windows(self):
        row = select(self.frame, **tsb("TSBooster-vanilla"), dataset="AA_TWD", seed=42).iloc[0]
        volatility = load_volatility("AA_TWD", self.root / "dataset")["Volatility"].to_numpy()
        rows, _ = test_origins("AA_TWD", len(np.load(row["path"] / "last.npy")), self.root / "dataset")
        backtest.check_windows(row["path"], volatility, rows)  # aligned: no error
        with self.assertRaises(ValueError):
            backtest.check_windows(row["path"], volatility, rows + 1)

    def test_unchanged_share_uses_the_data(self):
        values = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
        self.assertAlmostEqual(case_studies.unchanged_share(values, np.array([0, 3]), 3), 100 * 3 / 6)

    def test_case_studies(self):
        out = self.out / "case_studies"
        summary = case_studies.run(self.frame, out, datasets=self.datasets, seeds=SEEDS_3, data_root=self.root / "dataset")
        for name in ("table_datasets.tex", "table_0050_backbones.md", "table_0050_comparators.tex",
                     "table_0050_bias_tercile.md", "table_0050_bias_year.md", "case_studies.json"):
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(summary["focus_dataset"], self.datasets[0])
        self.assertIsNone(summary["krw"])
        rows = pd.read_csv(out / "table_datasets.csv")
        self.assertEqual(len(rows), len(self.datasets))
        self.assertIn("First test origin", rows.columns)


class RunAll(unittest.TestCase):
    def setUp(self):
        self.root, self.datasets = CAMPAIGN["root"], CAMPAIGN["datasets"]

    def test_run_all_only(self):
        out = self.root / "all"
        code = run_all.main(["--results", str(self.root / "results"), "--output", str(out),
                             "--data_root", str(self.root / "dataset"), "--datasets", *self.datasets,
                             "--seeds", "42", "123", "456", "--bootstrap", "200", "--only", "main_results", "efficiency"])
        self.assertEqual(code, 0)
        summary = (out / "SUMMARY.md").read_text()
        self.assertIn("## main_results", summary)
        self.assertIn("## efficiency", summary)
        self.assertNotIn("## backtest", summary)
        self.assertTrue((out / "summary.json").is_file())

    def test_run_all_only_merges_into_the_existing_summary(self):
        out = self.root / "merged"
        common = ["--results", str(self.root / "results"), "--output", str(out), "--data_root", str(self.root / "dataset"),
                  "--datasets", *self.datasets, "--seeds", "42", "123", "456", "--bootstrap", "200"]
        self.assertEqual(run_all.main([*common, "--only", "main_results"]), 0)
        self.assertEqual(run_all.main([*common, "--only", "efficiency"]), 0)
        summary = (out / "SUMMARY.md").read_text()
        self.assertIn("## main_results", summary)
        self.assertIn("## efficiency", summary)
        self.assertEqual(set(json.loads((out / "summary.json").read_text())), {"main_results", "efficiency"})

    def test_tex_tables_compile_safe(self):
        out = self.root / "all"  # written by test_run_all_only (alphabetically earlier) or here
        if not (out / "SUMMARY.md").is_file():
            run_all.main(["--results", str(self.root / "results"), "--output", str(out), "--data_root", str(self.root / "dataset"),
                          "--datasets", *self.datasets, "--seeds", "42", "123", "456", "--bootstrap", "200"])
        full = self.root / "all_modules"
        run_all.main(["--results", str(self.root / "results"), "--output", str(full), "--data_root", str(self.root / "dataset"),
                      "--datasets", *self.datasets, "--seeds", "42", "123", "456", "--bootstrap", "200"])
        checked = 0
        for path in full.rglob("*.tex"):
            body = path.read_text().split(r"\midrule", 1)[1]
            self.assertTrue(body.isascii(), f"{path.name} has non-ASCII in its body")
            for line in body.splitlines():
                self.assertFalse(line.startswith("["), f"{path.name}: a row starts with [")
                self.assertNotIn("& [", line, f"{path.name}: a cell starts with [")
            checked += 1
        self.assertGreater(checked, 40)

    def test_run_all_rejects_unknown_module(self):
        with self.assertRaises(SystemExit):
            run_all.main(["--only", "nope"])

    def test_run_all_reports_missing_runs(self):
        code = run_all.main(["--results", str(self.root / "results"), "--output", str(self.root / "missing"),
                             "--data_root", str(self.root / "dataset"), "--datasets", "AA_TWD", "ZZ_TWD",
                             "--seeds", "42", "123", "456", "--only", "main_results"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
