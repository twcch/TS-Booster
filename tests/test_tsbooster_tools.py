"""Data and statistics tools for TS-Booster: parity with the journal pipeline."""

import importlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_Custom
from tools.data import adjust_tw_prices, repair_fx
from tools.data.build_volatility_dataset import build, volatility_frame
from tools.stats import significance

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "docs" / "TS-Booster_ori"


def prices(n=260, seed=0):
    rng = np.random.default_rng(seed)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Date": pd.bdate_range("2015-01-01", periods=n).strftime("%Y-%m-%d"), "Close": close})


class VolatilityDataset(unittest.TestCase):
    def test_features_follow_the_paper_formula_and_drop_the_warmup(self):
        frame = prices()
        built = volatility_frame(frame)
        returns = np.log(frame.Close / frame.Close.shift(1)).fillna(0.0)
        expected = (returns.rolling(7).std() * np.sqrt(252) * 100).dropna()
        self.assertEqual(len(built), len(frame) - 6)
        np.testing.assert_array_equal(built.Volatility.to_numpy(), expected.to_numpy())
        self.assertEqual(list(built), ["date", "log_return", "Volatility"])

    def test_jumps_are_reported_not_altered(self):
        frame = prices()
        frame.loc[100:, "Close"] /= 4  # an unadjusted 1:4 split
        with TemporaryDirectory() as directory:
            source = Path(directory) / "split.csv"
            frame.to_csv(source, index=False)
            target, report = build(source, Path(directory) / "out")
            self.assertEqual([move["date"] for move in report["large_moves"]], [frame.Date[100]])
            self.assertTrue(target.with_suffix(".csv.manifest.json").is_file())
            with self.assertRaises(FileExistsError):
                build(source, Path(directory) / "out")

    @unittest.skipUnless((ORIGINAL / "src/utils/data_loader.py").is_file(), "journal code not present")
    def test_loader_matches_prepare_data_scaler_and_train_windows(self):
        sys.path.insert(0, str(ORIGINAL))
        try:
            prepare_data = importlib.import_module("src.utils.data_loader").prepare_data
            frame = prices(523)
            train, val, test, scaler = prepare_data(frame, lookback=30, horizon=3, batch_size=32)
        finally:
            sys.path.remove(str(ORIGINAL))
            for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
                del sys.modules[name]
        with TemporaryDirectory() as directory:
            volatility_frame(frame).to_csv(Path(directory) / "vol.csv", index=False)
            args = SimpleNamespace(augmentation_ratio=0, allow_remote_data=False, seed=2)
            datasets = {flag: Dataset_Custom(args, directory, flag=flag, size=[30, 0, 3], features="MS",
                                             data_path="vol.csv", target="Volatility", timeenc=1, freq="b")
                        for flag in ("train", "val", "test")}
        # TSLib orders [log_return, Volatility]; the journal code [Volatility, log_return].
        np.testing.assert_allclose(datasets["train"].scaler.mean_[::-1], scaler.mean_, rtol=1e-12)
        np.testing.assert_allclose(datasets["train"].scaler.scale_[::-1], scaler.scale_, rtol=1e-12)
        self.assertEqual(len(datasets["train"]), len(train.dataset))
        # n = 517 is not a multiple of five: TSLib's test border n - int(0.2n) is one row later.
        self.assertEqual(len(datasets["val"]), len(val.dataset) + 1)
        self.assertEqual(len(datasets["test"]), len(test.dataset) - 1)


def write_run(root, name, model, dataset, seed, rmse, pred, true, **config):
    run = root / name
    run.mkdir(parents=True)
    ids = {"experiment_id": name, "group_id": model, "attempt_id": name}
    full = {"model": model, "data": "custom", "data_path": f"{dataset}.csv", "rand_seed": seed,
            "seq_len": 30, "pred_len": 3, "task_name": "long_term_forecast", **config}
    (run / "result.json").write_text(json.dumps({
        "status": "complete", **ids, "config": full, "metrics": {"rmse": rmse, "mse": rmse ** 2},
        "details": {"metric_space": "original"}}))
    (run / "metadata.json").write_text(json.dumps({"status": "complete", **ids,
                                                    "finished_at": f"2026-01-01T00:00:{seed % 60:02d}"}))
    np.save(run / "pred.npy", pred)
    np.save(run / "true.npy", true)


class TaiwanPriceAdjustment(unittest.TestCase):
    def test_reference_prices_adjust_every_earlier_close(self):
        dates = pd.bdate_range("2020-01-01", periods=6)
        prices = pd.DataFrame({"date": dates, "close": [100.0, 102.0, 95.0, 96.0, 190.0, 191.0]})
        events = [(f"{dates[2]:%Y-%m-%d}", 97.0 / 102.0, "ex-息", {"before_price": 102.0, "reference_price": 97.0}),
                  (f"{dates[4]:%Y-%m-%d}", 192.0 / 96.0, "capital reduction",
                   {"last_close": 96.0, "reference_price": 192.0})]
        adjusted = adjust_tw_prices.adjust(prices, events)
        returns = np.log(adjusted / adjusted.shift(1)).to_numpy()[1:]
        expected = np.log([102 / 100, 95 / 97, 96 / 95, 190 / 192, 191 / 190])  # ex-dates measured from the reference
        np.testing.assert_allclose(returns, expected, rtol=1e-12)
        self.assertEqual(adjusted.iloc[-1], 191.0)
        with self.assertRaises(ValueError):  # the event's last close must match the price series
            adjust_tw_prices.adjust(prices, [(events[0][0], 0.9, "ex-息", {"before_price": 101.0})])

    def test_moves_beyond_the_daily_limit_are_reported(self):
        dates = pd.DatetimeIndex(["2014-01-02", "2014-01-03", "2020-01-02", "2020-01-03", "2020-01-06"])
        closes = pd.Series([100.0, 108.0, 100.0, 109.0, 121.0], index=dates)  # +8% (limit 7%), +9%, +11% (limit 10%)
        flagged = adjust_tw_prices.limit_violations(closes)
        self.assertEqual([f"{day:%Y-%m-%d}" for day in flagged.index], ["2014-01-03", "2020-01-06"])


class FxRepair(unittest.TestCase):
    def setUp(self):
        dates = pd.bdate_range("2000-11-01", "2008-06-30")
        self.cross = repair_fx.cross_rate(pd.Series(32.0, index=dates),
                                          pd.Series(np.linspace(40.0, 30.0, len(dates)), index=dates))

    def test_bad_thb_quotes_are_replaced_and_good_ones_kept(self):
        quotes = self.cross.copy()
        quotes["2001-03-01"] *= 1.05                        # alternating episode
        quotes["2007-11-05":"2008-02-29"] = quotes["2007-11-02"]  # stale stretch
        repaired, changes = repair_fx.repair_thb(quotes, self.cross)
        np.testing.assert_allclose(repaired.to_numpy(), self.cross.to_numpy(), rtol=1e-9)
        dates = [change["date"] for change in changes]
        self.assertEqual(dates[0], "2001-03-01")
        self.assertEqual(len(dates), 1 + len(self.cross["2007-11-05":"2008-02-29"]))

    def test_cny_gap_is_filled_from_the_cross(self):
        quotes = self.cross.drop(self.cross[repair_fx.CNY_GAP[0]:repair_fx.CNY_GAP[1]].index)
        filled, changes = repair_fx.fill_cny_gap(quotes, self.cross)
        np.testing.assert_allclose(filled.to_numpy(), self.cross.to_numpy(), rtol=1e-9)
        self.assertEqual(len(changes), len(self.cross[repair_fx.CNY_GAP[0]:repair_fx.CNY_GAP[1]]))
        with self.assertRaises(ValueError):
            repair_fx.fill_cny_gap(self.cross, self.cross)


class Significance(unittest.TestCase):
    def test_paired_dm_and_friedman_on_written_runs(self):
        rng = np.random.default_rng(1)
        true = rng.normal(size=(120, 3, 1))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for d, dataset in enumerate(["A", "B", "C"]):
                for seed in (42, 123, 456):
                    for model, scale, extra in (("TSBooster-vanilla", 1.0, {}), ("TSBooster", 0.6, {}),
                                                ("TSBooster", 0.9, {"tsb_backbone": "lstm"})):
                        noise = rng.normal(scale=scale, size=true.shape)
                        name = f"{dataset}-{model}-{extra.get('tsb_backbone', 'linear')}-{seed}"
                        write_run(root, name, model, dataset, seed, float(np.sqrt(np.mean(noise ** 2))) + d,
                                  true + noise, true, tsb_backbone=extra.get("tsb_backbone", "linear"))
            out = root / "paired.json"
            self.assertEqual(significance.main(["--results", directory, "--output", str(out), "paired",
                                                "--baseline", "TSBooster-vanilla",
                                                "--candidate", "TSBooster:tsb_backbone=linear"]), 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["overall"]["wins"], 3)
            self.assertTrue(all(row["mean_gap_pct"] > 0 for row in report["per_dataset"]))
            self.assertEqual(significance.main(["--results", directory, "--output", str(out), "dm",
                                                "--baseline", "TSBooster-vanilla",
                                                "--candidate", "TSBooster:tsb_backbone=linear", "--seed", "42"]), 0)
            dm = json.loads(out.read_text())["results"]
            self.assertTrue(all(row["dm_hln"] > 0 and row["horizon"] == 3 for row in dm))
            self.assertEqual(significance.main(["--results", directory, "--output", str(out), "friedman",
                                                "--methods", "TSBooster-vanilla", "TSBooster:tsb_backbone=linear",
                                                "TSBooster:tsb_backbone=lstm", "--plot", str(root / "cd.png")]), 0)
            ranks = json.loads(out.read_text())["average_ranks"]
            self.assertEqual(ranks["TSBooster:tsb_backbone=linear"], 1.0)
            self.assertTrue((root / "cd.png").is_file())

    def test_holm_stops_at_the_first_acceptance(self):
        self.assertEqual(significance.holm_bonferroni([0.001, 0.04, 0.012, 0.3]), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
