"""TS-Booster port: parameter counts, parity with the journal code, loss and metric contracts."""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from utils.forecast_losses import HybridDirectionalLoss, build_forecast_criterion
from utils.metrics import MDA, metric_report

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "docs" / "TS-Booster_ori"
MODEL_FILES = {
    "TSBooster": "TSBooster.py",
    "TSBooster-vanilla": "ablation_variant/TSBooster-vanilla.py",
    "TSBooster-woexpert": "ablation_variant/TSBooster-woexpert.py",
    "TSBooster-wodecomp": "ablation_variant/TSBooster-wodecomp.py",
    "TSBooster-nogate": "gate_variant/TSBooster-nogate.py",
    "TSBooster-constgate": "gate_variant/TSBooster-constgate.py",
    "TSBooster-learnedgate": "gate_variant/TSBooster-learnedgate.py",
    "MLP7K": "baselines/MLP7K.py",
    "DLinear-paper": "baselines/DLinear-paper.py",
    "PatchTST-paper": "baselines/PatchTST-paper.py",
    "iTransformer-paper": "baselines/iTransformer-paper.py",
}
# Paper Table 8 / compute_ext_results.json at L=30, H=3, C=2: (host alone, full TS-Booster).
PAPER_PARAMS = {"linear": (366, 7156), "lstm": (50883, 57673), "gru": (38211, 45001),
                "transformer": (67331, 74121)}


def load_model(name):
    path = ROOT / "models/my_models/tsbooster" / MODEL_FILES[name]
    spec = importlib.util.spec_from_file_location("tsb_" + name.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


def config(**overrides):
    values = dict(task_name="long_term_forecast", seq_len=30, pred_len=3, enc_in=2, c_out=1, features="MS",
                  target_channel_index=-1, d_model=64, e_layers=2, n_heads=4, d_ff=128, dropout=0.3,
                  tsb_backbone="linear", tsb_decomp_kernels=[7, 13], tsb_expert_kernels=[3, 5, 7],
                  tsb_expert_hidden=32, tsb_n_experts=2, tsb_expert_type="cnn", tsb_enrich="trend",
                  tsb_aux_weight=0.3, tsb_base_weight=0.1)
    values.update(overrides)
    return SimpleNamespace(**values)


def count(model):
    return sum(p.numel() for p in model.parameters())


class ArchitectureContracts(unittest.TestCase):
    def test_parameter_counts_match_the_paper(self):
        for backbone, (host, full) in PAPER_PARAMS.items():
            with self.subTest(backbone=backbone):
                self.assertEqual(count(load_model("TSBooster-vanilla")(config(tsb_backbone=backbone))), host)
                self.assertEqual(count(load_model("TSBooster")(config(tsb_backbone=backbone))), full)
        expected = {"TSBooster-woexpert": 734, "TSBooster-wodecomp": 6552, "TSBooster-nogate": 7154,
                    "TSBooster-constgate": 7154, "TSBooster-learnedgate": 7155, "MLP7K": 7171,
                    # Table 8: the paper's deep comparators
                    "DLinear-paper": 372, "PatchTST-paper": 402823, "iTransformer-paper": 269571}
        for name, value in expected.items():
            with self.subTest(model=name):
                self.assertEqual(count(load_model(name)(config())), value)

    def test_target_channel_is_moved_first_and_output_is_single_target(self):
        model = load_model("TSBooster")(config(enc_in=3)).eval()
        self.assertEqual(model.effective_config["internal_channel_order"], [2, 0, 1])
        x = torch.randn(4, 30, 3)
        with torch.no_grad():
            prediction, aux = model(x, None, None, None, return_aux=True)
        self.assertEqual(tuple(prediction.shape), (4, 3, 1))
        self.assertEqual(tuple(aux["gate"].shape), (4,))
        self.assertTrue(torch.all((aux["gate"] > 0) & (aux["gate"] < 1)))

    def test_invalid_configurations_fail_before_training(self):
        cases = [config(c_out=2), config(features="M"), config(tsb_backbone="transformer", n_heads=5),
                 config(tsb_decomp_kernels=[6, 13]), config(task_name="imputation"),
                 config(target_channel_index=2)]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                load_model("TSBooster")(case)

    def test_effective_config_records_fixed_and_ignored_settings(self):
        linear = load_model("TSBooster")(config()).effective_config
        self.assertEqual(linear["host"], {"type": "linear"})
        self.assertIn("d_model", linear["ignored_config_parameters"])
        self.assertEqual(linear["expert_branch_dim"], 10)
        lstm = load_model("TSBooster")(config(tsb_backbone="lstm")).effective_config
        self.assertEqual(lstm["host"]["dropout"], 0.2)
        vanilla = load_model("TSBooster-vanilla")(config()).effective_config
        self.assertIsNone(vanilla["auxiliary_loss"])
        self.assertIn("dropout", vanilla["ignored_config_parameters"])
        patch = load_model("PatchTST-paper")(config()).effective_config
        self.assertEqual((patch["n_patches"], patch["dropout"], patch["revin"]), (9, 0.1, True))
        self.assertIn("dropout", patch["ignored_config_parameters"])
        with self.assertRaises(ValueError):
            load_model("PatchTST-paper")(config(seq_len=4))

    def test_auxiliary_loss_exists_only_with_experts(self):
        x, target = torch.randn(8, 30, 2), torch.randn(8, 3, 1)
        for name, expected in (("TSBooster", True), ("TSBooster-wodecomp", True),
                               ("TSBooster-vanilla", False), ("TSBooster-woexpert", False)):
            with self.subTest(model=name):
                model = load_model(name)(config())
                self.assertEqual(model.has_auxiliary_loss, expected)
                _, aux = model(x, None, None, None, return_aux=True)
                self.assertEqual(model.auxiliary_loss(aux, target) is not None, expected)


class HybridLossAndDirectionalAccuracy(unittest.TestCase):
    def test_hybrid_matches_the_formula_and_ignores_small_changes(self):
        pred = torch.tensor([[[1.0], [2.0]], [[0.0], [0.5]]])
        target = torch.tensor([[[1.5], [1.005]], [[-1.0], [0.2]]])
        last = torch.tensor([[[1.0]], [[0.0]]])
        loss = HybridDirectionalLoss(delta=1.0, dir_weight=0.5, threshold=0.01)(pred, target, last)
        huber = torch.nn.functional.huber_loss(pred, target, delta=1.0)
        mask = (target - last).abs() > 0.01
        direction = torch.nn.functional.soft_margin_loss((pred - last)[mask], torch.sign((target - last)[mask]))
        self.assertTrue(torch.allclose(loss, 0.5 * huber + 0.5 * direction))
        self.assertEqual(int(mask.sum()), 3)

    def test_criterion_names(self):
        for name, kind in (("MSE", torch.nn.MSELoss), ("MAE", torch.nn.L1Loss), ("HUBER", torch.nn.HuberLoss),
                           ("hybrid", HybridDirectionalLoss)):
            self.assertIsInstance(build_forecast_criterion(SimpleNamespace(loss=name)), kind)
        with self.assertRaises(ValueError):
            build_forecast_criterion(SimpleNamespace(loss="SMAPE"))

    def test_mda_counts_exact_signs_against_the_last_value(self):
        pred = np.array([[1.0, 2.0, 0.0], [3.0, 3.0, 1.0]])
        true = np.array([[2.0, 0.0, 1.0], [4.0, 2.0, 3.0]])
        anchor = np.array([1.0, 3.0])
        # signs: row0 pred (0,+,-) true (+,-,0); row1 pred (0,0,-) true (+,-,0)
        self.assertEqual(MDA(pred, true, anchor), 0.0)
        self.assertEqual(MDA(pred[..., None], pred[..., None], anchor[:, None]), 1.0)
        report = metric_report(true[..., None], true[..., None] + 0.5, anchor[:, None])
        self.assertIn("mda", report["overall"])
        self.assertEqual(len(report["per_horizon"]), 3)


@unittest.skipUnless((ORIGINAL / "src/models/flexible_booster.py").is_file(), "journal code not present")
class JournalParity(unittest.TestCase):
    """Same seed, same weights and outputs as docs/TS-Booster_ori (target first there, last here)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ORIGINAL))
        cls.layers = importlib.import_module("src.models.layers")
        cls.flexible = importlib.import_module("src.models.flexible_booster").FlexibleTSBooster
        cls.metrics = importlib.import_module("src.utils.metrics")
        cls.sota = importlib.import_module("src.models.sota_baselines")

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ORIGINAL))
        for name in [n for n in sys.modules if n == "src" or n.startswith("src.")]:
            del sys.modules[name]

    def original(self, variant, backbone):
        layers, flexible = self.layers, self.flexible
        hosts = {"linear": lambda: layers.LinearBackbone(30, 2, 3),
                 "lstm": lambda: layers.LSTMBackbone(30, 2, 3, hidden_dim=64, num_layers=2),
                 "gru": lambda: layers.GRUBackbone(30, 2, 3, hidden_dim=64, num_layers=2),
                 "transformer": lambda: layers.TransformerBackbone(30, 2, 3, hidden_dim=64, num_layers=2)}
        kwargs = dict(seq_len=30, pred_len=3, input_channels=2, decomp_kernels=[7, 13], expert_kernels=[3, 5, 7],
                      expert_hidden_dim=32, dropout_p=0.3, n_boost_rounds=2, expert_type="cnn")
        flags = {"TSBooster": dict(use_decomp=True, use_experts=True, expert_gate="snr", enrich_mode="trend"),
                 "TSBooster-vanilla": dict(use_decomp=False, use_experts=False),
                 "TSBooster-woexpert": dict(use_decomp=True, use_experts=False),
                 "TSBooster-wodecomp": dict(use_decomp=False, use_experts=True),
                 "TSBooster-nogate": dict(use_decomp=True, use_experts=True, expert_gate=None, enrich_mode="trend"),
                 "TSBooster-learnedgate": dict(use_decomp=True, use_experts=True, expert_gate="learned",
                                               enrich_mode="trend"),
                 "TSBooster-constgate": dict(use_decomp=True, use_experts=True, expert_gate=None,
                                             enrich_mode="trend")}[variant]

        class ConstGateBooster(flexible):  # revision/phase6_ablation_ext/snr_gate_ablation.py
            def _compute_expert_gate(self, x, seasonal_part):
                return torch.tensor(0.5, device=x.device)

        cls = ConstGateBooster if variant == "TSBooster-constgate" else flexible
        return cls(base_model=hosts[backbone](), **kwargs, **flags)

    def pair(self, variant, backbone, seed=42):
        torch.manual_seed(seed)
        original = self.original(variant, backbone)
        torch.manual_seed(seed)
        port = load_model(variant)(config(tsb_backbone=backbone))
        return original, port

    def assert_same_weights(self, original, port, prefix="booster."):
        theirs = original.state_dict()
        ours = {key.removeprefix(prefix): value for key, value in port.state_dict().items()}
        self.assertEqual(list(ours), list(theirs))
        for key in theirs:
            self.assertTrue(torch.equal(ours[key], theirs[key]), key)

    def test_paper_comparators_match_sota_baselines(self):
        x = torch.randn(16, 30, 2, generator=torch.Generator().manual_seed(0))
        sota = self.sota
        originals = {"DLinear-paper": lambda: sota.DLinear(30, 3, 2),
                     # revision/phase10_lr_tuning (the paper's Table 6) builds PatchTST with RevIN
                     "PatchTST-paper": lambda: sota.PatchTST(30, 3, 2, use_revin=True),
                     "iTransformer-paper": lambda: sota.iTransformer(30, 3, 2)}
        for name, build in originals.items():
            with self.subTest(model=name):
                torch.manual_seed(42)
                original = build()
                torch.manual_seed(42)
                port = load_model(name)(config())
                self.assert_same_weights(original, port, prefix="net.")
                for training in (False, True):
                    original.train(training), port.train(training)
                    torch.manual_seed(7)
                    expected = original(x)[0]
                    torch.manual_seed(7)
                    actual = port(x[..., [1, 0]], None, None, None)[..., 0]
                    self.assertTrue(torch.equal(actual, expected))

    def test_weights_and_outputs_match_for_all_variants_and_hosts(self):
        x = torch.randn(16, 30, 2, generator=torch.Generator().manual_seed(0))
        x_tslib = x[..., [1, 0]]  # TSLib MS order: [log_return, Volatility]
        for variant in ("TSBooster", "TSBooster-vanilla", "TSBooster-woexpert", "TSBooster-wodecomp",
                        "TSBooster-nogate", "TSBooster-learnedgate", "TSBooster-constgate"):
            for backbone in PAPER_PARAMS:
                with self.subTest(variant=variant, backbone=backbone):
                    original, port = self.pair(variant, backbone)
                    self.assert_same_weights(original, port)
                    original.eval(), port.eval()
                    with torch.no_grad():
                        expected = original(x)[0]
                        actual = port(x_tslib, None, None, None)[..., 0]
                    self.assertTrue(torch.equal(actual, expected))
                    original.train(), port.train()
                    torch.manual_seed(7)
                    expected = original(x)[0]
                    torch.manual_seed(7)
                    actual = port(x_tslib, None, None, None)[..., 0]
                    self.assertTrue(torch.equal(actual, expected))

    def test_training_loss_and_gradients_match_train_v11(self):
        # Forwards are bit-identical (test above). The TSLib loss runs on [B, H, 1] tensors,
        # so reduction order may differ from the [B, H] original at float32 rounding level.
        x = torch.randn(32, 30, 2, generator=torch.Generator().manual_seed(1))
        y = torch.randn(32, 3, generator=torch.Generator().manual_seed(2))
        for backbone in PAPER_PARAMS:
            with self.subTest(backbone=backbone):
                original, port = self.pair("TSBooster", backbone, seed=123)
                torch.manual_seed(3)
                output = original(x)
                criterion = self.metrics.HybridDirectionalLoss(dir_weight=0.5)
                expected = (criterion(output[0], y, x[:, -1, 0:1])
                            + 0.3 * torch.nn.functional.mse_loss(output[5], y - output[1].detach())
                            + 0.1 * torch.nn.functional.mse_loss(output[1], y))
                expected.backward()
                torch.manual_seed(3)
                x_tslib = x[..., [1, 0]]
                prediction, aux = port(x_tslib, None, None, None, return_aux=True)
                target = y[..., None]
                actual = (HybridDirectionalLoss(dir_weight=0.5)(prediction, target, x_tslib[:, -1:, -1:])
                          + port.auxiliary_loss(aux, target))
                actual.backward()
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                grads = {name: p.grad for name, p in original.named_parameters()}
                self.assertEqual(len(grads), len(list(port.parameters())))
                for name, p in port.named_parameters():
                    torch.testing.assert_close(p.grad, grads[name.removeprefix("booster.")], rtol=1e-5, atol=1e-6,
                                               msg=name)

    def test_mda_matches_the_evaluate_formula(self):
        rng = np.random.default_rng(0)
        preds, targets, last = rng.normal(size=(50, 3)), rng.normal(size=(50, 3)), rng.normal(size=50)
        expected = float(np.mean(np.sign(preds - last[:, None]) == np.sign(targets - last[:, None])))
        self.assertEqual(MDA(preds[..., None], targets[..., None], last[:, None]), expected)


class EndToEnd(unittest.TestCase):
    def test_cli_trains_tsbooster_with_the_paper_protocol(self):
        from run import main
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            index = np.arange(140)
            returns = np.sin(index / 5) / 50 + np.cos(index / 3) / 80
            volatility = pd.Series(returns).rolling(7, min_periods=1).std().fillna(0).to_numpy() * 1000 + 5
            pd.DataFrame({"date": pd.date_range("2020-01-01", periods=140, freq="B"), "log_return": returns,
                          "Volatility": volatility}).to_csv(directory / "vol.csv", index=False)
            argv = ["--task_name", "long_term_forecast", "--is_training", "1", "--model_id", "e2e",
                    "--model", "TSBooster", "--data", "custom", "--root_path", str(directory),
                    "--data_path", "vol.csv", "--features", "MS", "--target", "Volatility", "--freq", "b",
                    "--enc_in", "2", "--dec_in", "2", "--c_out", "1", "--seq_len", "16", "--label_len", "0",
                    "--pred_len", "3", "--dropout", "0.3", "--train_epochs", "2", "--patience", "30",
                    "--batch_size", "16", "--learning_rate", "0.001", "--optimizer", "adamw",
                    "--weight_decay", "1e-4", "--lradj", "plateau", "--min_lr", "1e-5", "--clip_grad", "5",
                    "--loss", "HYBRID", "--train_drop_last", "--inverse", "--no_plots", "--no_use_gpu",
                    "--rand_seed", "42", "--checkpoints", str(directory / "weights"),
                    "--results_root", str(directory / "results"),
                    "--test_results_root", str(directory / "plots")]
            self.assertEqual(main(argv), 0)
            run, = (directory / "results").iterdir()
            result = json.loads((run / "result.json").read_text())
            summary = result["summary"]
            for key in ("mse", "rmse", "mae", "mape", "r2", "mda", "train_time_s", "inference_time_s",
                        "params", "trainable_params", "model_size_mb", "epochs_trained", "best_epoch"):
                self.assertIsNotNone(summary.get(key), key)
            # L=16: Linear host 198 + 2 experts x 2,133 + 2 decomposition logits + 2 gate scalars.
            self.assertEqual(summary["params"], 4468)
            self.assertEqual(summary["metric_space"], "original")
            self.assertEqual(set(result["artifacts"]) >= {"last.npy", "gate.npy"}, True)
            self.assertEqual(np.load(run / "gate.npy").shape, (np.load(run / "pred.npy").shape[0],))
            effective = json.loads((run / "effective_config.json").read_text())
            self.assertEqual(effective["variant"], "TS-Booster")
            history = result["details"]["training_history"]
            self.assertEqual([record["epoch"] for record in history], [1, 2])
            self.assertIn("decomp_mixture", result["details"]["model_diagnostics"])


if __name__ == "__main__":
    unittest.main()
