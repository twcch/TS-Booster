"""Model contracts, without provider downloads or optional model installations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from layers.channel_adapter import flatten_channels, restore_channels
from layers.foundation_adapter import FoundationRuntime, timesfm_compatible_class
from layers.SelfAttention_Family import AttentionLayer, FullAttention, ReformerLayer
from models.capabilities import MODEL_CAPABILITIES, validate_model_task
from layers.Embed import TimeFeatureEmbedding, TemporalEmbedding, time_feature_dimension


ROOT = Path(__file__).resolve().parents[1]


def load_model(relative):
    path = ROOT / "models" / relative
    spec = importlib.util.spec_from_file_location("contract_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(**changes):
    values = dict(task_name="long_term_forecast", seq_len=8, pred_len=3, label_len=4,
                  enc_in=3, dec_in=3, c_out=3, d_model=8, n_heads=2, d_ff=16,
                  dropout=0.0, embed="timeF", freq="h", factor=3, activation="gelu",
                  e_layers=1, d_layers=1, moving_avg=3, target_channel_index=-1,
                  runtime_device="cpu", runtime_dtype="auto", model_revision="test-revision",
                  model_cache_dir="/tmp/tsbooster-provider-test", local_files_only=True)
    values.update(changes)
    return SimpleNamespace(**values)


class ChannelContracts(unittest.TestCase):
    def test_channel_order_and_restore_multiple_batches(self):
        x = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
        flattened = flatten_channels(x)
        for batch in range(2):
            for channel in range(3):
                torch.testing.assert_close(flattened[batch * 3 + channel], x[batch, :, channel])
        restored = restore_channels(flattened[:, -2:], 2, 3, 2)
        torch.testing.assert_close(restored, x[:, -2:, :])

    def test_single_channel_and_invalid_provider_shapes(self):
        x = torch.randn(2, 5, 1)
        torch.testing.assert_close(restore_channels(flatten_channels(x), 2, 1, 5), x)
        with self.assertRaises(ValueError):
            restore_channels(torch.zeros(5, 2), 2, 3)
        with self.assertRaises(ValueError):
            restore_channels(torch.zeros(6, 2), 2, 3, 3)


class CapabilityContracts(unittest.TestCase):
    def test_all_model_files_have_explicit_task_metadata(self):
        paths = [p.stem for p in (ROOT / "models").rglob("*.py")
                 if p.name != "capabilities.py" and not p.name.startswith("_")]
        self.assertEqual(set(paths), set(MODEL_CAPABILITIES))
        self.assertEqual(len(paths), 52)
        validate_model_task("KANAD", "anomaly_detection")
        validate_model_task("MambaSingleLayer", "classification")
        for name, task in (("KANAD", "long_term_forecast"), ("TiDE", "classification"),
                           ("WPMixer", "imputation"), ("TSBooster", "classification"),
                           ("Chronos", "long_term_forecast")):
            with self.assertRaises(ValueError):
                validate_model_task(name, task)

    def test_optional_reformer_is_loaded_only_when_requested(self):
        layer = AttentionLayer(FullAttention(False, attention_dropout=0), 8, 2)
        x = torch.randn(2, 3, 8)
        self.assertEqual(layer(x, x, x, None)[0].shape, x.shape)
        with patch.dict("sys.modules", {"reformer_pytorch": None}):
            with self.assertRaisesRegex(ImportError, "reformer-pytorch"):
                ReformerLayer(None, 8, 2)

    def test_new_model_file_can_declare_or_infer_tasks_without_manual_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("models.capabilities.MODEL_ROOT", root):
                (root / "FutureModel.py").write_text(
                    'SUPPORTED_TASKS = ("classification",)\nclass Model: pass\n')
                validate_model_task("FutureModel", "classification")
                with self.assertRaises(ValueError):
                    validate_model_task("FutureModel", "long_term_forecast")
                (root / "NewDispatcher.py").write_text(
                    'class Model:\n'
                    '    def forward(self, x):\n'
                    '        if self.task_name == "long_term_forecast": return x\n'
                    '        if self.task_name == "imputation": raise NotImplementedError\n')
                validate_model_task("NewDispatcher", "long_term_forecast")
                with self.assertRaises(ValueError):
                    validate_model_task("NewDispatcher", "imputation")
                (root / "Undeclared.py").write_text('class Model:\n    def forward(self, x): return x\n')
                with self.assertRaisesRegex(ValueError, "declare literal"):
                    validate_model_task("Undeclared", "long_term_forecast")

    def test_time_embedding_uses_the_same_frequency_dimension_as_data(self):
        dimensions = {"h": 4, "d": 3, "b": 3, "min": 5, "15min": 5,
                      "m": 1, "MS": 1, "QS": 1, "y": 0, "a": 0, "YE": 0, "YS": 0}
        for freq, dimension in dimensions.items():
            with self.subTest(freq=freq):
                self.assertEqual(time_feature_dimension(freq), dimension)
                embedding = TimeFeatureEmbedding(8, freq=freq)
                self.assertEqual(embedding(torch.zeros(2, 3, dimension)).shape, (2, 3, 8))
        self.assertTrue(hasattr(TemporalEmbedding(8, freq="min"), "minute_embed"))
        self.assertTrue(hasattr(TemporalEmbedding(8, freq="15min"), "minute_embed"))


class FoundationContracts(unittest.TestCase):
    def test_runtime_and_local_timesfm_compatibility_do_not_mutate_provider(self):
        runtime = FoundationRuntime.from_config(config())
        self.assertEqual(runtime.device, torch.device("cpu"))
        self.assertEqual(runtime.dtype, torch.float32)
        self.assertEqual(runtime.hub_kwargs()["revision"], "test-revision")
        self.assertTrue(runtime.hub_kwargs()["local_files_only"])
        with self.assertRaises(ValueError):
            FoundationRuntime.from_config(config(runtime_dtype="float16"), fp32_only=True)

        class OldTimesFM:
            def __init__(self, torch_compile=True, config=None):
                self.config = config

        original = OldTimesFM.__init__
        adapter = timesfm_compatible_class(OldTimesFM)
        self.assertIs(OldTimesFM.__init__, original)
        self.assertEqual(adapter(config={"x": 1}, proxies=None).config, {"x": 1})
        with self.assertRaises(TypeError):
            adapter(unknown_argument=True)

    def test_univariate_foundation_wrappers_preserve_channels_and_runtime(self):
        x = torch.arange(2 * 8 * 3, dtype=torch.float32).reshape(2, 8, 3)
        for name in ("TimeMoE", "TimesFM", "TiRex"):
            module = load_model("tslib_models/" + name + ".py")
            calls = {}

            class ProviderModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = nn.Linear(1, 1)

                def generate(self, inputs, max_new_tokens):
                    calls["inputs"] = inputs
                    return torch.cat((inputs, inputs[:, -1:].expand(-1, max_new_tokens)), dim=1)

                def forecast(self, inputs=None, horizon=None, prediction_length=None):
                    calls["inputs"] = torch.as_tensor(inputs)
                    length = horizon or prediction_length
                    points = torch.as_tensor(inputs)[:, -1:].expand(-1, length)
                    return (points.numpy(), None) if name == "TimesFM" else (None, points)

                def compile(self, cfg):
                    calls["compile"] = cfg

            instance = ProviderModel()

            def pretrained(model_id, **kwargs):
                calls["kwargs"] = kwargs
                return instance

            if name == "TimeMoE":
                provider = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=pretrained))
            elif name == "TiRex":
                provider = SimpleNamespace(load_model=pretrained)
            else:
                class TimesFM(ProviderModel):
                    @classmethod
                    def from_pretrained(cls, model_id, **kwargs):
                        return pretrained(model_id, **kwargs)
                provider = SimpleNamespace(TimesFM_2p5_200M_torch=TimesFM, ForecastConfig=lambda **kwargs: kwargs)
            with patch.object(module, "require_provider", return_value=provider):
                model = module.Model(config(task_name="zero_shot_forecast"))
                output = model(x, None, None, None)
            self.assertEqual(output.shape, (2, 3, 3))
            torch.testing.assert_close(output, x[:, -1:, :].expand(-1, 3, -1))
            self.assertEqual(calls["inputs"].shape, (6, 8))
            kwargs = calls["kwargs"].get("hf_kwargs", calls["kwargs"])
            self.assertEqual(kwargs["revision"], "test-revision")
            self.assertTrue(kwargs["local_files_only"])
            self.assertEqual(model.effective_config["runtime_device"], "cpu")

    def test_moirai_retained_target_axis_has_three_dimensional_output(self):
        module = load_model("tslib_models/Moirai.py")

        class Forecast(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def predict(self, inputs):
                points = inputs[:, -1:, None]
                return np.repeat(np.repeat(points[:, None, ...], 4, axis=1), 3, axis=2)

        provider = SimpleNamespace(Moirai2Forecast=Forecast,
            Moirai2Module=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()))
        x = torch.randn(2, 8, 3)
        with patch.object(module, "require_provider", return_value=provider):
            model = module.Model(config(task_name="zero_shot_forecast"))
            output = model(x, None, None, None)
        self.assertEqual(output.shape, (2, 3, 3))
        torch.testing.assert_close(output, x[:, -1:, :].expand(-1, 3, -1))

    def test_chronos_and_sundial_provider_group_shapes_and_options(self):
        x = torch.randn(2, 8, 3)
        for name in ("Chronos", "Chronos2", "Sundial"):
            module = load_model("tslib_models/" + name + ".py")
            calls = {}

            class Pipeline(nn.Module):
                def predict(self, inputs, prediction_length):
                    calls["shape"] = inputs.shape
                    return inputs[:, -1:, None].expand(-1, 4, prediction_length)

                def generate(self, inputs, max_new_tokens, num_samples):
                    calls["num_samples"] = num_samples
                    return inputs[:, -1:, None].expand(-1, num_samples, max_new_tokens)

                def predict_quantiles(self, inputs, prediction_length, quantile_levels):
                    calls["shape"] = inputs.shape
                    points = torch.as_tensor(inputs)[:, :, -1:].expand(-1, -1, prediction_length)
                    return None, list(points.unbind(0))

            def pretrained(model_id, **kwargs):
                calls["kwargs"] = kwargs
                return Pipeline()

            factory = SimpleNamespace(from_pretrained=pretrained)
            provider = SimpleNamespace(BaseChronosPipeline=factory, AutoModelForCausalLM=factory)
            with patch.object(module, "require_provider", return_value=provider):
                model = module.Model(config(task_name="zero_shot_forecast"))
                output = model(x, None, None, None)
            self.assertEqual(output.shape, (2, 3, 3))
            torch.testing.assert_close(output, x[:, -1:, :].expand(-1, 3, -1))
            self.assertEqual(calls["kwargs"]["torch_dtype"], torch.float32)
            self.assertEqual(calls["kwargs"]["revision"], "test-revision")
            if name in ("Chronos", "Chronos2"):
                self.assertEqual(calls["kwargs"]["device_map"], "cpu")
            if name == "Chronos2":
                self.assertEqual(calls["shape"], (2, 3, 8))
            if name == "Sundial":
                self.assertEqual(calls["num_samples"], 20)

    def test_provider_dependency_error_is_actionable(self):
        from layers.foundation_adapter import require_provider
        with patch.dict("sys.modules", {"chronos": None}):
            with self.assertRaisesRegex(ImportError, "optional dependency group"):
                require_provider("chronos")

    def test_invalid_task_fails_before_provider_load(self):
        for name in ("Chronos", "Chronos2", "TimesFM", "TimeMoE", "Moirai", "Sundial", "TiRex"):
            module = load_model("tslib_models/" + name + ".py")
            with patch.object(module, "require_provider") as provider:
                with self.assertRaises(ValueError):
                    module.Model(config())
                provider.assert_not_called()

    def test_zero_shot_experiment_preserves_provider_bfloat16_parameters(self):
        from importlib import import_module
        from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast

        module = import_module("models.tslib_models.TimeMoE")
        input_dtypes = []

        class Provider(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)

            def generate(self, inputs, max_new_tokens):
                input_dtypes.append(inputs.dtype)
                return self.projection(inputs[:, -1:]).expand(-1, max_new_tokens)

        factory = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Provider())
        cfg = config(task_name="zero_shot_forecast", model="TimeMoE", use_gpu=False,
                     use_multi_gpu=False, runtime_dtype="bfloat16")
        with patch.object(module, "require_provider", return_value=SimpleNamespace(AutoModelForCausalLM=factory)):
            experiment = Exp_Zero_Shot_Forecast(cfg)
            output = experiment.model(torch.randn(2, 8, 3), None, None, None)
        self.assertEqual(experiment.model.model.projection.weight.dtype, torch.bfloat16)
        self.assertEqual(input_dtypes, [torch.bfloat16])
        self.assertEqual(output.shape, (2, 3, 3))
        self.assertTrue(torch.isfinite(output).all())


class BenchmarkContracts(unittest.TestCase):
    def test_existing_output_is_preserved_without_starting_measurement(self):
        from tools import benchmark

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            output.write_text("existing measurement", encoding="utf-8")
            with patch.object(benchmark, "benchmark_models") as measure:
                with self.assertRaisesRegex(FileExistsError, "select a new --output path"):
                    benchmark.main(["--output", str(output), "--models-only"])
                measure.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "existing measurement")

    def test_model_benchmark_runs_on_the_requested_device(self):
        from tools import benchmark

        args = benchmark.parser().parse_args(["--device", "cpu", "--models", "DLinear-paper", "TSBooster:linear",
                                              "--rounds", "1", "--forwards", "2", "--warmup", "1"])
        report = benchmark.benchmark_models(args)
        self.assertEqual(report["configuration"]["device"], "cpu")
        self.assertEqual(set(report["models"]), {"DLinear-paper", "TSBooster:linear"})
        self.assertEqual(report["models"]["DLinear-paper"]["params"], 372)
        self.assertEqual(benchmark.parser().parse_args(["--device", "cuda"]).device, "cuda")

    def test_loader_fully_consumes_the_bounded_dataset(self):
        from tools import benchmark

        args = benchmark.parser().parse_args(["--loader-source", "memory", "--memory-samples", "33",
            "--seq-len", "4", "--pred-len", "2", "--loader-batch-size", "4", "--loader-batches", "3",
            "--rounds", "1", "--workers", "0"])
        report = benchmark.benchmark_loader(args)
        self.assertEqual(report["source"]["windows"], 33)
        self.assertEqual(report["source"]["benchmark_windows"], 12)
        row = report["measurements"]["0"]["rounds"][0]
        self.assertEqual((row["batches"], row["samples"]), (3, 12))
        self.assertTrue(report["data_order_parity"])


if __name__ == "__main__":
    unittest.main()
