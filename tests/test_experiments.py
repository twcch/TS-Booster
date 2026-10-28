"""Microfixtures exercise real task loops without weights or dataset downloads."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from exp.common.evaluation import metric_spaces
from exp.common.runtime import TrainResult
from exp.common.runtime import file_fingerprint
from exp.common.training import WeightedLoss, require_same_shape
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation, imputation_mask
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast, m4_loss
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from utils.losses import smape_loss
from utils.metrics import metric


def arguments(directory, task):
    return SimpleNamespace(task_name=task, model='toy', model_id='test', data='custom',
                           seq_len=4, label_len=2, pred_len=2, enc_in=2, features='M',
                           batch_size=2, train_epochs=1, patience=2, learning_rate=0.001,
                           use_amp=False, lradj='type1', no_plots=True, inverse=False,
                           metric_scale='both', diagnostic_test=False, rand_seed=123,
                           mask_rate=0.25, mask_eval_seed=None, anomaly_ratio=50.0,
                           anomaly_protocol='calibration', seasonal_patterns='Yearly',
                           frequency_map=1, loss='MSE', use_dtw=False, checkpoint_run=None,
                           checkpoints=str(Path(directory) / 'checkpoints'),
                           results_root=str(Path(directory) / 'results'),
                           test_results_root=str(Path(directory) / 'plots'))


class ForecastModel(nn.Module):
    def __init__(self, horizon=2):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))
        self.horizon = horizon
        self.seen = []

    def forward(self, x, *_):
        self.seen.append(x.detach().cpu().clone())
        return x[:, -1:, :].expand(-1, self.horizon, -1) + self.bias


class ReconstructionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.grad_modes = []

    def forward(self, x, *_):
        self.grad_modes.append(torch.is_grad_enabled())
        return torch.zeros_like(x) + self.bias


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x, *_):
        return self.linear(x.mean(dim=1))


class ScaledDataset:
    scale = True

    def inverse_transform(self, value, target_only=False):
        mean, scale = np.array([10.0, 100.0]), np.array([2.0, 4.0])
        return value * (scale[-1:] if target_only else scale) + (mean[-1:] if target_only else mean)


class M4Fixture:
    scale = False

    def __init__(self, context=None):
        self.context = np.ones((3, 4), dtype=np.float32) if context is None else context
        self.timeseries = [np.array([2.0, 3.0], dtype=np.float32) for _ in self.context]
        self.ids = [f'Y{i}' for i in range(len(self.context))]

    def last_insample_window(self):
        return self.context.copy(), np.ones_like(self.context)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def experiment(self, cls, task, model):
        experiment = cls.__new__(cls)
        experiment.args = arguments(self.temporary.name, task)
        experiment.device = torch.device('cpu')
        experiment.model = model
        return experiment

    def forecasting_loader(self, batch_size=2, count=5, channels=2):
        x = torch.arange(count * 4 * channels).reshape(count, 4, channels).float() / 10 + 1
        y = torch.cat((x[:, -2:], x[:, -1:].expand(-1, 2, -1) + 1), dim=1)
        dataset = TensorDataset(x, y, torch.zeros(count, 4, 3), torch.zeros(count, 4, 3))
        dataset.scale = False
        return dataset, DataLoader(dataset, batch_size=batch_size)

    def classification_loader(self, batch_size=2):
        x = torch.arange(40).reshape(5, 4, 2).float() / 10
        labels = torch.tensor([[0], [1], [1], [0], [1]])
        dataset = TensorDataset(x, labels, torch.ones(5, 4))
        return dataset, DataLoader(dataset, batch_size=batch_size)

    def reconstruction_loader(self, batch_size=2):
        x = torch.arange(40).reshape(5, 4, 2).float() / 10 + 1
        dataset = TensorDataset(x, torch.zeros(5, 4))
        return dataset, DataLoader(dataset, batch_size=batch_size)

    def attach(self, experiment, mappings):
        calls = []
        def get_data(flag):
            calls.append(flag)
            return mappings[flag]
        experiment._get_data = get_data
        return calls

    def test_weighted_mean_is_invariant_to_grouping(self):
        first, second = WeightedLoss(), WeightedLoss()
        first.update(0, 2)
        first.update(9, 1)
        second.update(0, 1)
        second.update(4.5, 2)
        self.assertEqual(first.mean(), 3)
        self.assertEqual(second.mean(), 3)
        with self.assertRaises(ValueError):
            WeightedLoss().mean()
        with self.assertRaises(ValueError):
            first.update(float('nan'), 1)

    def test_classification_validation_keeps_singleton_batch_and_weighting(self):
        experiment = self.experiment(Exp_Classification, 'classification', Classifier())
        dataset, loader = self.classification_loader(2)
        loss, accuracy = experiment.vali(dataset, loader, nn.CrossEntropyLoss())
        full = self.classification_loader(5)[1]
        full_loss, full_accuracy = experiment.vali(dataset, full, nn.CrossEntropyLoss())
        self.assertAlmostEqual(loss, full_loss, places=6)
        self.assertEqual(accuracy, full_accuracy)
        single = DataLoader(dataset, batch_size=1)
        self.assertAlmostEqual(loss, experiment.vali(dataset, single, nn.CrossEntropyLoss())[0], places=6)

    def test_classification_architecture_does_not_read_test(self):
        experiment = self.experiment(Exp_Classification, 'classification', Classifier())
        metadata = SimpleNamespace(max_seq_len=4, feature_df=np.empty((3, 2)), class_names=['a', 'b'])
        calls = self.attach(experiment, {'TRAIN': (metadata, None), 'VAL': (metadata, None)})
        with patch('exp.exp_basic.Exp_Basic._build_model', return_value=experiment.model):
            experiment._build_model()
        self.assertEqual(calls, ['TRAIN', 'VAL'])

    def test_forecast_loss_is_invariant_to_batch_size(self):
        experiment = self.experiment(Exp_Long_Term_Forecast, 'long_term_forecast', ForecastModel())
        reference = experiment.vali(*self.forecasting_loader(5), nn.MSELoss())
        for size in [1, 2, 3]:
            self.assertAlmostEqual(reference, experiment.vali(*self.forecasting_loader(size), nn.MSELoss()), places=5)

    def test_target_specific_inverse_and_masked_metrics(self):
        args = arguments(self.temporary.name, 'long_term_forecast')
        args.features, args.inverse = 'MS', True
        predictions = np.array([[[1.0], [2.0]]])
        targets = np.array([[[2.0], [3.0]]])
        prediction, target, result, details = metric_spaces(args, ScaledDataset(), predictions, targets)
        np.testing.assert_allclose(prediction, predictions * 4 + 100)
        self.assertEqual(result['mse'], 16.0)
        self.assertEqual(details['metrics_normalized']['mse'], 1.0)
        masked = np.array([[[0], [1]]])
        prediction, target, result, details = metric_spaces(args, ScaledDataset(), predictions, targets, mask=masked)
        self.assertEqual(result['mse'], 16.0)
        self.assertEqual(details['metric_report_original']['overall']['mse'], 16.0)

    def test_m4_mse_and_benchmark_mask_adapters(self):
        forecast = torch.tensor([[3.0, 100.0]])
        target = torch.tensor([[0.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0]])
        loss, count = m4_loss(nn.MSELoss(), torch.ones(1, 4), 1, forecast, target, mask)
        self.assertEqual((loss.item(), count), (9.0, 1))
        loss, count = m4_loss(smape_loss(), torch.ones(1, 4), 1, forecast, target, mask)
        self.assertEqual((loss.item(), count), (200.0, 1))
        with self.assertRaises(ValueError):
            m4_loss(nn.MSELoss(), torch.ones(1, 4), 1, forecast, target, torch.zeros_like(mask))

    def test_losses_reject_channel_broadcasting(self):
        with self.assertRaisesRegex(ValueError, 'broadcasting'):
            require_same_shape(torch.zeros(2, 3, 1), torch.zeros(2, 3, 2))

    def test_m4_validation_uses_own_holdout_context(self):
        experiment = self.experiment(Exp_Short_Term_Forecast, 'short_term_forecast', ForecastModel())
        validation = M4Fixture(np.full((3, 4), 7.0, dtype=np.float32))
        experiment.vali(SimpleNamespace(dataset=M4Fixture()), SimpleNamespace(dataset=validation), nn.MSELoss())
        self.assertTrue(all(torch.all(value == 7) for value in experiment.model.seen))

    def test_m4_single_series_plot_and_ids_survive_csv(self):
        experiment = self.experiment(Exp_Short_Term_Forecast, 'short_term_forecast', ForecastModel())
        experiment.args.no_plots = False
        fixture = M4Fixture(np.ones((1, 4), dtype=np.float32))
        calls = self.attach(experiment, {'test': (fixture, SimpleNamespace(dataset=fixture))})
        with patch('exp.exp_short_term_forecasting.visual') as plot:
            experiment.test('small-m4')
        self.assertEqual(plot.call_count, 1)
        self.assertEqual(calls, ['test'])
        csv = pd.read_csv(Path(experiment.args.results_root) / 'small-m4' / 'Yearly_forecast.csv')
        self.assertEqual(list(csv.columns), ['id', 'V1', 'V2'])
        self.assertEqual(csv.loc[0, 'id'], 'Y0')

    def test_imputation_masks_are_fixed_and_have_valid_context_targets(self):
        experiment = self.experiment(Exp_Imputation, 'imputation', ReconstructionModel())
        first = imputation_mask((5, 4, 2), 0.00001, experiment.device,
                                experiment._mask_generator('test'), target_only=True)
        torch.rand(100)
        second = imputation_mask((5, 4, 2), 0.00001, experiment.device,
                                 experiment._mask_generator('test'), target_only=True)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first.sum(dim=1) >= 1))
        self.assertTrue(torch.all((first[:, :, -1] == 0).sum(dim=1) >= 1))
        with self.assertRaises(ValueError):
            imputation_mask((1, 4, 2), 0, experiment.device)
        with self.assertRaises(ValueError):
            imputation_mask((1, 4, 2), 1, experiment.device)

    def test_imputation_artifact_recomputes_masked_metrics(self):
        experiment = self.experiment(Exp_Imputation, 'imputation', ReconstructionModel())
        dataset, loader = self.forecasting_loader()
        self.attach(experiment, {'test': (dataset, loader)})
        experiment.test('masked-1')
        torch.rand(200)
        experiment.test('masked-2')
        root = Path(experiment.args.results_root)
        first = np.load(root / 'masked-1' / 'mask.npy')
        second = np.load(root / 'masked-2' / 'mask.npy')
        np.testing.assert_array_equal(first, second)
        predictions = np.load(root / 'masked-1' / 'pred.npy')
        targets = np.load(root / 'masked-1' / 'true.npy')
        saved = np.load(root / 'masked-1' / 'metrics.npy')
        np.testing.assert_allclose(saved, metric(predictions[first == 0], targets[first == 0]), equal_nan=True)

    def anomaly_loader(self, values, labels=None):
        x = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1)
        labels = torch.zeros(1, x.size(1)) if labels is None else torch.tensor(labels).reshape(1, -1)
        data = TensorDataset(x, labels)
        return data, DataLoader(data, batch_size=1)

    def test_anomaly_threshold_is_independent_of_test_and_all_forward_is_no_grad(self):
        experiment = self.experiment(Exp_Anomaly_Detection, 'anomaly_detection', ReconstructionModel())
        val = self.anomaly_loader([1, 3])
        test = self.anomaly_loader([1, 3, 1], [1, 1, 1])
        calls = self.attach(experiment, {'val': val, 'test': test})
        result = experiment.test('anomaly-first')
        self.assertEqual(calls, ['val', 'test'])
        self.assertEqual(result['details']['threshold'], 5.0)
        np.testing.assert_array_equal(np.load(Path(experiment.args.results_root) / 'anomaly-first' / 'pred.npy'), [0, 1, 0])
        np.testing.assert_array_equal(np.load(Path(experiment.args.results_root) / 'anomaly-first' / 'point_adjusted_pred.npy'), [1, 1, 1])
        calls = self.attach(experiment, {'val': val, 'test': self.anomaly_loader([10, 100, 10], [1, 1, 1])})
        result = experiment.test('anomaly-second')
        self.assertEqual(result['details']['threshold'], 5.0)
        self.assertFalse(any(experiment.model.grad_modes))

    def test_anomaly_legacy_is_named_and_ms_rejected(self):
        experiment = self.experiment(Exp_Anomaly_Detection, 'anomaly_detection', ReconstructionModel())
        experiment.args.anomaly_protocol = 'legacy'
        calls = self.attach(experiment, {'train': self.anomaly_loader([1, 3]),
                                       'test': self.anomaly_loader([10, 100, 10], [1, 0, 1])})
        result = experiment.test('anomaly-legacy')
        self.assertEqual(calls, ['train', 'test'])
        self.assertEqual(result['details']['threshold_source'], 'train+test energies')
        experiment.args.features = 'MS'
        with self.assertRaisesRegex(ValueError, 'MS'):
            experiment.test('invalid-ms')

    def test_five_training_tasks_return_contract_and_never_read_test(self):
        cases = [
            (Exp_Long_Term_Forecast, 'long_term_forecast', ForecastModel(), self.forecasting_loader()),
            (Exp_Imputation, 'imputation', ReconstructionModel(), self.forecasting_loader()),
            (Exp_Classification, 'classification', Classifier(), self.classification_loader()),
            (Exp_Anomaly_Detection, 'anomaly_detection', ReconstructionModel(), self.reconstruction_loader()),
        ]
        m4 = M4Fixture()
        x = torch.ones(3, 4, 1)
        y = torch.tensor([1., 1., 2., 3.]).reshape(1, 4, 1).expand(3, -1, -1)
        m4_train = TensorDataset(x, y, torch.ones_like(x), torch.ones_like(y))
        cases.append((Exp_Short_Term_Forecast, 'short_term_forecast', ForecastModel(),
                      (m4, DataLoader(m4_train, batch_size=2))))
        for cls, task, model, train_pair in cases:
            with self.subTest(task=task):
                experiment = self.experiment(cls, task, model)
                train_flag, val_flag = ('TRAIN', 'VAL') if task == 'classification' else ('train', 'val')
                val_pair = (m4, SimpleNamespace(dataset=m4)) if task == 'short_term_forecast' else train_pair
                test_flag = 'TEST' if task == 'classification' else 'test'
                calls = self.attach(experiment, {train_flag: train_pair, val_flag: val_pair, test_flag: val_pair})
                result = experiment.train(task)
                self.assertIsInstance(result, TrainResult)
                self.assertIs(result.model, model)
                self.assertGreater(result.train_time_s, 0)
                self.assertEqual(calls, [train_flag, val_flag])
                self.assertEqual(result.best_epoch, 1)
                snapshot = torch.load(Path(experiment.args.checkpoints) / task / 'training_state.pth',
                                      map_location='cpu', weights_only=True)
                self.assertIn('optimizer', snapshot)
                self.assertIn('torch_cpu', snapshot['rng'])
                self.assertFalse(snapshot['resume_supported'])
                self.assertEqual(snapshot['model_checkpoint_sha256'],
                                 file_fingerprint(Path(experiment.args.checkpoints) / task / 'checkpoint.pth'))
                self.assertEqual(snapshot['scheduler']['enabled'], task != 'classification')
                evaluation = experiment.test(task, train_time=result.train_time_s)
                self.assertEqual(calls[:2], [train_flag, val_flag])
                self.assertEqual(evaluation['status'], 'complete')
                self.assertTrue((Path(experiment.args.results_root) / task / 'pred.npy').is_file())

    def test_test_diagnostics_are_opt_in_and_do_not_select_checkpoint(self):
        experiment = self.experiment(Exp_Long_Term_Forecast, 'long_term_forecast', ForecastModel())
        experiment.args.diagnostic_test = True
        calls = self.attach(experiment, {flag: self.forecasting_loader() for flag in ['train', 'val', 'test']})
        experiment.train('diagnostics')
        self.assertEqual(calls, ['train', 'val', 'test'])
        self.assertIn('diagnostic_test', experiment.training_history[0])

    def test_training_reloads_current_best_even_when_standalone_override_is_set(self):
        experiment = self.experiment(Exp_Long_Term_Forecast, 'long_term_forecast', ForecastModel())
        experiment.args.checkpoint_run = 'does-not-exist'
        self.attach(experiment, {flag: self.forecasting_loader() for flag in ['train', 'val']})
        result = experiment.train('current-training')
        self.assertIsInstance(result, TrainResult)

    def test_zero_shot_inference_has_no_training_or_checkpoint_dependency(self):
        experiment = self.experiment(Exp_Zero_Shot_Forecast, 'zero_shot_forecast', ForecastModel())
        calls = self.attach(experiment, {'test': self.forecasting_loader()})
        result = experiment.test('pretrained-fixture', test=1)
        self.assertEqual(calls, ['test'])
        self.assertEqual(result['status'], 'complete')
        with self.assertRaisesRegex(ValueError, 'inference only'):
            experiment.train('invalid-training')

    def test_actual_tsbooster_single_target_trains_and_reloads_checkpoint_exactly(self):
        from run import main
        requested_device = os.environ.get('TSBOOSTER_TEST_DEVICE', 'cpu')
        device_flags = ['--gpu_type', requested_device,
                        '--no_use_gpu' if requested_device == 'cpu' else '--use_gpu']
        directory = Path(self.temporary.name)
        index = np.arange(60)
        # Target starts before the other feature: the real loader must reorder it.
        pd.DataFrame({'date': pd.date_range('2020-01-01', periods=60, freq='D'),
                      'OT': 3 + np.cos(index / 5) + index / 20,
                      'auxiliary': np.sin(index / 4) + index / 10}).to_csv(directory / 'tiny.csv', index=False)
        common = [
            '--task_name', 'long_term_forecast', '--model_id', 'actual-tsb',
            '--model', 'TSBooster', '--data', 'custom', '--no_plots',
            '--features', 'MS', '--freq', 'd', '--seq_len', '4', '--label_len', '2', '--pred_len', '2',
            '--enc_in', '2', '--dec_in', '2', '--c_out', '1', '--d_model', '8', '--n_heads', '2',
            '--e_layers', '1', '--d_layers', '1', '--d_ff', '16', '--dropout', '0',
            '--train_epochs', '2', '--batch_size', '8', '--num_workers', '0',
            '--root_path', str(directory), '--data_path', 'tiny.csv',
            '--checkpoints', str(directory / 'weights'),
            '--results_root', str(directory / 'actual-results'),
            '--test_results_root', str(directory / 'plots')] + device_flags
        self.assertEqual(main(common + ['--is_training', '1']), 0)
        trained_directory, = (directory / 'actual-results').iterdir()
        self.assertTrue((directory / 'weights' / trained_directory.name / 'checkpoint.pth').is_file())
        first = json.loads((trained_directory / 'result.json').read_text())
        first_metadata = json.loads((trained_directory / 'metadata.json').read_text())
        self.assertEqual(first_metadata['device'].split(':')[0], requested_device)
        self.assertEqual(first['status'], 'complete')
        first_prediction = np.load(trained_directory / 'pred.npy')
        self.assertEqual(first_prediction.shape, (11, 2, 1))
        self.assertEqual(main(common + ['--is_training', '0', '--checkpoint_run', trained_directory.name]), 0)
        evaluated_directory, = [path for path in (directory / 'actual-results').iterdir()
                                if path != trained_directory]
        second_metadata = json.loads((evaluated_directory / 'metadata.json').read_text())
        self.assertEqual(second_metadata['status'], 'complete')
        self.assertEqual(second_metadata['device'].split(':')[0], requested_device)
        self.assertEqual(second_metadata['checkpoint_source']['sha256'],
                         file_fingerprint(directory / 'weights' / trained_directory.name / 'checkpoint.pth'))
        self.assertNotEqual(first['experiment_id'], second_metadata['experiment_id'])
        second_prediction = np.load(evaluated_directory / 'pred.npy')
        if requested_device == 'cpu':
            np.testing.assert_array_equal(first_prediction, second_prediction)
        else:
            np.testing.assert_allclose(first_prediction, second_prediction, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
