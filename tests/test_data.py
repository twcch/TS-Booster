import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Custom, Dataset_M4, MSLSegLoader, UEAloader
from data_provider.uea import collate_fn, padding_mask
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.timefeatures import time_features
from tools.data.fetch_yfinance_data import fetch


class SpawnClassificationFixture(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return torch.ones(index + 1, 1), torch.tensor([index])


def arguments(**kwargs):
    values = dict(augmentation_ratio=0, seed=2, rand_seed=2021, val_ratio=.2,
                  allow_remote_data=False, data_revision=None, features='MS', target='Close',
                  task_name='long_term_forecast', data='custom', embed='timeF', batch_size=32,
                  freq='b', num_workers=0, seq_len=5, label_len=2, pred_len=2, seasonal_patterns='Yearly')
    values.update(kwargs)
    return SimpleNamespace(**values)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=100),
                                   'Close': np.arange(100.) * 2 + 20,
                                   'Open': np.arange(100.) + 10})
        self.frame.to_csv(self.root / 'data.csv', index=False)
        self.args = arguments(root_path=str(self.root), data_path='data.csv')

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, flag='train', **kwargs):
        options = dict(size=[5, 2, 2], features='MS', target='Close', timeenc=1, freq='b')
        options.update(kwargs)
        return Dataset_Custom(self.args, str(self.root), data_path='data.csv', flag=flag, **options)

    def test_chronological_targets_and_train_only_scaler(self):
        train, val, test = [self.dataset(flag) for flag in ['train', 'val', 'test']]
        self.assertEqual([len(s) for s in [train, val, test]], [64, 9, 19])
        expected = self.frame.Close.iloc[:70].mean()
        for dataset in [train, val, test]:
            self.assertEqual(dataset.scaler.mean_[-1], expected)
        self.assertEqual(val.inverse_transform_target(val[0][1][-2:, -1:])[0, 0], self.frame.Close.iloc[70])
        self.assertEqual(test.inverse_transform_target(test[0][1][-2:, -1:])[0, 0], self.frame.Close.iloc[80])

    def test_target_inverse_and_full_inverse(self):
        dataset = self.dataset()
        np.testing.assert_allclose(dataset.inverse_transform(dataset.data_x), self.frame[['Open', 'Close']].iloc[:70])
        np.testing.assert_allclose(dataset.inverse_transform(dataset.data_x[:, -1:], target_only=True), self.frame[['Close']].iloc[:70])
        with self.assertRaisesRegex(ValueError, 'feature count'):
            dataset.inverse_transform(dataset.data_x[:, -1:])

    def test_schema_rejects_unsorted_duplicate_nonfinite_and_short(self):
        variants = [self.frame.iloc[::-1], pd.concat([self.frame, self.frame.iloc[-1:]]),
                    self.frame.assign(Close=np.nan), self.frame.iloc[:7]]
        for frame in variants:
            with self.subTest(rows=len(frame)):
                frame.to_csv(self.root / 'data.csv', index=False)
                with self.assertRaises(ValueError):
                    self.dataset()

    def test_timeenc_zero_uses_vectorized_calendar(self):
        dataset = self.dataset(timeenc=0)
        self.assertEqual(dataset.data_stamp.shape, (70, 4))
        np.testing.assert_array_equal(dataset.data_stamp[0], [1, 1, 2, 0])
        for frequency in ['min', '15min', 't']:
            self.assertEqual(self.dataset(timeenc=0, freq=frequency).data_stamp.shape, (70, 5))

    def test_validation_sampler_keeps_tail_and_collate_is_pickleable(self):
        _, loader = data_provider(self.args, 'val')
        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertFalse(loader.drop_last)
        self.assertEqual(sum(len(batch[0]) for batch in loader), 9)
        from functools import partial
        restored = pickle.loads(pickle.dumps(partial(collate_fn, max_len=5)))
        batch = restored([(torch.ones(2, 1), torch.tensor([0]))])
        self.assertEqual(batch[2].sum().item(), 2)

    def test_loader_memory_and_worker_options_apply_to_every_task(self):
        class LoaderFixture(SpawnClassificationFixture):
            def __init__(self, **kwargs):
                pass

        for task in ['long_term_forecast', 'classification', 'anomaly_detection']:
            for device, workers, enabled, expected_pin in [
                ('cpu', 0, True, False), ('mps', 2, True, False),
                ('cuda', 2, True, True), ('cuda', 2, False, False),
                ('cuda', 0, True, True),
            ]:
                with self.subTest(task=task, device=device, workers=workers, enabled=enabled):
                    args = arguments(root_path=str(self.root), data_path='data.csv', task_name=task,
                                     device=torch.device(device), num_workers=workers,
                                     pin_memory=enabled, persistent_workers=enabled)
                    with patch.dict('data_provider.data_factory.data_dict', {'custom': LoaderFixture}):
                        _, loader = data_provider(args, 'val')
                    self.assertEqual(loader.pin_memory, expected_pin)
                    self.assertEqual(loader.persistent_workers, enabled and workers > 0)
                    self.assertFalse(loader.drop_last)
                    self.assertIsInstance(loader.sampler, SequentialSampler)

    def test_remote_is_disabled_and_revision_is_required(self):
        self.args.data_path = 'missing.csv'
        with patch('data_provider.data_loader.load_dataset') as remote:
            with self.assertRaises(FileNotFoundError):
                Dataset_Custom(self.args, str(self.root), data_path='missing.csv', target='Close')
            remote.assert_not_called()
            self.args.allow_remote_data = True
            with self.assertRaisesRegex(ValueError, 'data_revision'):
                Dataset_Custom(self.args, str(self.root), data_path='missing.csv', target='Close')
            remote.assert_not_called()

    def test_forecast_augmentation_preserves_labels_and_calendar(self):
        base = self.dataset()
        self.args.augmentation_ratio = 2
        self.args.jitter = True
        expanded = self.dataset()
        self.assertEqual(len(expanded), len(base) * 3)
        for replica in [1, 2]:
            original, changed = base[3], expanded[replica * len(base) + 3]
            self.assertFalse(np.array_equal(original[0], changed[0]))
            for i in [1, 2, 3]:
                np.testing.assert_array_equal(original[i], changed[i])
        self.args.timewarp = True
        with self.assertRaisesRegex(ValueError, 'target/time mapping'):
            self.dataset()

    def test_anomaly_holdout_is_disjoint_and_scaler_excludes_it(self):
        original = np.arange(200.).reshape(100, 2)
        np.save(self.root / 'MSL_train.npy', original)
        np.save(self.root / 'MSL_test.npy', original + 1000)
        np.save(self.root / 'MSL_test_label.npy', np.zeros((100, 1)))
        datasets = [MSLSegLoader(self.args, str(self.root), 10, flag=f) for f in ['train', 'val', 'test']]
        for data in datasets:
            np.testing.assert_allclose(data.scaler.mean_, original[:80].mean(axis=0))
            self.assertEqual((len(data.train), len(data.val)), (80, 20))
        np.testing.assert_allclose(datasets[0].scaler.inverse_transform(datasets[0].train)[-1], original[79], atol=1e-5)
        np.testing.assert_allclose(datasets[1].scaler.inverse_transform(datasets[1].val)[0], original[80], atol=1e-5)

    def test_classification_shared_scaler_labels_and_stratified_holdout(self):
        class FixtureUEA(UEAloader):
            def _resolve_ts_path(self, root, name, flag):
                return flag
            def load_single(self, flag):
                count = 4 if flag == 'TEST' else 10
                ids = np.repeat(np.arange(count), 3)
                offset = 1000 if flag == 'TEST' else 0
                data = pd.DataFrame({'feature': np.arange(count * 3.) + offset}, index=ids)
                # TEST omits class a: the remaining b must keep code 1.
                labels = ['b'] * count if flag == 'TEST' else ['a'] * 5 + ['b'] * 5
                return data, pd.DataFrame({0: labels})
        self.args.model_id = 'fixture'
        train, val, test = [FixtureUEA(self.args, str(self.root), flag=f) for f in ['TRAIN', 'VAL', 'TEST']]
        self.assertFalse(set(train.all_IDs) & set(val.all_IDs))
        self.assertEqual(len(train) + len(val), 10)
        self.assertEqual(set(val.labels_df[0]), {0, 1})
        self.assertEqual(set(test.labels_df[0]), {1})
        for data in [val, test]:
            pd.testing.assert_series_equal(data.normalizer.mean, train.normalizer.mean)
            pd.testing.assert_series_equal(data.normalizer.std, train.normalizer.std)
        self.assertGreater(test.feature_df.feature.mean(), 10)

    def test_m4_train_tail_val_and_official_test_context(self):
        train_series = np.empty(2, dtype=object)
        train_series[:] = [np.arange(12.), np.arange(15.) + 100]
        test_series = np.array([[20., 21.], [120., 121.]])
        info = dict(ids=np.array(['Y1', 'Y2']), groups=np.array(['Yearly', 'Yearly']))
        def fixture(training, **kwargs):
            return SimpleNamespace(**info, values=train_series if training else test_series)
        with patch('data_provider.data_loader.M4Dataset.load', side_effect=fixture) as load:
            train, val, test = [Dataset_M4(self.args, str(self.root), flag=f, size=[5, 2, 2]) for f in ['train', 'val', 'test']]
        self.assertEqual(load.call_count, 4)
        np.testing.assert_array_equal(train.timeseries[0], np.arange(10.))
        np.testing.assert_array_equal(val.timeseries[0], [10., 11.])
        np.testing.assert_array_equal(val.last_insample_window()[0][0], [5., 6., 7., 8., 9.])
        np.testing.assert_array_equal(test.last_insample_window()[0][0], [7., 8., 9., 10., 11.])
        np.testing.assert_array_equal(test.timeseries, test_series)
        np.testing.assert_array_equal(test.inverse_transform(np.array([3.])), [3.])

    def test_m4_real_ragged_archive_needs_no_test_for_validation(self):
        series = np.empty(2, dtype=object)
        series[:] = [np.arange(12.), np.arange(15.)]
        pd.DataFrame({'M4id': ['Y1', 'Y2'], 'SP': ['Yearly', 'Yearly'],
                      'Frequency': [1, 1], 'Horizon': [2, 2]}).to_csv(self.root / 'M4-info.csv', index=False)
        np.savez(self.root / 'training.npz', values=series)
        validation = Dataset_M4(self.args, str(self.root), flag='val', size=[5, 2, 2])
        self.assertEqual([len(v) for v in validation.timeseries], [2, 2])
        self.assertFalse((self.root / 'test.npz').exists())

    def test_m4_short_history_padding_preserves_forecast_boundary(self):
        series = np.empty(1, dtype=object)
        series[0] = np.arange(10.) + 10
        fixture = SimpleNamespace(ids=np.array(['Y1']), groups=np.array(['Yearly']), values=series)
        with patch('data_provider.data_loader.M4Dataset.load', return_value=fixture):
            train = Dataset_M4(self.args, str(self.root), flag='train', size=[5, 4, 2])
        train._rng = SimpleNamespace(integers=lambda *args: 1)
        train._worker_seed = self.args.rand_seed
        _, target, _, mask = train[0]
        np.testing.assert_array_equal(target[:, 0], [0, 0, 0, 10, 11, 12])
        np.testing.assert_array_equal(mask[:, 0], [0, 0, 0, 1, 1, 1])
        # A cut near the end has a missing future position, not missing history.
        train._rng = SimpleNamespace(integers=lambda *args: len(train.context[0]) - 1)
        _, target, _, mask = train[0]
        np.testing.assert_array_equal(target[-2:, 0], [17, 0])
        np.testing.assert_array_equal(mask[-2:, 0], [1, 0])

        series[0] = np.arange(4.) + 10
        with patch('data_provider.data_loader.M4Dataset.load', return_value=fixture):
            validation = Dataset_M4(self.args, str(self.root), flag='val', size=[5, 4, 2])
        _, target, _, mask = validation[0]
        np.testing.assert_array_equal(target[:, 0], [0, 0, 10, 11, 12, 13])
        np.testing.assert_array_equal(mask[:, 0], [0, 0, 1, 1, 1, 1])

    def test_yfinance_fetch_manifest_and_refuses_overwrite_before_download(self):
        frame = self.frame.iloc[:5].copy().set_index('date')
        frame['High'], frame['Low'] = frame.Close + 1, frame.Close - 1
        frame.index.name = 'Date'
        download = unittest.mock.Mock(return_value=frame)
        provider = SimpleNamespace(download=download, __version__='fixture')
        destination = self.root / 'fetched.csv'
        with patch.dict('sys.modules', {'yfinance': provider}):
            output, manifest = fetch('TEST', '2020-01-01', '2020-01-06', destination, auto_adjust=False)
            metadata = json.loads(manifest.read_text())
            self.assertEqual(metadata['rows'], 5)
            self.assertFalse(metadata['auto_adjust'])
            self.assertEqual(metadata['actual_start'][:10], '2020-01-01')
            with self.assertRaises(FileExistsError):
                fetch('TEST', '2020-01-01', '2020-01-06', destination)
            self.assertEqual(download.call_count, 1)
            self.assertEqual(list(pd.read_csv(output)), ['date', 'Open', 'High', 'Low', 'Close'])


class UtilityTests(unittest.TestCase):
    def test_augmentation_local_rng_reproducibility_and_ratio(self):
        args = arguments(augmentation_ratio=2, jitter=True)
        x, y = np.ones((2, 5, 1)), np.array([0, 1])
        state = np.random.get_state()
        first = run_augmentation(x, y, args)
        second = run_augmentation(x, y, args)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[0].shape, (6, 5, 1))
        np.testing.assert_array_equal(first[1], [0, 1, 0, 1, 0, 1])
        current = np.random.get_state()
        np.testing.assert_array_equal(state[1], current[1])
        self.assertEqual(state[2:], current[2:])
        single, _, _ = run_augmentation_single(x[0], y, args)
        self.assertEqual(single.shape, x[0].shape)

    def test_padding_and_yearly_calendar(self):
        np.testing.assert_array_equal(padding_mask(torch.tensor([2, 3])).numpy(), [[1, 1, 0], [1, 1, 1]])
        dates = pd.date_range('2020-01-01', periods=3)
        self.assertEqual(time_features(dates, 'y').shape, (0, 3))
        self.assertEqual(time_features(dates, 'b').shape, (3, 3))
        for frequency, count in [('YS', 0), ('MS', 1), ('QS', 1)]:
            self.assertEqual(time_features(dates, frequency).shape, (count, 3))

    @unittest.skipUnless(os.environ.get('TSBOOSTER_TEST_SPAWN') == '1',
                         'Set TSBOOSTER_TEST_SPAWN=1 in an environment permitting shared memory')
    def test_classification_collate_spawn_worker_keeps_tail(self):
        from functools import partial
        loader = DataLoader(SpawnClassificationFixture(), batch_size=2, num_workers=1,
                            multiprocessing_context='spawn', drop_last=False,
                            collate_fn=partial(collate_fn, max_len=4))
        batches = list(loader)
        self.assertEqual([len(batch[0]) for batch in batches], [2, 1])
        self.assertEqual(torch.cat([batch[1] for batch in batches]).flatten().tolist(), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
