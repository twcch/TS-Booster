"""Datasets with explicit chronological and train-only preprocessing policies."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info
from sklearn.preprocessing import StandardScaler

from data_provider.m4 import M4Dataset, M4Meta
from data_provider.preparation import (
    calendar_features, read_forecast_csv, split_anomaly_data, validate_forecast_frame, validate_window,
)
from data_provider.uea import interpolate_missing, Normalizer
from utils.augmentation import augment, validate_forecast_augmentation

HUGGINGFACE_REPO = "thuml/Time-Series-Library"


def _remote_options(args):
    if not getattr(args, 'allow_remote_data', False):
        raise FileNotFoundError("Local dataset missing; enable allow_remote_data with a pinned data_revision to download")
    revision = getattr(args, 'data_revision', None)
    if not revision:
        raise ValueError("Remote data requires an explicit data_revision")
    return {'revision': revision}


def load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset as load
    except ImportError as exc:
        raise ImportError("Remote tabular data requires the optional datasets package") from exc
    return load(*args, **kwargs)


def hf_hub_download(*args, **kwargs):
    try:
        from huggingface_hub import hf_hub_download as download
    except ImportError as exc:
        raise ImportError("Remote data requires the optional huggingface_hub package") from exc
    return download(*args, **kwargs)


class _ForecastDataset(Dataset):
    ett_multiplier = None

    def __init__(self, args, root_path, flag='train', size=None, features='S',
                 data_path='ETTh1.csv', target='OT', scale=True, timeenc=0,
                 freq='h', seasonal_patterns=None):
        self.args = args
        self.seq_len, self.label_len, self.pred_len = size or [384, 96, 96]
        validate_window([self.seq_len, self.label_len, self.pred_len])
        # Steps skipped between the origin and the first forecast step (targets t+o+1 .. t+o+H).
        self.pred_offset = int(getattr(args, 'pred_offset', 0) or 0)
        if self.pred_offset < 0:
            raise ValueError("pred_offset cannot be negative")
        horizon = self.pred_offset + self.pred_len
        flag = flag.lower()
        if flag not in {'train', 'val', 'test'}:
            raise ValueError("forecast flag must be train, val, or test")
        self.flag, self.set_type = flag, {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features, self.target, self.scale = features, target, scale
        self.timeenc, self.freq = timeenc, freq
        self.root_path, self.data_path = root_path, data_path
        path = os.path.join(root_path, data_path)
        if os.path.isfile(path):
            frame = read_forecast_csv(path)
        else:
            options = _remote_options(args)
            config = os.path.splitext(os.path.basename(data_path))[0]
            remote = load_dataset(HUGGINGFACE_REPO, name=config, **options)
            split = 'train' if 'train' in remote else next(iter(remote))
            frame = remote[split].to_pandas()
        frame, self.feature_names, values = validate_forecast_frame(frame, target, features)
        self.target_index = self.feature_names.index(target)
        if self.ett_multiplier is None:
            n_train, n_test = int(len(frame) * .7), int(len(frame) * .2)
            boundaries = [n_train, len(frame) - n_test, len(frame)]
        else:
            unit = 30 * 24 * self.ett_multiplier
            boundaries = [12 * unit, 16 * unit, 20 * unit]
            if len(frame) < boundaries[-1]:
                raise ValueError(f"ETT data requires at least {boundaries[-1]} observations")
        starts = [0, boundaries[0] - self.seq_len, boundaries[1] - self.seq_len]
        first, last = starts[self.set_type], boundaries[self.set_type]
        if boundaries[0] < self.seq_len + horizon or first < 0 or last - first < self.seq_len + horizon:
            raise ValueError(f"{flag} split is too short for seq_len={self.seq_len}, pred_len={self.pred_len}, "
                             f"pred_offset={self.pred_offset}")
        self.scaler = StandardScaler()
        if scale:
            self.scaler.fit(values[:boundaries[0]])
            values = self.scaler.transform(values)
        self.data_x = values[first:last]
        self.data_y = self.data_x
        self.data_stamp = calendar_features(frame.date.iloc[first:last], timeenc, freq,
                                            minute=self.ett_multiplier == 4)
        self.split_bounds = (first, last)
        self.train_boundary = boundaries[0]
        self.base_length = len(self.data_x) - self.seq_len - horizon + 1
        self.augmentation_ratio = int(getattr(args, 'augmentation_ratio', 0)) if flag == 'train' else 0
        if self.augmentation_ratio < 0:
            raise ValueError("augmentation_ratio cannot be negative")
        if self.augmentation_ratio:
            validate_forecast_augmentation(args)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        replica, index = divmod(index, self.base_length)
        end = index + self.seq_len
        y_begin, y_end = end - self.label_len, end + self.pred_offset + self.pred_len
        x = self.data_x[index:end]
        if replica:
            seed = np.random.SeedSequence([getattr(self.args, 'seed', 2), replica, index])
            x, _ = augment(x[None].copy(), self.data_y[y_begin:y_end], self.args,
                           rng=np.random.default_rng(seed))
            x = x[0]
        return x, self.data_y[y_begin:y_end], self.data_stamp[index:end], self.data_stamp[y_begin:y_end]

    def __len__(self):
        return self.base_length * (1 + self.augmentation_ratio)

    def inverse_transform(self, data, target_only=False):
        values = np.asarray(data)
        if target_only:
            return self.inverse_transform_target(values)
        if not self.scale:
            return values.copy()
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("inverse_transform feature count mismatch; use target_only=True for a single target")
        return values * self.scaler.scale_ + self.scaler.mean_

    def inverse_transform_target(self, data, target_channel_index=None):
        values = np.asarray(data)
        if not self.scale:
            return values.copy()
        index = self.target_index if target_channel_index is None else target_channel_index
        if values.ndim > 1 and values.shape[-1] != 1:
            raise ValueError("target inverse expects a final dimension of one")
        return values * self.scaler.scale_[index] + self.scaler.mean_[index]


class Dataset_ETT_hour(_ForecastDataset):
    ett_multiplier = 1


class Dataset_ETT_minute(_ForecastDataset):
    ett_multiplier = 4

    def __init__(self, *args, freq='t', **kwargs):
        super().__init__(*args, freq=freq, **kwargs)


class Dataset_Custom(_ForecastDataset):
    pass


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='train', size=None, features='S',
                 data_path='ETTh1.csv', target='OT', scale=False, inverse=False,
                 timeenc=0, freq='15min', seasonal_patterns='Yearly'):
        self.args, self.root_path, self.flag = args, root_path, flag.lower()
        self.seq_len, self.label_len, self.pred_len = size or [36, 18, 6]
        validate_window([self.seq_len, self.label_len, self.pred_len])
        if self.flag not in {'train', 'train_full', 'val', 'test', 'pred'}:
            raise ValueError("M4 flag must be train, train_full, val, or test")
        if scale:
            raise ValueError("M4 currently supports scale=False only")
        self.features, self.target, self.scale = features, target, False
        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        options = {'allow_remote_data': getattr(args, 'allow_remote_data', False),
                   'revision': getattr(args, 'data_revision', None)}
        training = M4Dataset.load(training=True, dataset_file=root_path, **options)
        selected = training.groups == seasonal_patterns
        self.ids = np.asarray(training.ids)[selected]
        full = [np.asarray(v, dtype=float)[~np.isnan(v)] for v in training.values[selected]]
        if any(not np.isfinite(v).all() for v in full):
            raise ValueError('M4 training values must be finite after removing NaN padding')
        if not full:
            raise ValueError(f'M4 contains no series for {seasonal_patterns}')
        if any(len(v) < self.pred_len + 2 for v in full):
            raise ValueError("M4 training series requires a validation horizon and at least two training values")
        before = [v[:-self.pred_len] for v in full]
        if self.flag == 'train':
            self.timeseries, self.context = before, before
        elif self.flag == 'val':
            self.timeseries, self.context = [v[-self.pred_len:] for v in full], before
        elif self.flag == 'train_full':
            self.timeseries, self.context = full, full
        else:
            testing = M4Dataset.load(training=False, dataset_file=root_path, **options)
            test_map = {str(i): np.asarray(v, dtype=float)[~np.isnan(v)]
                        for i, group, v in zip(testing.ids, testing.groups, testing.values)
                        if group == seasonal_patterns}
            self.timeseries = [test_map[str(i)] for i in self.ids]
            if any(not np.isfinite(v).all() for v in self.timeseries):
                raise ValueError('M4 test values must be finite after removing NaN padding')
            if any(len(v) != self.pred_len for v in self.timeseries):
                raise ValueError("M4 test horizon does not match pred_len")
            self.context = full
        self._rng = None
        self._worker_seed = None

    def __getitem__(self, index):
        series = self.context[index]
        if self.flag in {'train', 'train_full'}:
            worker = get_worker_info()
            seed = worker.seed if worker else getattr(self.args, 'rand_seed', 2021)
            if self._rng is None or seed != self._worker_seed:
                self._rng, self._worker_seed = np.random.default_rng(seed), seed
            cut = self._rng.integers(max(1, len(series) - self.window_sampling_limit), len(series))
            future = series[cut:cut + self.pred_len]
        else:
            cut = len(series)
            future = self.timeseries[index]
        history = series[max(0, cut - self.label_len):cut]
        x, xm = np.zeros((self.seq_len, 1)), np.zeros((self.seq_len, 1))
        y, ym = np.zeros((self.label_len + self.pred_len, 1)), np.zeros((self.label_len + self.pred_len, 1))
        window = series[max(0, cut - self.seq_len):cut]
        x[-len(window):, 0], xm[-len(window):, 0] = window, 1
        if len(history):
            begin = self.label_len - len(history)
            y[begin:self.label_len, 0], ym[begin:self.label_len, 0] = history, 1
        end = self.label_len + len(future)
        y[self.label_len:end, 0], ym[self.label_len:end, 0] = future, 1
        return x, y, xm, ym

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data, target_only=False):
        return np.asarray(data).copy()

    inverse_transform_target = inverse_transform

    def last_insample_window(self):
        x, mask = np.zeros((len(self.context), self.seq_len)), np.zeros((len(self.context), self.seq_len))
        for i, series in enumerate(self.context):
            window = series[-self.seq_len:]
            x[i, -len(window):], mask[i, -len(window):] = window, 1
        return x, mask


class _AnomalyDataset(Dataset):
    name = None
    kind = 'npy'

    def __init__(self, args, root_path, win_size, step=1, flag='train'):
        self.flag, self.win_size, self.step = flag.lower(), win_size, step
        if win_size <= 0 or step <= 0:
            raise ValueError("win_size and step must be positive")
        if self.name == 'PSM':
            paths = [os.path.join(root_path, n) for n in ['train.csv', 'test.csv', 'test_label.csv']]
            if all(os.path.isfile(p) for p in paths):
                tr, te, labels = [pd.read_csv(p).to_numpy()[:, 1:] for p in paths]
            else:
                options = _remote_options(args)
                data = load_dataset(HUGGINGFACE_REPO, name='PSM-data', **options)
                lab = load_dataset(HUGGINGFACE_REPO, name='PSM-label', **options)
                tr, te = data['train'].to_pandas().to_numpy()[:, 1:], data['test'].to_pandas().to_numpy()[:, 1:]
                labels = lab[next(iter(lab))].to_pandas().to_numpy()[:, 1:]
        elif self.name == 'SWAT':
            paths = [os.path.join(root_path, n) for n in ['swat_train2.csv', 'swat2.csv']]
            if all(os.path.isfile(p) for p in paths):
                tr, te = [pd.read_csv(p).to_numpy() for p in paths]
            else:
                data = load_dataset(HUGGINGFACE_REPO, name='SWaT', **_remote_options(args))
                tr, te = data['train'].to_pandas().to_numpy(), data['test'].to_pandas().to_numpy()
            labels, tr, te = te[:, -1:], tr[:, :-1], te[:, :-1]
        else:
            paths = [os.path.join(root_path, f'{self.name}_{suffix}.npy') for suffix in ['train', 'test', 'test_label']]
            if not all(os.path.isfile(p) for p in paths):
                options = _remote_options(args)
                paths = [hf_hub_download(HUGGINGFACE_REPO, filename=f'{self.name}/{os.path.basename(p)}',
                                         repo_type='dataset', **options) for p in paths]
            tr, te, labels = [np.load(p, allow_pickle=False) for p in paths]
        train, val, test = split_anomaly_data(tr, te, win_size, getattr(args, 'val_ratio', .2))
        self.scaler = StandardScaler().fit(train)
        self.train, self.val, self.test = [self.scaler.transform(v).astype(np.float32) for v in [train, val, test]]
        self.test_labels = np.asarray(labels)
        if len(labels) != len(test):
            raise ValueError("anomaly test labels must match test observations")
        if self.flag not in {'train', 'val', 'test', 'thre'}:
            raise ValueError("anomaly flag must be train, val, test, or thre")

    def __len__(self):
        data = self.test if self.flag == 'thre' else getattr(self, self.flag)
        stride = self.win_size if self.flag == 'thre' else self.step
        return (len(data) - self.win_size) // stride + 1

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        data = self.test if self.flag == 'thre' else getattr(self, self.flag)
        stride = self.win_size if self.flag == 'thre' else self.step
        begin = index * stride
        labels = self.test_labels[begin:begin + self.win_size] if self.flag in {'test', 'thre'} else np.zeros((self.win_size, 1))
        return data[begin:begin + self.win_size], np.asarray(labels, dtype=np.float32)


class PSMSegLoader(_AnomalyDataset):
    name = 'PSM'


class MSLSegLoader(_AnomalyDataset):
    name = 'MSL'


class SMAPSegLoader(_AnomalyDataset):
    name = 'SMAP'


class SMDSegLoader(_AnomalyDataset):
    name = 'SMD'


class SWATSegLoader(_AnomalyDataset):
    name = 'SWAT'


class UEAloader(Dataset):
    """TRAIN-derived stratified holdout, shared feature statistics and label codes."""
    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args, self.root_path, self.flag = args, root_path, (flag or 'TRAIN').upper()
        if self.flag not in {'TRAIN', 'VAL', 'TEST'}:
            raise ValueError("classification flag must be TRAIN, VAL, or TEST")
        train_path = self._resolve_ts_path(root_path, args.model_id, 'TRAIN')
        train, labels = self.load_single(train_path)
        self.class_names = pd.Index(sorted(labels[0].unique()))
        label_map = {name: i for i, name in enumerate(self.class_names)}
        labels = pd.DataFrame({0: labels[0].map(label_map).astype(np.int64)})
        train_ids, val_ids = self._stratified_ids(labels[0].to_numpy(), getattr(args, 'val_ratio', .2), getattr(args, 'rand_seed', 2021))
        self.normalizer = Normalizer()
        self.normalizer.normalize(train.loc[train_ids])
        if self.flag == 'TEST':
            frame, test_labels = self.load_single(self._resolve_ts_path(root_path, args.model_id, 'TEST'))
            if list(frame.columns) != list(train.columns):
                raise ValueError("classification TEST features differ from TRAIN")
            mapped = test_labels[0].map(label_map)
            if mapped.isna().any():
                raise ValueError("classification TEST contains labels absent from TRAIN")
            labels = pd.DataFrame({0: mapped.astype(np.int64)})
            selected = frame.index.unique().to_numpy()
        else:
            frame, selected = train, train_ids if self.flag == 'TRAIN' else val_ids
        if limit_size is not None:
            count = int(limit_size) if limit_size > 1 else int(limit_size * len(selected))
            selected = selected[:count]
        if len(selected) == 0:
            raise ValueError("classification split is empty")
        self.all_IDs = np.asarray(selected)
        self.labels_df = labels.loc[self.all_IDs]
        self.all_df = frame.loc[self.all_IDs]
        self.feature_names = self.all_df.columns
        self.feature_df = self.normalizer.normalize(self.all_df)
        if not np.isfinite(self.feature_df.to_numpy()).all():
            raise ValueError("classification data remains nonfinite after interpolation")
        self.max_seq_len = int(self.all_df.groupby(level=0).size().max())
        self.augmentation_ratio = int(getattr(args, 'augmentation_ratio', 0)) if self.flag == 'TRAIN' else 0
        if self.augmentation_ratio < 0:
            raise ValueError("augmentation_ratio cannot be negative")

    @staticmethod
    def _stratified_ids(labels, val_ratio, seed):
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        rng, train, val = np.random.default_rng(seed), [], []
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label)
            if len(ids) < 2:
                raise ValueError(f"class {label!r} needs at least two TRAIN samples for stratified validation")
            ids = rng.permutation(ids)
            count = min(len(ids) - 1, max(1, int(round(len(ids) * val_ratio))))
            val.extend(ids[:count]); train.extend(ids[count:])
        return np.asarray(sorted(train)), np.asarray(sorted(val))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        filename = f'{dataset_name}_{flag}.ts'
        path = os.path.join(root_path, filename)
        if os.path.isfile(path):
            return path
        return hf_hub_download(HUGGINGFACE_REPO, filename=f'{dataset_name}/{filename}',
                               repo_type='dataset', **_remote_options(self.args))

    def load_all(self, root_path, file_list=None, flag=None):
        return self.load_single(self._resolve_ts_path(root_path, self.args.model_id, (flag or 'TRAIN').upper()))

    def load_single(self, filepath):
        try:
            from sktime.datasets import load_from_tsfile_to_dataframe
        except ImportError as exc:
            raise ImportError("UEA classification requires the optional sktime package") from exc
        frame, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
        records = []
        for sample in range(len(frame)):
            # Unequal channel lengths align on position; remaining values interpolate per sample.
            values = pd.DataFrame({col: pd.Series(np.asarray(frame.iloc[sample][col], dtype=float)) for col in frame})
            values = values.apply(interpolate_missing)
            values.index = pd.Index([sample] * len(values))
            records.append(values)
        return pd.concat(records), pd.DataFrame({0: np.asarray(labels).astype(str)})

    def instance_norm(self, case):
        if 'EthanolConcentration' in str(self.root_path):
            case = case - case.mean(0, keepdim=True)
            case = case / torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return case

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        replica, position = divmod(index, len(self.all_IDs))
        sample = self.all_IDs[position]
        values = self.feature_df.loc[[sample]].to_numpy()
        labels = self.labels_df.loc[sample].to_numpy(dtype=np.int64)
        if replica:
            rng = np.random.default_rng(np.random.SeedSequence([getattr(self.args, 'seed', 2), replica, int(sample)]))
            values, _ = augment(values[None].copy(), labels, self.args, rng=rng)
            values = values[0]
        return self.instance_norm(torch.from_numpy(values)), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs) * (1 + self.augmentation_ratio)
