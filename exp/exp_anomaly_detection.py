import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import evaluation_stats
from exp.common.runtime import atomic_save_array, load_checkpoint, precision_context, save_evaluation
from exp.common.training import WeightedLoss, require_same_shape, synchronize, train_model
from utils.tools import adjustment


def anomaly_metrics(targets, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, average='binary', zero_division=0)
    return {'accuracy': float(accuracy_score(targets, predictions)),
            'precision': float(precision), 'recall': float(recall), 'f_score': float(f1)}


class Exp_Anomaly_Detection(Exp_Basic):
    def _validate_features(self):
        if self.args.features == 'MS':
            raise ValueError('anomaly_detection requires M or S; MS reconstruction is not supported.')
        if not 0 < self.args.anomaly_ratio < 100:
            raise ValueError('anomaly_ratio is a percentile and must be between 0 and 100.')

    def _build_model(self):
        self._validate_features()
        return super()._build_model()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _batch_loss(self, batch, criterion):
        self._validate_features()
        x = batch[0].float().to(self.device)
        output = self.model(x, None, None, None)
        require_same_shape(output, x)
        return criterion(output, x), x.numel()

    def vali(self, vali_data, vali_loader, criterion):
        total = WeightedLoss()
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for batch in vali_loader:
                    with precision_context(self.args, self.device):
                        loss, count = self._batch_loss(batch, criterion)
                    total.update(loss, count)
            return total.mean()
        finally:
            self.model.train(previous_mode)

    def train(self, setting):
        _, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')
        criterion = self._select_criterion()
        diagnostic = None
        if getattr(self.args, 'diagnostic_test', False):
            test_data, test_loader = self._get_data('test')
            diagnostic = lambda: self.vali(test_data, test_loader, criterion)
        return train_model(self, setting, train_loader,
                           lambda batch: self._batch_loss(batch, criterion),
                           lambda: self.vali(val_data, val_loader, criterion),
                           diagnostic=diagnostic)

    def _energy(self, loader, include_labels=False):
        self._validate_features()
        energies, labels, forward_s, samples = [], [], 0.0, 0
        self.model.eval()
        with torch.inference_mode():
            for x, label in loader:
                x = x.float().to(self.device)
                synchronize(self.device)
                before = time.perf_counter()
                with precision_context(self.args, self.device):
                    output = self.model(x, None, None, None)
                synchronize(self.device)
                forward_s += time.perf_counter() - before
                require_same_shape(output, x)
                energies.append(((x.float() - output.float()) ** 2).mean(dim=-1).cpu().numpy())
                samples += x.size(0)
                if include_labels:
                    labels.append(torch.as_tensor(label).cpu().numpy())
        if not energies:
            raise ValueError('No anomaly windows are available for evaluation.')
        energy = np.concatenate(energies).reshape(-1)
        if not np.isfinite(energy).all():
            raise ValueError('Anomaly energy must be finite.')
        truth = np.concatenate(labels).reshape(-1).astype(int) if include_labels else None
        if include_labels and (truth.shape != energy.shape or not np.isin(truth, [0, 1]).all()):
            raise ValueError('Anomaly labels must be binary and align with per-time-point energies.')
        return energy, truth, forward_s, samples

    def test(self, setting, test=0, train_time=None):
        self._validate_features()
        protocol = getattr(self.args, 'anomaly_protocol', 'calibration')
        if protocol not in {'calibration', 'legacy'}:
            raise ValueError('anomaly_protocol must be calibration or legacy.')
        if test:
            load_checkpoint(self.model, self.args, setting, self.device)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        # The deployment-style default estimates its threshold from an independent
        # validation calibration split, before loading or scoring TEST.
        calibration_flag = 'val' if protocol == 'calibration' else 'train'
        _, calibration_loader = self._get_data(calibration_flag)
        calibration_energy, _, calibration_s, _ = self._energy(calibration_loader)
        threshold = float(np.percentile(calibration_energy, 100 - self.args.anomaly_ratio))
        _, test_loader = self._get_data('test')
        test_energy, truth, forward_s, sample_count = self._energy(test_loader, include_labels=True)
        if protocol == 'legacy':
            threshold = float(np.percentile(np.concatenate((calibration_energy, test_energy)),
                                            100 - self.args.anomaly_ratio))
        raw_prediction = (test_energy > threshold).astype(int)
        _, adjusted_prediction = adjustment(truth.copy(), raw_prediction.copy())
        metrics = anomaly_metrics(truth, raw_prediction)
        adjusted_metrics = anomaly_metrics(truth, adjusted_prediction)
        extra = evaluation_stats(self, train_time, forward_s, sample_count, time.perf_counter() - started)
        extra.update({'evaluation_protocol': protocol, 'threshold': threshold,
                      'threshold_source': 'validation-only' if protocol == 'calibration' else 'train+test energies',
                      'primary_prediction': 'raw; no ground-truth point adjustment',
                      'point_adjusted_metrics': adjusted_metrics,
                      'num_scored_time_points': int(truth.size),
                      'calibration_forward_s': calibration_s, 'anomaly_ratio_percent': self.args.anomaly_ratio,
                      'evaluation_scope': 'calibration, TEST energies, threshold, raw/adjusted metrics; excludes artifact save',
                      'additional_artifacts': ['energy.npy', 'calibration_energy.npy', 'point_adjusted_pred.npy']})
        directory = os.path.join(getattr(self.args, 'results_root', './results'), setting)
        for name, value in [('energy.npy', test_energy), ('calibration_energy.npy', calibration_energy),
                            ('point_adjusted_pred.npy', adjusted_prediction)]:
            atomic_save_array(os.path.join(directory, name), value)
        result = save_evaluation(self.args, setting, metrics, raw_prediction, truth, extra=extra)
        print(f'Threshold:{threshold}, protocol:{protocol}, raw:{metrics}, point_adjusted:{adjusted_metrics}')
        return result
