import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import evaluation_stats, plot_directory, regression_metrics
from exp.common.runtime import load_checkpoint, precision_context, save_evaluation
from exp.common.training import WeightedLoss, require_same_shape, synchronize, train_model
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.tools import visual


def m4_loss(criterion, insample, frequency, prediction, target, mask):
    """Adapt ordinary MSE and benchmark M4 losses to effective-mask means."""
    require_same_shape(prediction, target)
    if mask.shape != target.shape:
        raise ValueError('M4 target and mask shapes must agree.')
    valid = mask > 0
    count = int(valid.sum().item())
    if not count:
        raise ValueError('M4 loss requires at least one valid forecast target.')
    if isinstance(criterion, nn.MSELoss):
        return criterion(prediction[valid], target[valid]), count
    loss = criterion(insample, frequency, prediction, target, mask)
    # Original benchmark losses divide by all padded positions; use the actual
    # valid target count so different padding/batch grouping cannot reweight loss.
    return loss * target.numel() / count, count


class Exp_Short_Term_Forecast(Exp_Basic):
    def _build_model(self):
        if self.args.data != 'm4':
            raise ValueError('short_term_forecast currently supports --data m4 only.')
        from data_provider.m4 import M4Meta
        self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]
        self.args.seq_len = 2 * self.args.pred_len
        self.args.label_len = self.args.pred_len
        self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        return super()._build_model()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self, loss_name='MSE'):
        criteria = {'MSE': nn.MSELoss, 'MAPE': mape_loss, 'MASE': mase_loss, 'SMAPE': smape_loss}
        try:
            return criteria[loss_name.upper()]()
        except KeyError as error:
            raise ValueError(f'Unsupported M4 loss: {loss_name}') from error

    def _batch_loss(self, batch, criterion):
        x, y, _, target_mask = batch
        x, y, target_mask = [value.float().to(self.device) for value in (x, y, target_mask)]
        decoder = torch.cat((y[:, :self.args.label_len],
                             torch.zeros_like(y[:, -self.args.pred_len:])), dim=1)
        output = self.model(x, None, decoder, None)
        if not torch.is_tensor(output):
            raise TypeError('The selected model does not return an M4 forecast tensor.')
        prediction = output[:, -self.args.pred_len:, :]
        target = y[:, -self.args.pred_len:, :]
        mask = target_mask[:, -self.args.pred_len:, :]
        require_same_shape(prediction, target)
        if target.size(-1) != 1:
            raise ValueError('The M4 adapter requires single-channel series.')
        return m4_loss(criterion, x[:, :, 0], self.args.frequency_map,
                       prediction[:, :, 0], target[:, :, 0], mask[:, :, 0])

    def _window_batches(self, dataset):
        # Each split owns its context: val stops before TRAIN holdout; test uses
        # the complete official TRAIN sequence, without reading TEST labels.
        context, _ = dataset.last_insample_window()
        targets = np.stack(dataset.timeseries).astype(np.float32)
        if targets.shape != (len(context), self.args.pred_len):
            raise ValueError('M4 holdout targets must match [series,pred_len].')
        size = getattr(self.args, 'eval_batch_size', None) or self.args.batch_size
        if size < 1:
            raise ValueError('eval_batch_size must be positive.')
        for start in range(0, len(context), size):
            x = torch.as_tensor(context[start:start + size], dtype=torch.float32, device=self.device).unsqueeze(-1)
            target = torch.as_tensor(targets[start:start + size], dtype=torch.float32, device=self.device).unsqueeze(-1)
            decoder = torch.cat((x[:, -self.args.label_len:],
                                 torch.zeros((x.size(0), self.args.pred_len, 1), device=self.device)), dim=1)
            yield start, x, target, decoder

    def vali(self, train_loader, vali_loader, criterion):
        total = WeightedLoss()
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for _, x, target, decoder in self._window_batches(vali_loader.dataset):
                    with precision_context(self.args, self.device):
                        output = self.model(x, None, decoder, None)
                        if not torch.is_tensor(output):
                            raise TypeError('The selected model does not return an M4 forecast tensor.')
                        output = output[:, -self.args.pred_len:]
                        require_same_shape(output, target)
                        loss, count = m4_loss(criterion, x[:, :, 0], self.args.frequency_map,
                                              output[:, :, 0], target[:, :, 0],
                                              torch.ones_like(target[:, :, 0]))
                    total.update(loss, count)
            return total.mean()
        finally:
            self.model.train(previous_mode)

    def train(self, setting):
        _, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')
        criterion = self._select_criterion(self.args.loss)
        diagnostic = None
        if getattr(self.args, 'diagnostic_test', False):
            _, test_loader = self._get_data('test')
            diagnostic = lambda: self.vali(train_loader, test_loader, criterion)
        return train_model(self, setting, train_loader,
                           lambda batch: self._batch_loss(batch, criterion),
                           lambda: self.vali(train_loader, val_loader, criterion),
                           diagnostic=diagnostic)

    def test(self, setting, test=0, train_time=None):
        dataset, _ = self._get_data('test')
        if test:
            load_checkpoint(self.model, self.args, setting, self.device)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        predictions, targets, contexts, forward_s = [], [], [], 0.0
        self.model.eval()
        with torch.inference_mode():
            for _, x, target, decoder in self._window_batches(dataset):
                synchronize(self.device)
                before = time.perf_counter()
                with precision_context(self.args, self.device):
                    output = self.model(x, None, decoder, None)
                synchronize(self.device)
                forward_s += time.perf_counter() - before
                if not torch.is_tensor(output):
                    raise TypeError('The selected model does not return an M4 forecast tensor.')
                output = output[:, -self.args.pred_len:]
                require_same_shape(output, target)
                predictions.append(output.float().cpu().numpy())
                targets.append(target.cpu().numpy())
                if not getattr(self.args, 'no_plots', False):
                    contexts.append(x.cpu().numpy())
        if not predictions:
            raise ValueError('No M4 test series are available.')
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        count = len(predictions)
        if not getattr(self.args, 'no_plots', False):
            directory, inputs = plot_directory(self.args, setting), np.concatenate(contexts)
            for index in range(0, count, max(1, count // 10)):
                visual(np.concatenate((inputs[index, :, 0], targets[index, :, 0])),
                       np.concatenate((inputs[index, :, 0], predictions[index, :, 0])),
                       os.path.join(directory, f'{index}.pdf'))
        metrics = regression_metrics(predictions, targets)
        extra = evaluation_stats(self, train_time, forward_s, count, time.perf_counter() - started)
        extra.update({'metric_space': 'original', 'evaluation_protocol': 'TRAIN tail holdout; official TEST final only',
                      'evaluation_scope': 'M4 contexts, transfers, forward, metrics, sampled plots; excludes CSV/NPY/JSON save',
                      'seasonal_patterns': self.args.seasonal_patterns, 'series_ids': list(map(str, dataset.ids))})
        # Save IDs explicitly instead of replacing the ID index with forecast V1.
        # Each attempt owns its CSV; six seasonal groups cannot silently mix runs.
        directory = os.path.join(getattr(self.args, 'results_root', './results'), setting)
        os.makedirs(directory, exist_ok=True)
        table = pd.DataFrame(predictions[:, :, 0], index=dataset.ids,
                             columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        table.index.name = 'id'
        handle, temporary = tempfile.mkstemp(prefix='.forecast.', dir=directory)
        try:
            with os.fdopen(handle, 'w', encoding='utf-8', newline='') as stream:
                table.to_csv(stream)
            os.replace(temporary, os.path.join(directory, f'{self.args.seasonal_patterns}_forecast.csv'))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        extra['forecast_csv'] = f'{self.args.seasonal_patterns}_forecast.csv'
        result = save_evaluation(self.args, setting, metrics, predictions, targets, extra=extra)
        print(', '.join(f'{name}:{value}' for name, value in metrics.items()))
        return result
