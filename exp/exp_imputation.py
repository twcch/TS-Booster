import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import evaluation_stats, metric_spaces, plot_directory
from exp.common.runtime import load_checkpoint, precision_context, save_evaluation
from exp.common.training import WeightedLoss, require_same_shape, synchronize, train_model
from utils.tools import visual


def imputation_mask(shape, rate, device, generator=None, target_only=False):
    """Sample a mask, keeping context and a hidden target in every sample.

    Evaluation uses a local CPU generator so masks survive changes to training
    RNG use or backend. This constrained Bernoulli policy is recorded in results.
    """
    if not 0 < rate < 1:
        raise ValueError('mask_rate must be strictly between 0 and 1.')
    if len(shape) != 3 or shape[0] < 1 or shape[1] < 2 or shape[2] < 1:
        raise ValueError('Imputation requires [B,T,C] with T >= 2.')
    random_device = torch.device('cpu') if generator is not None else device
    values = torch.rand(shape, device=random_device, generator=generator)
    mask = values > rate
    # A feature with no observed context makes several imputation models divide by zero.
    observed_index = values.argmax(dim=1, keepdim=True)
    mask.scatter_(1, observed_index, True)
    target_start = shape[2] - 1 if target_only else 0
    for sample in range(shape[0]):
        selected = mask[sample, :, target_start:]
        if bool(selected.all()):
            eligible = torch.ones_like(selected)
            local_observed = observed_index[sample, :, target_start:]
            eligible.scatter_(0, local_observed, False)
            scores = values[sample, :, target_start:].masked_fill(~eligible, float('inf'))
            position = int(scores.reshape(-1).argmin().item())
            row, column = divmod(position, selected.size(1))
            mask[sample, row, target_start + column] = False
    return mask.to(device=device, dtype=torch.float32)


class Exp_Imputation(Exp_Basic):
    def _build_model(self):
        if not 0 < self.args.mask_rate < 1 or self.args.seq_len < 2:
            raise ValueError('Imputation requires 0 < mask_rate < 1 and seq_len >= 2.')
        return super()._build_model()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _mask_generator(self, split):
        seed = getattr(self.args, 'mask_eval_seed', None)
        if seed is None:
            seed = getattr(self.args, 'rand_seed', 2021) + 10000
        return torch.Generator(device='cpu').manual_seed(int(seed) + (1 if split == 'test' else 0))

    def _masked_inputs(self, batch, generator=None):
        x, _, x_mark, _ = batch
        x, x_mark = x.float().to(self.device), x_mark.float().to(self.device)
        mask = imputation_mask(x.shape, self.args.mask_rate, self.device, generator,
                               target_only=self.args.features == 'MS')
        return x, x_mark, mask

    def _masked_targets(self, output, x, mask):
        first = -1 if self.args.features == 'MS' else 0
        if not torch.is_tensor(output):
            raise TypeError('The selected model does not return an imputation tensor.')
        output, target, mask = output[:, :, first:], x[:, :, first:], mask[:, :, first:]
        require_same_shape(output, target)
        return output, target, mask

    def _batch_loss(self, batch, criterion, generator=None):
        x, x_mark, mask = self._masked_inputs(batch, generator)
        output = self.model(x.masked_fill(mask == 0, 0), x_mark, None, None, mask)
        output, target, mask = self._masked_targets(output, x, mask)
        hidden = mask == 0
        return criterion(output[hidden], target[hidden]), int(hidden.sum().item())

    def vali(self, vali_data, vali_loader, criterion, split='val'):
        total = WeightedLoss()
        generator = self._mask_generator(split)
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for batch in vali_loader:
                    with precision_context(self.args, self.device):
                        loss, count = self._batch_loss(batch, criterion, generator)
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
            diagnostic = lambda: self.vali(test_data, test_loader, criterion, split='test')
        return train_model(self, setting, train_loader,
                           lambda batch: self._batch_loss(batch, criterion),
                           lambda: self.vali(val_data, val_loader, criterion),
                           diagnostic=diagnostic)

    def test(self, setting, test=0, train_time=None):
        dataset, loader = self._get_data('test')
        if test:
            load_checkpoint(self.model, self.args, setting, self.device)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        predictions, targets, masks = [], [], []
        forward_s, count = 0.0, 0
        generator = self._mask_generator('test')
        self.model.eval()
        with torch.inference_mode():
            for batch in loader:
                x, x_mark, mask = self._masked_inputs(batch, generator)
                inputs = x.masked_fill(mask == 0, 0)
                synchronize(self.device)
                before = time.perf_counter()
                with precision_context(self.args, self.device):
                    output = self.model(inputs, x_mark, None, None, mask)
                synchronize(self.device)
                forward_s += time.perf_counter() - before
                count += x.size(0)
                output, target, mask = self._masked_targets(output, x, mask)
                predictions.append(output.float().cpu().numpy())
                targets.append(target.float().cpu().numpy())
                masks.append(mask.cpu().numpy())
        if not predictions:
            raise ValueError('No imputation test samples are available.')
        predictions, targets, masks = [np.concatenate(values) for values in (predictions, targets, masks)]
        predictions, targets, metrics, extra = metric_spaces(
            self.args, dataset, predictions, targets, mask=masks)
        if not getattr(self.args, 'no_plots', False):
            directory = plot_directory(self.args, setting)
            for index in range(0, count, max(1, self.args.batch_size * 20)):
                filled = np.where(masks[index, :, -1] != 0, targets[index, :, -1],
                                  predictions[index, :, -1])
                visual(targets[index, :, -1], filled, os.path.join(directory, f'{index}.pdf'))
        extra.update(evaluation_stats(self, train_time, forward_s, count, time.perf_counter() - started))
        extra.update({'mask': masks, 'mask_rate': self.args.mask_rate,
                      'evaluation_scope': 'loader, transfers, imputation forward, inverse, masked metrics, sampled plots; excludes artifact save',
                      'mask_eval_seed': int(generator.initial_seed()),
                      'mask_policy': 'Bernoulli; at least one observed value per feature and hidden target per sample',
                      'effective_mask_rate': float((masks == 0).mean()),
                      'effective_loss_elements': int((masks == 0).sum())})
        result = save_evaluation(self.args, setting, metrics, predictions, targets, extra=extra)
        print(', '.join(f'{name}:{value}' for name, value in metrics.items()))
        return result
