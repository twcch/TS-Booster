import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import evaluation_stats
from exp.common.runtime import load_checkpoint, precision_context, save_evaluation
from exp.common.training import WeightedLoss, synchronize, train_model


class Exp_Classification(Exp_Basic):
    def _build_model(self):
        # TEST is never read to resolve architecture or select a checkpoint.
        train_data, _ = self._get_data('TRAIN')
        val_data, _ = self._get_data('VAL')
        self.args.seq_len = max(train_data.max_seq_len, val_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        return super()._build_model()

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def _classification_batch(self, batch):
        x, label, padding = batch
        x, padding = x.float().to(self.device), padding.float().to(self.device)
        label = label.long().reshape(-1).to(self.device)
        output = self.model(x, padding, None, None)
        if not torch.is_tensor(output) or output.ndim != 2 or output.size(0) != label.numel():
            raise ValueError('Classification requires logits [B,classes] and labels [B].')
        return output, label

    def _batch_loss(self, batch, criterion):
        output, label = self._classification_batch(batch)
        return criterion(output, label), label.numel()

    def vali(self, vali_data, vali_loader, criterion):
        loss = WeightedLoss()
        correct = 0
        previous_mode = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for batch in vali_loader:
                    with precision_context(self.args, self.device):
                        output, label = self._classification_batch(batch)
                        batch_loss = criterion(output, label)
                    loss.update(batch_loss, label.numel())
                    correct += int((output.argmax(dim=1) == label).sum().item())
            return loss.mean(), correct / loss.count
        finally:
            self.model.train(previous_mode)

    def train(self, setting):
        _, train_loader = self._get_data('TRAIN')
        val_data, val_loader = self._get_data('VAL')
        criterion = self._select_criterion()
        diagnostic = None
        if getattr(self.args, 'diagnostic_test', False):
            test_data, test_loader = self._get_data('TEST')
            diagnostic = lambda: self.vali(test_data, test_loader, criterion)
        return train_model(self, setting, train_loader,
                           lambda batch: self._batch_loss(batch, criterion),
                           lambda: self.vali(val_data, val_loader, criterion),
                           select_metric=lambda validation: -validation[1],
                           diagnostic=diagnostic, clip_grad=4.0, adjust_lr=False)

    def test(self, setting, test=0, train_time=None):
        _, loader = self._get_data('TEST')
        if test:
            load_checkpoint(self.model, self.args, setting, self.device)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        synchronize(self.device)
        started = time.perf_counter()
        predictions, targets, forward_s = [], [], 0.0
        self.model.eval()
        with torch.inference_mode():
            for batch in loader:
                # Measure forward separately from host/device transfers.
                x, label, padding = batch
                x, padding = x.float().to(self.device), padding.float().to(self.device)
                label = label.long().reshape(-1)
                synchronize(self.device)
                before = time.perf_counter()
                with precision_context(self.args, self.device):
                    output = self.model(x, padding, None, None)
                synchronize(self.device)
                forward_s += time.perf_counter() - before
                if not torch.is_tensor(output) or output.ndim != 2 or output.size(0) != label.numel():
                    raise ValueError('Classification requires logits [B,classes] and labels [B].')
                predictions.append(output.argmax(dim=1).cpu().numpy())
                targets.append(label.cpu().numpy())
        if not predictions:
            raise ValueError('No classification test samples are available.')
        predictions, targets = np.concatenate(predictions), np.concatenate(targets)
        metrics = {'accuracy': float(np.mean(predictions == targets))}
        extra = evaluation_stats(self, train_time, forward_s, targets.size,
                                 time.perf_counter() - started)
        extra['evaluation_scope'] = 'loader, transfers, logits, accuracy; excludes artifact save'
        extra['evaluation_protocol'] = 'TRAIN stratified holdout selection; TEST final evaluation'
        result = save_evaluation(self.args, setting, metrics, predictions, targets, extra=extra)
        print(f'accuracy:{metrics["accuracy"]}')
        return result
