from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import ForecastEvaluationMixin, forecast_batch, forecast_targets, last_observed
from exp.common.runtime import precision_context, unwrap_model
from exp.common.training import WeightedLoss, build_optimizer, train_model
from utils.forecast_losses import apply_criterion, build_forecast_criterion
import torch
import torch.nn.functional as F


class Exp_Long_Term_Forecast(ForecastEvaluationMixin, Exp_Basic):
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return build_optimizer(self.model, self.args)

    def _select_criterion(self):
        return build_forecast_criterion(self.args)

    def _batch_loss(self, batch, criterion, training=False):
        """--loss on the forecast; in training, plus the model's auxiliary terms if it defines them
        and --train_mse_weight * MSE(forecast, target).

        Validation and checkpoint selection use the criterion alone, as in the
        original TS-Booster trainer.
        """
        x, y, x_mark, y_mark, decoder = forecast_batch(self, batch)
        model = unwrap_model(self.model)
        with_aux = training and getattr(model, 'has_auxiliary_loss', False)
        if with_aux:
            outputs, aux = self.model(x, x_mark, decoder, y_mark, return_aux=True)
        else:
            outputs = self.model(x, x_mark, decoder, y_mark)
        prediction, target = forecast_targets(self.args, outputs, y)
        loss = apply_criterion(criterion, prediction, target, last_observed(self.args, x))
        if with_aux:
            extra = model.auxiliary_loss(aux, target)
            if extra is not None:
                loss = loss + extra
        mse_weight = getattr(self.args, 'train_mse_weight', 0.0)
        if training and mse_weight:
            loss = loss + mse_weight * F.mse_loss(prediction, target)
        return loss, target.numel()

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
                           lambda batch: self._batch_loss(batch, criterion, training=True),
                           lambda: self.vali(val_data, val_loader, criterion),
                           diagnostic=diagnostic)
