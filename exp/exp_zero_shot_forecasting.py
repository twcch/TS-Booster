from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.common.evaluation import ForecastEvaluationMixin


class Exp_Zero_Shot_Forecast(ForecastEvaluationMixin, Exp_Basic):
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def train(self, setting):
        raise ValueError('zero_shot_forecast supports pretrained inference only; use --is_training 0.')
