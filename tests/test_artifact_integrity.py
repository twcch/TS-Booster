"""Invalid numeric predictions cannot produce a successful result artifact."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from exp.common.evaluation import evaluation_stats, runtime_parameter_stats
from exp.common.runtime import save_evaluation


def arguments(tmp_path, task='long_term_forecast'):
    return SimpleNamespace(results_root=str(tmp_path), task_name=task)


@pytest.mark.parametrize('field', ['prediction', 'target'])
@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_arrays_fail_before_artifact_write(tmp_path, field, value):
    predictions, targets = np.ones((2, 3, 1)), np.ones((2, 3, 1))
    (predictions if field == 'prediction' else targets)[0, 0, 0] = value
    with pytest.raises(ValueError, match='finite'):
        save_evaluation(arguments(tmp_path), 'invalid', {'mse': np.nan}, predictions, targets)
    assert not (tmp_path / 'invalid').exists()


def test_imputation_observed_nan_is_not_a_dense_output_exception(tmp_path):
    prediction, target = np.ones((1, 2, 1)), np.ones((1, 2, 1))
    mask = np.array([[[0], [1]]])
    prediction[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match='outside an imputation mask'):
        save_evaluation(arguments(tmp_path, 'imputation'), 'invalid', {'mse': 0.0},
                        prediction, target, {'mask': mask})
    assert not (tmp_path / 'invalid').exists()


@pytest.mark.parametrize('mask', [np.ones((1, 1, 1)), np.array([[[0], [np.nan]]]),
                                 np.array([[[0], [2]]]), np.array([[[1], [1]]])])
def test_imputation_mask_contract_is_checked_before_write(tmp_path, mask):
    with pytest.raises(ValueError, match='mask'):
        save_evaluation(arguments(tmp_path, 'imputation'), 'invalid', {'mse': 0.0},
                        np.ones((1, 2, 1)), np.ones((1, 2, 1)), {'mask': mask})
    assert not (tmp_path / 'invalid').exists()


def test_undefined_metric_is_distinct_from_nonfinite_predictions(tmp_path):
    metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': np.nan, 'mspe': np.nan, 'r2': np.nan}
    save_evaluation(arguments(tmp_path), 'valid-arrays', metrics, np.zeros((1, 2, 1)), np.zeros((1, 2, 1)))
    result = json.loads((tmp_path / 'valid-arrays' / 'result.json').read_text())
    assert result['status'] == 'complete'
    assert result['metrics']['mape'] is None and result['metric_states']['mape'] == 'nonfinite'
    np.testing.assert_array_equal(np.load(tmp_path / 'valid-arrays' / 'pred.npy'), 0)


@pytest.mark.parametrize('array', [np.array(['a']), np.array([1 + 2j])])
def test_non_real_predictions_are_rejected(tmp_path, array):
    with pytest.raises(ValueError, match='real numeric'):
        save_evaluation(arguments(tmp_path), 'invalid', {}, array, array)
    assert not (tmp_path / 'invalid').exists()


def test_provider_pipeline_parameters_and_actual_dtype_are_counted():
    module = torch.nn.Linear(2, 3).to(dtype=torch.bfloat16)
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Provider pipelines are plain objects and do not register their model.
            self.model = SimpleNamespace(model=module)
    wrapper = Wrapper()
    assert len(list(wrapper.parameters())) == 0
    stats = runtime_parameter_stats(wrapper)
    assert stats['params'] == stats['trainable_params'] == 9
    assert stats['floating_parameter_dtypes'] == ['torch.bfloat16']
    # The same underlying parameter cannot be counted twice via registered aliases.
    wrapper.registered_model = module
    assert runtime_parameter_stats(wrapper)['params'] == 9
    experiment = SimpleNamespace(model=wrapper, device=torch.device('cpu'), args=SimpleNamespace(use_amp=True))
    details = evaluation_stats(experiment, None, 0.1, 2, 0.2)
    assert details['floating_parameter_dtypes'] == ['torch.bfloat16']
    assert details['autocast_enabled'] is True and details['autocast_dtype'] == 'torch.bfloat16'
