import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import run
from exp.common.runtime import (TrainResult, atomic_save_checkpoint, canonical_config,
                                config_hash, load_checkpoint, prepare_run, resolve_device,
                                save_evaluation)


def arguments(tmp_path):
    args = run.build_parser().parse_args([
        '--task_name', 'long_term_forecast', '--is_training', '1', '--model_id', 'test',
        '--model', 'DLinear', '--data', 'custom', '--root_path', str(tmp_path),
        '--data_path', 'market.csv', '--results_root', str(tmp_path / 'results'),
        '--checkpoints', str(tmp_path / 'weights'), '--no_use_gpu', '--no_plots'])
    resolve_device(args)
    return args


def test_cpu_override_and_unavailable_backend(tmp_path):
    args = arguments(tmp_path)
    assert args.device == args.runtime_device == torch.device('cpu')
    args.use_gpu, args.gpu_type = True, 'cuda'
    with patch('torch.cuda.is_available', return_value=False), pytest.raises(ValueError, match='CUDA'):
        resolve_device(args)
    args.gpu_type, args.use_multi_gpu = 'cpu', True
    with pytest.raises(ValueError, match='CUDA'):
        resolve_device(args)


def test_logical_cuda_indices_are_not_rewritten(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    args.use_gpu, args.gpu_type, args.gpu = True, 'cuda', 1
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '4,7')
    with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
        assert resolve_device(args) == torch.device('cuda:1')
    import os
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4,7'


def test_identity_changes_with_data_and_config_but_attempts_do_not_collide(tmp_path):
    (tmp_path / 'market.csv').write_text('date,Close\n2020-01-01,1\n')
    args, model = arguments(tmp_path), torch.nn.Linear(1, 1)
    first = prepare_run(args, model, 'legacy')
    identity, group = args.experiment_id, args.group_id
    second = prepare_run(args, model, 'legacy')
    assert first != second and args.experiment_id == identity
    args.rand_seed += 1
    prepare_run(args, model, 'legacy')
    assert args.group_id == group and args.experiment_id != identity
    args.learning_rate *= 2
    prepare_run(args, model, 'legacy')
    assert args.group_id != group
    before = args.experiment_id
    (tmp_path / 'market.csv').write_text('date,Close\n2020-01-01,2\n')
    prepare_run(args, model, 'legacy')
    assert args.experiment_id != before
    assert config_hash({'a': 1, 'b': 2}) == config_hash({'b': 2, 'a': 1})


def test_artifacts_mask_and_nonfinite_values_are_explicit(tmp_path):
    args = arguments(tmp_path)
    setting = prepare_run(args, torch.nn.Linear(1, 1), 'legacy')
    predictions = np.ones((2, 3, 1))
    record = save_evaluation(args, setting,
        {'mae': 0., 'mse': 0., 'rmse': 0., 'mape': np.nan, 'mspe': np.inf, 'r2': 1.},
        predictions, predictions, {'mask': np.zeros_like(predictions), 'metric_space': 'original'})
    directory = Path(args.results_root) / setting
    parsed = json.loads((directory / 'result.json').read_text())
    assert parsed['metrics']['mape'] is None and parsed['metric_states']['mspe'] == 'nonfinite'
    assert json.loads(Path(args.manifest_path).read_text())['status'] == 'complete'
    assert np.load(directory / 'metrics.npy').shape == (6,)
    np.testing.assert_array_equal(np.load(directory / 'mask.npy'), 0)
    with pytest.raises(ValueError, match='shapes'):
        save_evaluation(args, 'invalid', {}, np.ones(2), np.ones(3))


def test_checkpoint_path_and_legacy_dataparallel_prefix(tmp_path):
    args = arguments(tmp_path)
    source, target = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    atomic_save_checkpoint(source, args, 'trained')
    args.checkpoint_run = 'trained'
    load_checkpoint(target, args, 'new-evaluation', torch.device('cpu'))
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)
    torch.save({'module.' + key: value for key, value in source.state_dict().items()},
               Path(args.checkpoints) / 'trained' / 'checkpoint.pth')
    load_checkpoint(target, args, 'other', torch.device('cpu'))


@pytest.mark.parametrize('task', list(run.TASKS))
def test_runner_dispatches_trainresult_and_inference(tmp_path, task):
    events = []
    class Experiment:
        def __init__(self, args):
            self.args, self.model = args, torch.nn.Linear(1, 1)
        def train(self, setting):
            events.append('train')
            return TrainResult(self.model, 0.25)
        def test(self, setting, test=0, train_time=None):
            events.append(('test', test, train_time))
            save_evaluation(self.args, setting, {'mse': 0.}, np.ones(2), np.ones(2))
    fake = SimpleNamespace(**{run.TASKS[task][1]: Experiment})
    model = 'Chronos' if task == 'zero_shot_forecast' else 'TimesNet'
    argv = ['--task_name', task, '--is_training', str(int(task != 'zero_shot_forecast')),
            '--model_id', 'contract', '--model', model, '--data', {
                'short_term_forecast': 'm4', 'classification': 'UEA', 'anomaly_detection': 'PSM'}.get(task, 'custom'),
            '--root_path', str(tmp_path), '--results_root', str(tmp_path / 'results'),
            '--checkpoints', str(tmp_path / 'weights'), '--no_use_gpu', '--no_plots']
    original_import = __import__('importlib').import_module
    def importing(name, *args, **kwargs):
        return fake if name == run.TASKS[task][0] else original_import(name, *args, **kwargs)
    with patch('importlib.import_module', side_effect=importing):
        assert run.main(argv) == 0
    assert events == ([('test', 1, None)] if task == 'zero_shot_forecast' else ['train', ('test', 0, 0.25)])


def test_invalid_task_and_zero_shot_training_reject_before_import(tmp_path):
    args = arguments(tmp_path)
    args.task_name = 'typo'
    with pytest.raises(ValueError, match='未知'):
        run.validate_args(args)
    args.task_name = 'zero_shot_forecast'
    with pytest.raises(ValueError, match='僅支援'):
        run.validate_args(args)
