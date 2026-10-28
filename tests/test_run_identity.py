import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

import run
from exp.exp_basic import Exp_Basic, LazyModelDict
from exp.common.runtime import TrainResult, atomic_save_checkpoint, prepare_run, save_evaluation
from tests.test_runtime import arguments


def test_standalone_identity_contains_checkpoint_content(tmp_path):
    args = arguments(tmp_path)
    model = torch.nn.Linear(1, 1)
    args.is_training, args.checkpoint_run = 0, 'trained'
    atomic_save_checkpoint(model, args, 'trained')
    prepare_run(args, model, 'standalone')
    before, before_group = args.experiment_id, args.group_id
    metadata = json.loads(Path(args.manifest_path).read_text())
    assert metadata['checkpoint_source']['sha256']
    with torch.no_grad():
        model.weight.add_(1)
    atomic_save_checkpoint(model, args, 'trained')
    prepare_run(args, model, 'standalone')
    assert args.experiment_id != before and args.group_id != before_group


def test_itr_records_actual_seed_and_keeps_requested_config(tmp_path):
    observed = []
    class Experiment:
        def __init__(self, args):
            self.args, self.model = args, torch.nn.Linear(1, 1)
            observed.append((args.rand_seed, torch.initial_seed()))
        def train(self, setting):
            return TrainResult(self.model, .1)
        def test(self, setting, train_time=None):
            save_evaluation(self.args, setting, {'mse': 0.}, np.zeros(2), np.zeros(2))
    fake = SimpleNamespace(Exp_Long_Term_Forecast=Experiment)
    argv = ['--task_name', 'long_term_forecast', '--is_training', '1', '--model_id', 'test',
            '--model', 'DLinear', '--data', 'custom', '--root_path', str(tmp_path),
            '--results_root', str(tmp_path / 'results'), '--no_use_gpu', '--itr', '2', '--rand_seed', '9']
    original_import = __import__('importlib').import_module
    def importing(name, *args, **kwargs):
        return fake if name == run.TASKS['long_term_forecast'][0] else original_import(name, *args, **kwargs)
    with patch('importlib.import_module', side_effect=importing):
        assert run.main(argv) == 0
    records = [json.loads(p.read_text()) for p in (tmp_path / 'results').glob('*/metadata.json')]
    assert observed == [(9, 9), (10, 10)]
    assert {r['config']['rand_seed'] for r in records} == {9, 10}
    assert {r['requested_config']['rand_seed'] for r in records} == {9}
    assert len({r['group_id'] for r in records}) == 1
    for path in (tmp_path / 'results').glob('*/result.json'):
        result = json.loads(path.read_text())
        metadata = json.loads((path.parent / 'metadata.json').read_text())
        assert result['config'] == metadata['config']


def test_missing_explicit_checkpoint_rejects_before_model_import(tmp_path):
    argv = ['--task_name', 'long_term_forecast', '--is_training', '0', '--model_id', 'test',
            '--model', 'DLinear', '--data', 'custom', '--no_use_gpu',
            '--checkpoints', str(tmp_path), '--checkpoint_run', 'missing']
    with patch('importlib.import_module') as importing, pytest.raises(SystemExit):
        run.main(argv)
    assert not any(call.args[0].startswith('exp.exp_') for call in importing.call_args_list)


def test_foundation_dtype_is_preserved_by_base_builder(tmp_path):
    args = arguments(tmp_path)
    args.model = 'TimeMoE'
    base = Exp_Basic.__new__(Exp_Basic)
    base.args = args
    base.model_dict = LazyModelDict({'TimeMoE': 'provider'})
    base.model_dict['TimeMoE'] = lambda _: torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    assert next(base._build_model().parameters()).dtype == torch.bfloat16


def test_csv_target_index_matches_reordered_target(tmp_path):
    args = arguments(tmp_path)
    args.features, args.enc_in, args.target_channel_index = 'MS', 2, 0
    with pytest.raises(ValueError, match='target_channel_index'):
        run.validate_args(args)


def test_foundation_multi_gpu_rejected_before_provider_loading(tmp_path):
    args = arguments(tmp_path)
    args.model, args.task_name, args.is_training = 'TimeMoE', 'zero_shot_forecast', 0
    args.use_multi_gpu = True
    with pytest.raises(ValueError, match='單裝置'):
        run.validate_args(args)


@pytest.mark.parametrize('task,data', [('classification', 'custom'), ('anomaly_detection', 'custom'),
                                      ('imputation', 'm4'), ('long_term_forecast', 'UEA')])
def test_invalid_task_dataset_pair_has_explicit_error(tmp_path, task, data):
    args = arguments(tmp_path)
    args.task_name, args.data = task, data
    with pytest.raises(ValueError, match='資料類型'):
        run.validate_args(args)
