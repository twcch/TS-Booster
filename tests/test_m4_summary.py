import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from data_provider.m4 import M4Dataset, M4Meta
from exp.common.runtime import METRIC_ORDER, file_fingerprint, prepare_run, resolve_device, save_evaluation
from run import build_parser
from tools.summarize_m4 import main, summarize_m4
from utils.m4_summary import group_values, mase, smape_2
from utils.metrics import metric


def fixture(tmp_path):
    root = tmp_path / 'official'
    root.mkdir()
    info, histories, targets = [], [], []
    for season in M4Meta.seasonal_patterns:
        horizon, frequency = M4Meta.horizons_map[season], M4Meta.frequency_map[season]
        for index in range(2):
            identifier = f'{season[0]}{index + 1}'
            info.append({'M4id': identifier, 'SP': season, 'Frequency': frequency, 'Horizon': horizon})
            histories.append(np.arange(frequency + 6 + index, dtype=float) + len(info))
            targets.append(np.arange(horizon, dtype=float) + 30 + len(info))
    frame = pd.DataFrame(info)
    frame.to_csv(root / 'M4-info.csv', index=False)
    for name, rows in [('training', histories), ('test', targets)]:
        values = np.empty(len(rows), dtype=object)
        values[:] = rows
        np.savez(root / f'{name}.npz', values=values)
    naive = np.full((len(targets), max(M4Meta.horizons)), np.nan)
    for index, target in enumerate(targets):
        naive[index, :len(target)] = target + 2
    naive_frame = pd.DataFrame(naive, columns=[f'F{i + 1}' for i in range(naive.shape[1])])
    naive_frame.insert(0, 'id', frame.M4id)
    naive_frame.to_csv(root / 'submission-Naive2.csv', index=False)
    fingerprints = {path.name: file_fingerprint(path) for path in root.iterdir()}
    runs = []
    for season in M4Meta.seasonal_patterns:
        horizon, frequency = M4Meta.horizons_map[season], M4Meta.frequency_map[season]
        selected = frame.SP == season
        directory = tmp_path / f'{season}__attempt_fixture'
        directory.mkdir()
        group_targets = np.stack([target for target, use in zip(targets, selected) if use]).astype(np.float32)
        prediction = group_targets + 1
        for name, values in [('pred', prediction[..., None]), ('true', group_targets[..., None]), ('metrics', np.ones(6))]:
            np.save(directory / f'{name}.npy', values)
        table = pd.DataFrame(prediction, columns=[f'V{i + 1}' for i in range(horizon)])
        ids = list(frame.M4id[selected])
        table.insert(0, 'id', ids)
        csv_name = f'{season}_forecast.csv'
        table.to_csv(directory / csv_name, index=False)
        config = {'task_name': 'short_term_forecast', 'data': 'm4', 'model': 'DLinear',
                  'model_id': season, 'rand_seed': 12, 'learning_rate': .001, 'd_model': 8,
                  'seasonal_patterns': season, 'pred_len': horizon, 'seq_len': 2 * horizon,
                  'label_len': horizon, 'frequency_map': frequency, 'root_path': str(root)}
        identity = {'setting': directory.name, 'experiment_id': f'config-{season}',
                    'group_id': f'group-{season}', 'attempt_id': f'attempt-{season}'}
        artifacts = ['pred.npy', 'true.npy', 'metrics.npy', csv_name]
        metadata = dict(identity, status='complete', evaluation_version=2, config=config,
                        effective_config={'seq_len': 2 * horizon, 'pred_len': horizon, 'd_model': 8},
                        source_hash='consistent-source', data=fingerprints, versions={'numpy': '2.1.2'},
                        artifacts=artifacts + ['result.json'])
        result = dict(identity, status='complete', evaluation_version=2, config=config,
                      artifacts=artifacts, metrics={'mse': 1.},
                      details={'metric_space': 'original', 'evaluation_protocol': 'TRAIN tail holdout; official TEST final only',
                               'seasonal_patterns': season, 'forecast_csv': csv_name, 'series_ids': ids})
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        (directory / 'result.json').write_text(json.dumps(result))
        runs.append(directory)
    return root, runs


def change_record(directory, file_name, transform):
    path = directory / file_name
    record = json.loads(path.read_text())
    transform(record)
    path.write_text(json.dumps(record))


def test_explicit_six_attempts_stage_numeric_forecasts_and_never_overwrite(tmp_path):
    root, runs = fixture(tmp_path)
    seen_staging = []
    expected_groups = ['Yearly', 'Quarterly', 'Monthly', 'Others', 'Average']
    def factory(staging, official):
        assert Path(official) == root
        seen_staging.append(Path(staging))
        for run, season in zip(runs, M4Meta.seasonal_patterns):
            staged = pd.read_csv(Path(staging) / f'{season}_forecast.csv')
            assert list(staged.columns) == [f'V{i + 1}' for i in range(M4Meta.horizons_map[season])]
            np.testing.assert_allclose(staged.values, np.load(run / 'pred.npy')[:, :, 0])
        return Mock(evaluate=Mock(return_value=tuple(dict.fromkeys(expected_groups, value) for value in [1., .5, 2., 3.])))
    before = {path: path.read_bytes() for run in runs for path in run.iterdir()}
    output = tmp_path / 'summary.json'
    result = summarize_m4(list(reversed(runs)), root, output, summary_factory=factory)
    assert result['metrics']['owa']['Average'] == .5
    assert list(result['runs']) == M4Meta.seasonal_patterns
    assert output.is_file() and all(not path.exists() for path in seen_staging)
    assert before == {path: path.read_bytes() for path in before}
    with pytest.raises(FileExistsError, match='already exists'):
        summarize_m4(runs, root, output, summary_factory=Mock(side_effect=AssertionError('must not calculate')))


@pytest.mark.parametrize('mutation', ['seed', 'model', 'architecture', 'source', 'data', 'protocol', 'incomplete'])
def test_incompatible_or_incomplete_attempts_are_rejected_before_summary(tmp_path, mutation):
    root, runs = fixture(tmp_path)
    if mutation in {'seed', 'model', 'architecture'}:
        key, value = {'seed': ('rand_seed', 13), 'model': ('model', 'TimesNet'), 'architecture': ('d_model', 16)}[mutation]
        for file_name in ['metadata.json', 'result.json']:
            change_record(runs[-1], file_name, lambda record: record['config'].update({key: value}))
    elif mutation == 'protocol':
        change_record(runs[-1], 'result.json', lambda record: record['details'].update(evaluation_protocol='TEST early stopping'))
    elif mutation == 'source':
        change_record(runs[-1], 'metadata.json', lambda record: record.update(source_hash='different-source'))
    elif mutation == 'data':
        change_record(runs[-1], 'metadata.json', lambda record: record['data'].update({'test.npz': 'different-data'}))
    else:
        change_record(runs[-1], 'metadata.json', lambda record: record.update(status='running'))
    output, summary = tmp_path / 'summary.json', Mock()
    with pytest.raises(ValueError, match='Incompatible|not complete'):
        summarize_m4(runs, root, output, summary_factory=summary)
    summary.assert_not_called()
    assert not output.exists()


def test_six_distinct_seasons_are_required(tmp_path):
    root, runs = fixture(tmp_path)
    for selection in [runs[:-1], [*runs[:-1], runs[0]]]:
        with pytest.raises(ValueError, match='six distinct'):
            summarize_m4(selection, root, tmp_path / 'summary.json')
    for file_name in ['metadata.json', 'result.json']:
        change_record(runs[-1], file_name,
                      lambda record: record['config'].update(seasonal_patterns='Yearly', pred_len=6, seq_len=12,
                                                             label_len=6, frequency_map=1))
    change_record(runs[-1], 'result.json', lambda record: record['details'].update(seasonal_patterns='Yearly', forecast_csv='Yearly_forecast.csv'))
    # It must still pass attempt artifact completeness before the season check.
    (runs[-1] / 'Hourly_forecast.csv').rename(runs[-1] / 'Yearly_forecast.csv')
    for file_name in ['metadata.json', 'result.json']:
        change_record(runs[-1], file_name,
                      lambda record: record.update(artifacts=[name.replace('Hourly_forecast', 'Yearly_forecast')
                                                             for name in record['artifacts']]))
    with pytest.raises(ValueError, match='each of the six'):
        summarize_m4(runs, root, tmp_path / 'summary.json')


@pytest.mark.parametrize('field', ['forecast_ids', 'naive_ids', 'csv_values', 'targets', 'official_data'])
def test_official_ids_and_artifact_alignment_are_verified(tmp_path, field):
    root, runs = fixture(tmp_path)
    if field == 'forecast_ids':
        path = runs[0] / 'Yearly_forecast.csv'
        pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
    elif field == 'naive_ids':
        path = root / 'submission-Naive2.csv'
        table = pd.read_csv(path)
        table.iloc[[1, 0, *range(2, len(table))]].to_csv(path, index=False)
    elif field == 'csv_values':
        path = runs[0] / 'Yearly_forecast.csv'
        table = pd.read_csv(path)
        table.loc[0, 'V1'] += 1
        table.to_csv(path, index=False)
    elif field == 'targets':
        values = np.load(runs[0] / 'true.npy')
        values[0, 0, 0] += 1
        np.save(runs[0] / 'true.npy', values)
    else:
        path = root / 'M4-info.csv'
        path.write_text(path.read_text() + '\n')
    summary = Mock()
    with pytest.raises(ValueError, match='IDs|disagree|fingerprint'):
        summarize_m4(runs, root, tmp_path / 'summary.json', summary_factory=summary)
    summary.assert_not_called()


def test_real_summary_keeps_ragged_history_and_original_owa_formula(tmp_path):
    root, runs = fixture(tmp_path)
    result = summarize_m4(runs, root, tmp_path / 'summary.json')
    assert set(result['metrics']) == {'smape', 'owa', 'mape', 'mase'}
    assert result['metrics']['mase']['Yearly'] == 1.
    targets = np.load(runs[0] / 'true.npy')[:, :, 0]
    predicted = targets + 1
    expected = (1 / 2 + smape_2(predicted, targets).mean() / smape_2(targets + 2, targets).mean()) / 2
    assert result['metrics']['owa']['Yearly'] == round(float(expected), 3)
    assert result['metrics']['mase']['Average'] > 0
    history = np.empty(2, dtype=object)
    history[:] = [np.arange(5.), np.arange(7.)]
    grouped = group_values(history, np.array(['Yearly', 'Yearly']), 'Yearly')
    assert grouped.dtype == object and len(grouped[0]) == 5 and len(grouped[1]) == 7
    assert mase(np.array([3.]), np.array([1., 2., 3.]), np.array([2.]), 1) == 1.


def test_cli_requires_explicit_six_attempts(tmp_path):
    root, runs = fixture(tmp_path)
    assert main(['--runs', *map(str, runs), '--root_path', str(root), '--output', str(tmp_path / 'cli-summary.json')]) == 0


def test_undefined_mase_and_owa_are_reported_without_changing_denominator(tmp_path):
    root, runs = fixture(tmp_path)
    with np.load(root / 'training.npz', allow_pickle=True) as archive:
        history = archive['values']
    for index, values in enumerate(history):
        history[index] = np.ones_like(values)
    np.savez(root / 'training.npz', values=history)
    fingerprint = file_fingerprint(root / 'training.npz')
    for run in runs:
        change_record(run, 'metadata.json', lambda record: record['data'].update({'training.npz': fingerprint}))
    with np.errstate(divide='ignore', invalid='ignore'):
        result = summarize_m4(runs, root, tmp_path / 'constant-summary.json')
    assert result['metrics']['mase']['Average'] is None
    assert result['metrics']['owa']['Average'] is None
    assert result['metric_states']['mase']['Average'] == 'nonfinite'
    assert result['metric_states']['owa']['Average'] == 'nonfinite'
    assert np.isfinite(result['metrics']['smape']['Average'])


def test_production_runtime_artifacts_are_accepted_for_all_six_seasons(tmp_path):
    root, _ = fixture(tmp_path)
    official = M4Dataset.load(training=False, dataset_file=str(root))
    results_root = tmp_path / 'production-results'
    runs = []
    for season in M4Meta.seasonal_patterns:
        args = build_parser().parse_args([
            '--task_name', 'short_term_forecast', '--is_training', '1',
            '--model_id', f'm4-{season}', '--model', 'DLinear', '--data', 'm4',
            '--root_path', str(root), '--results_root', str(results_root),
            '--seasonal_patterns', season, '--features', 'S', '--enc_in', '1',
            '--dec_in', '1', '--c_out', '1', '--rand_seed', '12', '--no_use_gpu', '--no_plots'])
        args.requested_config = deepcopy(vars(args))
        resolve_device(args)
        # These are the same forced values used by Exp_Short_Term_Forecast.
        horizon = M4Meta.horizons_map[season]
        args.pred_len, args.seq_len, args.label_len = horizon, 2 * horizon, horizon
        args.frequency_map = M4Meta.frequency_map[season]
        model = torch.nn.Linear(1, 1)
        model.effective_config = {'seq_len': args.seq_len, 'label_len': args.label_len,
                                  'pred_len': args.pred_len, 'enc_in': 1, 'c_out': 1}
        setting = prepare_run(args, model, f'm4-production-{season}', iteration=3)
        directory = results_root / setting
        ids = list(map(str, official.ids[official.groups == season]))
        targets = np.stack(official.values[official.groups == season].tolist()).astype(np.float32)[..., None]
        predictions = targets + 1
        table = pd.DataFrame(predictions[:, :, 0], columns=[f'V{i + 1}' for i in range(horizon)])
        table.insert(0, 'id', ids)
        csv_name = f'{season}_forecast.csv'
        table.to_csv(directory / csv_name, index=False)
        result = save_evaluation(args, setting, dict(zip(METRIC_ORDER, metric(predictions, targets))),
                                 predictions, targets,
                                 extra={'metric_space': 'original',
                                        'evaluation_protocol': 'TRAIN tail holdout; official TEST final only',
                                        'seasonal_patterns': season, 'series_ids': ids, 'forecast_csv': csv_name})
        metadata = json.loads((directory / 'metadata.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        assert result['config'] == metadata['config'] == config
        assert result['config']['iteration'] == 3
        assert metadata['requested_config']['pred_len'] == 96
        assert metadata['config']['pred_len'] == horizon
        assert metadata['status'] == result['status'] == 'complete'
        runs.append(directory)
    summary = summarize_m4(runs, root, tmp_path / 'production-summary.json')
    assert summary['status'] == 'complete'
    assert summary['common']['config']['iteration'] == 3
    assert summary['metrics']['mase']['Yearly'] == 1.
    assert len(summary['runs']) == 6
