import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.batch_runner import build_commands, command_key, manifest_lock, run_commands, verify_artifacts
from tools.parse_result_to_xlsx import export_legacy, parse, parse_results, summarize


def test_paired_windows_and_explicit_workers():
    config = {'data_names': ['A'], 'windows': [[12, 6], [24, 12]], 'pred_lens': [1],
              'rand_seeds': [1], 'models': ['DLinear'],
              'shared': {'task_name': 'long_term_forecast', 'is_training': 1, 'data': 'custom', 'num_workers': 0}}
    commands = build_commands(config, python='/tmp/python with spaces')
    assert len(commands) == 2
    assert commands[0][0] == '/tmp/python with spaces'
    for command, (seq, label) in zip(commands, config['windows']):
        assert command[command.index('--seq_len') + 1] == str(seq)
        assert command[command.index('--label_len') + 1] == str(label)
        assert command[command.index('--num_workers') + 1] == '0'


def test_resume_identity_reads_csv_and_code(tmp_path):
    data = tmp_path / 'A.csv'
    data.write_text('1')
    command = ['python', 'run.py', '--root_path', str(tmp_path), '--data_path', 'A.csv', '--model', 'DLinear']
    first = command_key(command, 'code-1', {'torch': '2.6'})
    reordered = command[:2] + command[6:] + command[2:6]
    assert command_key(reordered, 'code-1', {'torch': '2.6'}) == first
    assert command_key(command, 'code-2', {'torch': '2.6'}) != first
    data.write_text('2')
    assert command_key(command, 'code-1', {'torch': '2.6'}) != first


def test_dry_run_does_not_write_and_manifest_has_cross_process_lock(tmp_path):
    directory = tmp_path / 'batch'
    assert run_commands([['python', 'run.py']], directory, dry_run=True) == 0
    assert not directory.exists()
    with manifest_lock(directory):
        with pytest.raises(RuntimeError, match='另一個'):
            with manifest_lock(directory):
                pass
    with manifest_lock(directory):
        pass


def test_artifact_verification_rejects_missing_or_mismatched_results(tmp_path):
    metadata = {'status': 'complete', 'experiment_id': 'same', 'attempt_id': 'attempt',
                'group_id': 'group', 'setting': 'setting'}
    result = dict(metadata, artifacts=['pred.npy', 'true.npy', 'metrics.npy'])
    (tmp_path / 'metadata.json').write_text(json.dumps(metadata))
    (tmp_path / 'result.json').write_text(json.dumps(result))
    for name in ('pred', 'true', 'metrics'):
        np.save(tmp_path / f'{name}.npy', np.ones((2, 3)))
    assert verify_artifacts(tmp_path)
    (tmp_path / 'true.npy').unlink()
    assert not verify_artifacts(tmp_path)
    np.save(tmp_path / 'true.npy', np.ones((1, 3)))
    assert not verify_artifacts(tmp_path)


def test_legacy_parser_keeps_underscores_and_configuration_groups(tmp_path):
    template = 'long_term_forecast_AAPL-2013-2023_96_48_7_Nonstationary_Transformer_custom_ftMS_sl96_ll48_pl7_dm{}_nh2_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_seed{}_0'
    rows = [(32, 1, 1), (32, 1, 3), (64, 1, 10), (32, 2, 5)]
    path = tmp_path / 'legacy.txt'
    path.write_text('\n'.join(template.format(width, seed) + f'\nmse:{mse},mae:1,rmse:1,mape:nan,r2:-inf' for width, seed, mse in rows))
    frame = parse(path)
    assert set(frame.model) == {'Nonstationary_Transformer'}
    assert frame.MAPE.isna().all()
    with pytest.warns(UserWarning, match='repeated'):
        summary = summarize(frame)
    assert len(summary) == 2
    group = summary.loc[summary.n_seeds == 2].iloc[0]
    assert group.MSE_mean == 4. and group.n_attempts == 3


def test_json_reports_keep_metric_scale_and_unique_seeds(tmp_path):
    for index, (seed, mse) in enumerate([(1, 1.), (1, 3.), (2, 5.)]):
        directory = tmp_path / str(index)
        directory.mkdir()
        metadata = {'status': 'complete', 'experiment_id': f'config-seed-{seed}',
                    'group_id': 'same-config', 'attempt_id': str(index),
                    'finished_at': f'2026-10-04T12:00:0{index}Z'}
        config = {'rand_seed': seed, 'seq_len': 96, 'pred_len': 7, 'model': 'DLinear',
                  'data': 'custom', 'data_path': 'A.csv', 'task_name': 'long_term_forecast'}
        record = {'status': 'complete', 'experiment_id': metadata['experiment_id'],
                  'config': config, 'group_id': 'same-config', 'attempt_id': str(index),
                  'metrics': {'mse': mse, 'mae': 1., 'rmse': 1., 'mape': .1, 'r2': .5},
                  'details': {'metric_space': 'normalized', 'metrics_original': {'mse': 100 * mse, 'mae': 10., 'rmse': 10., 'mape': .01, 'r2': .5}}}
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        (directory / 'result.json').write_text(json.dumps(record))
    frame = parse_results(tmp_path)
    with pytest.warns(UserWarning, match='repeated'):
        summary = summarize(frame)
    assert summary.iloc[0].n_seeds == 2 and summary.iloc[0].MSE_mean == 4.
    original = parse_results(tmp_path, 'original')
    assert original.MSE.iloc[0] == 100. and set(original.metric_scale) == {'original'}


def _write_report_result(directory, *, attempt, finished_at, mse, seed=1):
    directory.mkdir()
    identity = {'experiment_id': f'config-seed-{seed}', 'group_id': 'same-config', 'attempt_id': attempt}
    metadata = dict(identity, status='complete', finished_at=finished_at,
                    legacy_setting=f'long_term_forecast_fixture_seed{seed}_0')
    config = {'rand_seed': seed, 'seq_len': 96, 'pred_len': 7, 'model': 'DLinear',
              'data': 'custom', 'data_path': 'A.csv', 'task_name': 'long_term_forecast'}
    result = dict(identity, status='complete', config=config, metrics={'mse': mse},
                  details={'metric_space': 'original'})
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    (directory / 'result.json').write_text(json.dumps(result))
    return metadata, result


@pytest.mark.parametrize('identifier', ['experiment_id', 'group_id', 'attempt_id'])
def test_json_report_rejects_mismatched_metadata_identifiers(tmp_path, identifier):
    directory = tmp_path / 'attempt'
    metadata, _ = _write_report_result(directory, attempt='one', finished_at='2026-10-04T12:00:00Z', mse=1.)
    metadata[identifier] = 'other'
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.warns(UserWarning, match='identity mismatch'):
        assert parse_results(tmp_path).empty


def test_report_latest_attempt_uses_completion_timestamp_not_folder_order(tmp_path):
    _write_report_result(tmp_path / 'aaa-new', attempt='new', finished_at='2026-10-05T12:00:00Z', mse=1.)
    _write_report_result(tmp_path / 'zzz-old', attempt='old', finished_at='2026-10-04T12:00:00Z', mse=9.)
    frame = parse_results(tmp_path)
    assert frame.metadata_verified.all()
    with pytest.warns(UserWarning, match='repeated'):
        summary = summarize(frame)
    assert summary.iloc[0].MSE_mean == 1.
    assert summary.iloc[0].n_seeds == 1 and summary.iloc[0].n_attempts == 2


def test_legacy_export_accepts_single_result_and_rejects_mismatched_pair(tmp_path):
    directory = tmp_path / 'attempt'
    metadata, _ = _write_report_result(directory, attempt='one', finished_at='2026-10-04T12:00:00Z', mse=1.)
    target = tmp_path / 'nested' / 'legacy.txt'
    export_legacy(directory / 'result.json', target)
    assert 'mse:1.0' in target.read_text()
    metadata['experiment_id'] = 'other'
    (directory / 'metadata.json').write_text(json.dumps(metadata))
    with pytest.warns(UserWarning, match='identity mismatch'):
        export_legacy(directory / 'result.json', target)
    assert target.read_text() == ''
