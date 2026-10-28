import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import run
from tools.batch_runner import build_commands, command_key, executable_versions, run_commands, sweep_main, verify_artifacts


def sweep(tmp_path):
    return {'data_names': ['market'], 'windows': [[4, 2]], 'pred_lens': [1],
            'rand_seeds': [7], 'models': ['DLinear'], 'shared': {
                'task_name': 'long_term_forecast', 'is_training': 1, 'data': 'custom',
                'root_path': str(tmp_path), 'features': 'MS', 'target': 'Close', 'freq': 'd',
                'enc_in': 2, 'dec_in': 2, 'c_out': 1, 'moving_avg': 3, 'train_epochs': 1,
                'batch_size': 8, 'patience': 1, 'num_workers': 0, 'use_gpu': False,
                'no_plots': True, 'results_root': str(tmp_path / 'results'),
                'checkpoints': str(tmp_path / 'weights'), 'test_results_root': str(tmp_path / 'plots')}}


def test_boolean_flags_and_lists_follow_actual_parser(tmp_path):
    config = sweep(tmp_path)
    config['shared'].update(distil=False, p_hidden_dims=[8, 4], use_amp=False)
    command = build_commands(config)[0]
    parsed = run.build_parser().parse_args(command[3:])
    assert parsed.use_gpu is False and parsed.distil is False and parsed.use_amp is False
    assert parsed.p_hidden_dims == [8, 4]


def test_config_file_survives_unspecified_cli_defaults(tmp_path, capsys):
    config = sweep(tmp_path)
    config['shared'].update(inverse=True, no_plots=True, use_dtw=True, num_workers=3, gpu_type='cuda')
    filename = tmp_path / 'sweep.json'
    filename.write_text(json.dumps(config))
    assert sweep_main({}, ['--config', str(filename), '--print_config']) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed == config
    sweep_main({}, ['--config', str(filename), '--print_config', '--no-inverse', '--plots', '--num_workers', '0'])
    parsed = json.loads(capsys.readouterr().out)
    assert parsed['shared']['inverse'] is False and parsed['shared']['no_plots'] is False
    assert parsed['shared']['num_workers'] == 0 and parsed['shared']['use_dtw'] is True


def test_dataset_directory_and_child_cwd_change_resume_identity(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    data = project / 'observations'
    data.mkdir(parents=True)
    np.save(data / 'train.npy', np.zeros((3, 2)))
    monkeypatch.setattr('tools.batch_runner.PROJECT_ROOT', project)
    argv = [sys.executable, 'run.py', '--data', 'PSM', '--root_path', 'observations']
    before = command_key(argv, 'source')
    monkeypatch.chdir(tmp_path)
    assert command_key(argv, 'source') == before
    np.save(data / 'train.npy', np.ones((3, 2)))
    assert command_key(argv, 'source') != before


def test_child_environment_and_venv_path_are_fingerprinted(tmp_path):
    alias = tmp_path / 'python-alias'
    alias.symlink_to(sys.executable)
    versions = executable_versions(str(alias))
    assert 'torch' in versions and versions['python'] == sys.version.split()[0]
    # Symlink paths remain distinct: their parent can select different pyvenv.cfg.
    assert command_key([str(alias)], 'source', versions) != command_key([sys.executable], 'source', versions)
    assert command_key([str(alias)], 'source', versions | {'torch': 'changed'}) != command_key([str(alias)], 'source', versions)


def test_real_batch_resume_rerun_data_version_and_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    frame = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=40),
                          'Open': np.linspace(1, 4, 40), 'Close': np.linspace(2, 7, 40)})
    filename = tmp_path / 'market.csv'
    frame.to_csv(filename, index=False)
    commands = build_commands(sweep(tmp_path))
    directory = tmp_path / 'batch'
    assert run_commands(commands, directory, timeout=60) == 0
    manifest_path = directory / 'manifest.json'
    entry = next(iter(json.loads(manifest_path.read_text())['experiments'].values()))
    assert entry['status'] == 'complete' and entry['attempts'] == 1
    first = Path(entry['result_directories'][0])
    assert verify_artifacts(first)
    assert run_commands(commands, directory, timeout=60) == 0
    assert next(iter(json.loads(manifest_path.read_text())['experiments'].values()))['attempts'] == 1
    (first / 'metrics.npy').unlink()
    assert run_commands(commands, directory, timeout=60) == 0
    entry = next(iter(json.loads(manifest_path.read_text())['experiments'].values()))
    assert entry['attempts'] == 2 and Path(entry['result_directories'][0]) != first
    frame.loc[0, 'Close'] += 1
    frame.to_csv(filename, index=False)
    assert run_commands(commands, directory, timeout=60) == 0
    assert len(json.loads(manifest_path.read_text())['experiments']) == 2
    invalid = commands[0] + ['--task_name', 'invalid']
    assert run_commands([invalid], directory, timeout=60) == 1
    assert any(item['status'] == 'failed' for item in json.loads(manifest_path.read_text())['experiments'].values())
    with pytest.raises(ValueError, match='duplicate'):
        run_commands([commands[0], commands[0]], directory)


def test_available_cpus_follow_affinity_and_the_cgroup_quota(monkeypatch):
    from tools import batch_runner
    files = {}

    def read_text(path, *args, **kwargs):
        if str(path) not in files:
            raise FileNotFoundError(path)
        return files[str(path)]

    monkeypatch.setattr(batch_runner.os, 'sched_getaffinity', lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(batch_runner.Path, 'read_text', read_text)
    assert batch_runner.available_cpus() == 64
    files['/sys/fs/cgroup/cpu.max'] = '1536000 100000\n'  # a Vast.ai container: 15.36 of 64 CPUs
    assert batch_runner.available_cpus() == 15
    files['/sys/fs/cgroup/cpu.max'] = 'max 100000\n'
    assert batch_runner.available_cpus() == 64
    files.clear()
    files.update({'/sys/fs/cgroup/cpu/cpu.cfs_quota_us': '400000', '/sys/fs/cgroup/cpu/cpu.cfs_period_us': '100000'})
    assert batch_runner.available_cpus() == 4


def test_default_parallel_is_one_run_per_cpu_and_at_most_32_per_gpu(monkeypatch):
    from tools import batch_runner
    monkeypatch.setattr(batch_runner, 'available_cpus', lambda: 15)
    assert batch_runner.default_parallel() == 15
    monkeypatch.setattr(batch_runner, 'available_cpus', lambda: 80)
    assert batch_runner.default_parallel() == 32
    assert batch_runner.default_parallel(['0', '1']) == 64


def test_progress_line_reports_each_finished_run():
    from tools.batch_runner import _progress_line
    argv = [sys.executable, 'run.py', '--model', 'TSBooster', '--tsb_backbone', 'lstm', '--data_path', 'USD_TWD.csv',
            '--pred_len', '3', '--pred_offset', '6', '--rand_seed', '42']
    line = _progress_line(4, 16, {'status': 'complete', 'elapsed_s': 52.4}, argv, 12, 0, 120.0)
    assert line == ('[ 4/16] complete TSBooster:lstm USD_TWD pl3 po6 seed 42 (52s) | running 12 | failed 0 '
                    '| elapsed 2m 00s | ETA 6m 00s')
    failed = _progress_line(16, 16, {'status': 'failed', 'elapsed_s': 3, 'log': '/tmp/run.log'}, argv, 0, 1, 4000)
    assert failed.endswith('| failed 1 | elapsed 1h 06m | log /tmp/run.log')
