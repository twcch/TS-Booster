"""Summarize six explicitly selected, compatible M4 attempts without mixing runs."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.m4 import M4Dataset, M4Meta
from exp.common.runtime import file_fingerprint
from utils.m4_summary import M4Summary


SEASON_CONFIG = {'seasonal_patterns', 'pred_len', 'seq_len', 'label_len', 'frequency_map'}


def _read_json(path):
    try:
        record = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Unreadable M4 record: {path}') from error
    if not isinstance(record, dict):
        raise ValueError(f'M4 record must contain an object: {path}')
    return record


def _finite_array(path, shape):
    values = np.load(path, allow_pickle=False)
    if values.shape != shape or values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError(f'Invalid finite array or shape at {path}; expected {shape}.')
    return values


def _complete_attempt(directory):
    directory = Path(directory).resolve()
    metadata = _read_json(directory / 'metadata.json')
    result = _read_json(directory / 'result.json')
    if any(record.get('status') != 'complete' for record in (metadata, result)):
        raise ValueError(f'M4 attempt is not complete: {directory}')
    for name in ('experiment_id', 'group_id', 'attempt_id', 'setting'):
        if not metadata.get(name) or metadata[name] != result.get(name):
            raise ValueError(f'M4 result/metadata identity mismatch ({name}): {directory}')
    if metadata['setting'] != directory.name:
        raise ValueError(f'M4 attempt directory does not match its setting: {directory}')
    config, details = result.get('config', {}), result.get('details', {})
    if config != metadata.get('config'):
        raise ValueError(f'M4 result/metadata config mismatch: {directory}')
    if config.get('task_name') != 'short_term_forecast' or config.get('data') != 'm4':
        raise ValueError(f'Only M4 short_term_forecast attempts can be summarized: {directory}')
    if not config.get('model') or 'rand_seed' not in config or not metadata.get('source_hash'):
        raise ValueError(f'M4 attempt lacks model, seed, or source provenance: {directory}')
    season = config.get('seasonal_patterns')
    if season not in M4Meta.seasonal_patterns or details.get('seasonal_patterns') != season:
        raise ValueError(f'M4 seasonal metadata mismatch: {directory}')
    horizon = M4Meta.horizons_map[season]
    expected_config = {'pred_len': horizon, 'seq_len': 2 * horizon, 'label_len': horizon,
                       'frequency_map': M4Meta.frequency_map[season]}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(f'M4 resolved horizon/context configuration is inconsistent: {directory}')
    if details.get('metric_space') != 'original' or not details.get('evaluation_protocol'):
        raise ValueError(f'M4 attempt lacks its original-scale evaluation protocol: {directory}')
    if metadata.get('evaluation_version') != result.get('evaluation_version') or result.get('evaluation_version') != 2:
        raise ValueError(f'M4 aggregation requires evaluation_version=2: {directory}')
    csv_name = f'{season}_forecast.csv'
    if details.get('forecast_csv') != csv_name:
        raise ValueError(f'M4 forecast CSV name mismatch: {directory}')
    required = {'pred.npy', 'true.npy', 'metrics.npy', csv_name}
    for record in (metadata, result):
        declared = record.get('artifacts', [])
        if not isinstance(declared, list) or not required.issubset(declared):
            raise ValueError(f'M4 attempt does not declare complete artifacts: {directory}')
        for name in declared:
            if not isinstance(name, str) or Path(name).name != name or not (directory / name).is_file():
                raise ValueError(f'Missing or invalid M4 artifact {name!r}: {directory}')
    common_config = {key: value for key, value in config.items()
                     if key not in SEASON_CONFIG | {'model_id'}}
    effective = {key: value for key, value in metadata.get('effective_config', {}).items()
                 if key not in SEASON_CONFIG}
    signature = {'config': common_config, 'effective_config': effective,
                 'source_hash': metadata['source_hash'], 'data': metadata.get('data'),
                 'versions': metadata.get('versions'), 'evaluation_version': result['evaluation_version'],
                 'evaluation_protocol': details['evaluation_protocol'], 'metric_space': details['metric_space']}
    if not isinstance(signature['data'], dict) or not signature['data'] or not signature['versions']:
        raise ValueError(f'M4 attempt lacks data/environment provenance: {directory}')
    return directory, metadata, result, signature


def _official_data(root_path):
    root = Path(root_path).resolve()
    filenames = ('M4-info.csv', 'training.npz', 'test.npz', 'submission-Naive2.csv')
    if any(not (root / name).is_file() for name in filenames):
        raise ValueError(f'M4 aggregation requires local {", ".join(filenames)} in {root}.')
    training = M4Dataset.load(training=True, dataset_file=str(root))
    test = M4Dataset.load(training=False, dataset_file=str(root))
    if not np.array_equal(training.ids, test.ids) or not np.array_equal(training.groups, test.groups):
        raise ValueError('Official M4 TRAIN/TEST IDs and groups must have identical order.')
    ids = list(map(str, test.ids))
    if len(ids) != len(set(ids)) or set(test.groups) != set(M4Meta.seasonal_patterns):
        raise ValueError('Official M4 data must contain unique IDs and all six seasonal groups.')
    naive = pd.read_csv(root / 'submission-Naive2.csv', dtype=str)
    if naive.shape[1] < 2 or list(naive.iloc[:, 0].astype(str)) != ids:
        raise ValueError('Naive2 IDs must match official M4 TEST IDs in exactly the same order.')
    naive_values = naive.iloc[:, 1:].to_numpy(dtype=np.float64)
    for index, group in enumerate(test.groups):
        horizon, frequency = M4Meta.horizons_map[group], M4Meta.frequency_map[group]
        context = np.asarray(training.values[index], dtype=np.float64)
        target = np.asarray(test.values[index], dtype=np.float64)
        context, target = context[~np.isnan(context)], target[~np.isnan(target)]
        if training.frequencies[index] != frequency or test.horizons[index] != horizon:
            raise ValueError(f'Official M4 frequency/horizon mismatch for {ids[index]}.')
        if len(context) <= frequency or not np.isfinite(context).all() or target.shape != (horizon,) or not np.isfinite(target).all():
            raise ValueError(f'Invalid official M4 TRAIN/TEST series for {ids[index]}.')
        row = naive_values[index]
        if len(row) < horizon or not np.isfinite(row[:horizon]).all() or not np.isnan(row[horizon:]).all():
            raise ValueError(f'Naive2 forecast must have a finite horizon followed by NaN padding: {ids[index]}.')
    return root, training, test, {name: file_fingerprint(root / name) for name in filenames}


def _write_new_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes a complete file and fails atomically if output
        # already exists, including a file created while evaluation was running.
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summarize_m4(runs, root_path, output, *, summary_factory=M4Summary):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'M4 summary output already exists: {output}')
    if len(runs) != 6 or len({Path(path).resolve() for path in runs}) != 6:
        raise ValueError('Select exactly six distinct complete M4 attempt directories.')
    attempts = [_complete_attempt(path) for path in runs]
    seasons = [attempt[2]['config']['seasonal_patterns'] for attempt in attempts]
    if set(seasons) != set(M4Meta.seasonal_patterns):
        raise ValueError('Select exactly one attempt for each of the six M4 seasonal groups.')
    signature = attempts[0][3]
    for directory, _, _, candidate in attempts[1:]:
        different = [key for key in signature if candidate[key] != signature[key]]
        if different:
            raise ValueError(f'Incompatible M4 attempts ({", ".join(different)}): {directory}')
    root, _, test, fingerprints = _official_data(root_path)
    for name in ('M4-info.csv', 'training.npz', 'test.npz'):
        if signature['data'].get(name) != fingerprints[name]:
            raise ValueError(f'Official M4 data fingerprint does not match selected attempts: {name}')
    selected = {}
    with tempfile.TemporaryDirectory(prefix='tsbooster-m4-summary-') as temporary:
        staging = Path(temporary)
        for directory, metadata, result, _ in attempts:
            season, details = result['config']['seasonal_patterns'], result['details']
            expected_ids = list(map(str, test.ids[test.groups == season]))
            csv_path = directory / f'{season}_forecast.csv'
            table = pd.read_csv(csv_path, dtype={'id': str})
            horizon = M4Meta.horizons_map[season]
            if list(table.columns) != ['id', *[f'V{i + 1}' for i in range(horizon)]]:
                raise ValueError(f'Invalid M4 forecast CSV columns: {csv_path}')
            if list(table.id) != expected_ids or details.get('series_ids') != expected_ids:
                raise ValueError(f'M4 forecast IDs must follow the official {season} order: {directory}')
            shape = (len(expected_ids), horizon, 1)
            prediction = _finite_array(directory / 'pred.npy', shape)
            target = _finite_array(directory / 'true.npy', shape)
            official_targets = np.stack([np.asarray(values)[~np.isnan(values)]
                                         for values in test.values[test.groups == season]])
            if not np.allclose(target[:, :, 0], official_targets, rtol=1e-6, atol=1e-6):
                raise ValueError(f'M4 saved targets disagree with official TEST: {directory}')
            values = table.iloc[:, 1:].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all() or not np.allclose(values, prediction[:, :, 0], rtol=1e-7, atol=1e-7):
                raise ValueError(f'M4 CSV disagrees with prediction artifact: {directory}')
            # Existing M4Summary reads numeric-only tables in official group order.
            pd.DataFrame(values, columns=table.columns[1:]).to_csv(staging / f'{season}_forecast.csv', index=False)
            selected[season] = {'directory': str(directory), 'experiment_id': metadata['experiment_id'],
                                'attempt_id': metadata['attempt_id'], 'forecast_sha256': file_fingerprint(csv_path)}
        smape, owa, mape, mase = summary_factory(str(staging) + os.sep, str(root)).evaluate()
    scores, states = {}, {}
    for metric_name, grouped in (('smape', smape), ('owa', owa), ('mape', mape), ('mase', mase)):
        scores[metric_name], states[metric_name] = {}, {}
        for group, value in grouped.items():
            number = float(value)
            finite = math.isfinite(number)
            scores[metric_name][group] = number if finite else None
            states[metric_name][group] = 'ok' if finite else 'nonfinite'
    record = {'schema_version': 1, 'status': 'complete', 'evaluation_version': 2,
              'aggregation': 'Existing M4Summary: per-series MASE, per-group sMAPE/MAPE, count-weighted groups, Naive2 OWA; rounded to 3 decimals',
              'common': signature, 'official_data': {'root_path': str(root), 'fingerprints': fingerprints},
              'summary_source_sha256': file_fingerprint(PROJECT_ROOT / 'utils' / 'm4_summary.py'),
              'runs': {season: selected[season] for season in M4Meta.seasonal_patterns},
              'metrics': scores, 'metric_states': states}
    _write_new_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=6, required=True, help='one complete attempt directory for each M4 seasonal group')
    parser.add_argument('--root_path', required=True, help='local official M4 data including submission-Naive2.csv')
    parser.add_argument('--output', required=True, help='new JSON summary path; existing files are never overwritten')
    args = parser.parse_args(argv)
    try:
        record = summarize_m4(args.runs, args.root_path, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(record['metrics'], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
