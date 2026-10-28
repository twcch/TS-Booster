"""One reproducible entrypoint for syntax and contract/integration checks."""

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spawn', action='store_true', help='include real spawn DataLoader workers')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                        help='run the real train/test smoke on the selected available device')
    args = parser.parse_args(argv)
    excluded = {'.git', '.venv', '__pycache__', 'dataset', 'data', 'results', 'test_results',
                'checkpoints', 'batch_runs', '.codex', '.agents'}
    count = 0
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in excluded]
        for name in files:
            if name.endswith('.py'):
                path = Path(directory) / name
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
                count += 1
    import torch
    available = args.device == 'cpu' or (torch.cuda.is_available() if args.device == 'cuda'
                                         else torch.backends.mps.is_available())
    if not available:
        parser.error(f'{args.device} is unavailable; no device checks were performed.')
    print(f'Syntax: {count} Python files; Python {sys.version.split()[0]}, torch {torch.__version__}', flush=True)
    environment = os.environ.copy()
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['TSBOOSTER_TEST_DEVICE'] = args.device
    if args.spawn:
        environment['TSBOOSTER_TEST_SPAWN'] = '1'
    return subprocess.call([sys.executable, '-m', 'pytest', '-q', 'tests'], cwd=ROOT, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
