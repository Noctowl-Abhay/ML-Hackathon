#!/usr/bin/env python3
"""Run PathSnap's EXISTING feature builder on evaluator trips, in isolation.

Some project versions restrict --partition to train/student_test. We stage
only the evaluator trips as a temporary `train` partition so the existing
route-alignment logic can produce evaluation labels, while avoiding edits to
real 80-trip train data and existing features_train.parquet.

The output partition column is relabeled evaluator_test. No model training.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def read_cols(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if isinstance(data, list) else data.get('feature_cols', data.get('features'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--processed-evaluator', default='Data/processed/evaluator_test')
    ap.add_argument('--feature-config', default='outputs/features/feature_config.json')
    ap.add_argument('--output', default='outputs/features/features_evaluator_test.parquet')
    args = ap.parse_args()
    root = Path.cwd().resolve()
    source = root / 'features'
    if not (source / 'build_candidates.py').exists():
        raise FileNotFoundError('Run from project root; features/build_candidates.py not found')
    eval_dir = (root / args.processed_evaluator).resolve()
    cfg_path = (root / args.feature_config).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f'Training feature config not found: {cfg_path}')
    expected_cols = read_cols(cfg_path)
    if not isinstance(expected_cols, list) or not expected_cols:
        raise ValueError('Training feature config lacks feature_cols')
    trips = sorted(p for p in eval_dir.iterdir() if p.is_dir() and re.fullmatch(r'\d{8}', p.name))
    if len(trips) != 10:
        raise ValueError(f'Expected 10 evaluator trips in {eval_dir}, found {len(trips)}')
    for trip in trips:
        if not (trip / f'{trip.name}_track_processed.csv').exists():
            raise FileNotFoundError(f'Missing processed GPS CSV for evaluator trip {trip.name}')
        if not (trip / f'{trip.name}.route').exists():
            raise FileNotFoundError(f'Missing reference route for evaluator trip {trip.name}')

    print('Isolated feature extraction: only evaluator trips; no real training files modified.', flush=True)
    print('NOTE: Route files will be read ONLY for deriving evaluation labels, never as features.', flush=True)
    with tempfile.TemporaryDirectory(prefix='pathsnap_eval_features_') as tmp:
        base = Path(tmp)
        # Source-based absolute path calculations now also stay inside tmp.
        shutil.copytree(source, base / 'features', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        staged = base / 'Data' / 'processed' / 'train'
        staged.mkdir(parents=True)
        for trip in trips:
            (staged / trip.name).symlink_to(trip, target_is_directory=True)
        env = os.environ.copy()
        # Use temporary features code before any project-global module version.
        env['PYTHONPATH'] = str(base) + os.pathsep + str(root) + os.pathsep + env.get('PYTHONPATH', '')
        cmd = [sys.executable, '-m', 'features.build_candidates',
               '--data-root', 'Data/processed', '--partition', 'train']
        print('Executing:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=base, env=env, check=True)
        produced = base / 'outputs' / 'features' / 'features_train.parquet'
        if not produced.exists():
            alt = base / 'outputs' / 'features' / 'features_train.csv.gz'
            if not alt.exists():
                found = [str(p.relative_to(base)) for p in base.rglob('features_train*')]
                raise FileNotFoundError(
                    'Existing generator did not produce expected features_train.parquet or CSV.gz '
                    f'in isolated workspace. Found: {found}'
                )
            table = pd.read_csv(alt)
        else:
            table = pd.read_parquet(produced)
        print('Generated candidate rows:', len(table), flush=True)
        if 'trip_id' not in table.columns:
            raise ValueError('Generator output does not contain trip_id')
        trip_id_values = set(table['trip_id'].astype(str))
        expected_ids = {trip.name for trip in trips}
        if trip_id_values != expected_ids:
            raise ValueError('Generator produced the wrong trip set: ' + repr(trip_id_values))
        missing_cols = sorted(set(expected_cols) - set(table.columns))
        if missing_cols:
            raise ValueError(f'Evaluator candidates missing trained features: {missing_cols}')
        generated_cfg = base / 'outputs' / 'features' / 'feature_config.json'
        if generated_cfg.exists():
            got_cols = read_cols(generated_cfg)
            if got_cols != expected_cols:
                raise ValueError('Evaluator feature configuration differs from training configuration')
        # Avoid any chance of recording evaluator examples as the train partition.
        table['partition'] = 'evaluator_test'
        output = (root / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output == (root / 'outputs/features/features_train.parquet').resolve():
            raise ValueError('Refusing to overwrite original training features')
        table.to_parquet(output, index=False)
        print('Saved evaluator features:', output, flush=True)
        print('Reference-aligned labels available:',
              bool('label_usable' in table and table['label_usable'].eq(True).any()), flush=True)
        print('You can now evaluate the FROZEN selected model, without retraining.', flush=True)


if __name__ == '__main__':
    main()
