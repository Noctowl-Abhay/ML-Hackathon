#!/usr/bin/env python3
"""Prepare new evaluator_test.zip in PathSnap's non-destructive preprocessing format."""
import argparse
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', dest='zip_path', required=True)
    ap.add_argument('--output-root', default='Data/processed/evaluator_test')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    src = Path(args.zip_path)
    dest = Path(args.output_root)
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    with zipfile.ZipFile(src) as archive:
        names = archive.namelist()
        tracks = sorted(
            name for name in names
            if re.fullmatch(r'evaluator_test/\d{8}/\d{8}\.track', name)
        )
        if not tracks:
            raise ValueError('ZIP does not contain evaluator_test/<tripid>/<tripid>.track')
        for trackname in tracks:
            trip = trackname.split('/')[1]
            expected = f'evaluator_test/{trip}/{trip}.track'
            if trackname != expected:
                raise ValueError(f'Unexpected track path: {trackname}')
            for ext in ('nodes', 'arcs', 'route'):
                if f'evaluator_test/{trip}/{trip}.{ext}' not in names:
                    raise ValueError(f'Missing {ext} for trip {trip}')
            folder = dest / trip
            output_csv = folder / f'{trip}_track_processed.csv'
            if output_csv.exists() and not args.overwrite:
                raise FileExistsError(f'{output_csv} exists. Pass --overwrite to replace.')
            folder.mkdir(exist_ok=True)
            with archive.open(trackname) as f:
                raw = f.read()
            df = pd.read_csv(io.BytesIO(raw), sep=r'\s+', header=None,
                             names=['longitude', 'latitude', 'timestamp'])
            for col in ['longitude', 'latitude', 'timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['time_difference_s'] = df['timestamp'].diff()
            df['gap_over_1s'] = df['time_difference_s'] > 1
            df['duplicate_observation'] = df.duplicated(
                subset=['longitude', 'latitude', 'timestamp'], keep='first'
            )
            df['valid_coordinates'] = (
                np.isfinite(df[['longitude', 'latitude', 'timestamp']]).all(axis=1)
                & df['longitude'].between(-180, 180)
                & df['latitude'].between(-90, 90)
            )
            df.to_csv(output_csv, index=False)
            for ext in ('track', 'nodes', 'arcs', 'route'):
                name = f'evaluator_test/{trip}/{trip}.{ext}'
                with archive.open(name) as fin, (folder / f'{trip}.{ext}').open('wb') as fout:
                    shutil.copyfileobj(fin, fout)
            row = {'trip_id': trip, 'gps_points': len(df),
                   'invalid_coordinates': int((~df['valid_coordinates']).sum()),
                   'time_gaps_over_1s': int(df['gap_over_1s'].sum()),
                   'duplicates': int(df['duplicate_observation'].sum()),
                   'route_available': True}
            report.append(row)
            print(f'{trip}: {len(df):,} GPS points, {row["time_gaps_over_1s"]} gaps', flush=True)
    summary = pd.DataFrame(report)
    summary.to_csv(dest / 'evaluator_preprocessing_summary.csv', index=False)
    print(f'\nPrepared {len(report)} evaluator trips, {summary.gps_points.sum():,} GPS points')
    print('Output:', dest)
    print('Summary:', dest / 'evaluator_preprocessing_summary.csv')
    print('Reference .route files were copied for evaluation ONLY; do not use them as model inputs.')


if __name__ == '__main__':
    main()
