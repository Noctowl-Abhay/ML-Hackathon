from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
OUTPUT = ROOT / 'outputs' / 'results'
OUTPUT.mkdir(parents=True, exist_ok=True)

records = []
for split in ('train', 'student_test'):
    split_dir = DATA / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f'Missing split directory: {split_dir}')
    for trip_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        trip_id = trip_dir.name
        paths = {suffix: trip_dir / f'{trip_id}.{suffix}'
                 for suffix in ('track', 'nodes', 'arcs', 'route')}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f'Missing file: {path}')

        track = np.loadtxt(paths['track'], ndmin=2)
        nodes = np.loadtxt(paths['nodes'], ndmin=2)
        arcs = np.loadtxt(paths['arcs'], dtype=int, ndmin=2)
        route = np.loadtxt(paths['route'], dtype=int, ndmin=1)
        if track.shape[1] != 3 or nodes.shape[1] != 2 or arcs.shape[1] != 2:
            raise ValueError(f'Unexpected column count in {trip_id}')
        timestamps = track[:, 2]
        gaps = np.diff(timestamps)
        valid_arc_endpoints = ((arcs >= 0) & (arcs < len(nodes))).all()
        valid_route_ids = ((route >= 0) & (route < len(arcs))).all()
        active_node_count = np.unique(arcs).size

        records.append({
            'partition': split, 'trip_id': trip_id,
            'gps_points': len(track), 'nodes': len(nodes),
            'arcs': len(arcs), 'route_arcs': len(route),
            'missing_gps_values': int(np.isnan(track).sum()),
            'invalid_gps_coords': int(((track[:, 0] < -180) | (track[:, 0] > 180) |
                                     (track[:, 1] < -90) | (track[:, 1] > 90)).sum()),
            'non_increasing_times': int(np.sum(gaps <= 0)),
            'gaps_over_1s': int(np.sum(gaps > 1)),
            'largest_gap_s': float(max(gaps, default=0)),
            'isolated_nodes': len(nodes) - active_node_count,
            'valid_arc_endpoints': bool(valid_arc_endpoints),
            'valid_route_ids': bool(valid_route_ids),
        })

result = pd.DataFrame.from_records(records)
result.to_csv(OUTPUT / 'dataset_inventory.csv', index=False)
summary = result.groupby('partition').agg(
    trips=('trip_id', 'count'),
    gps_points=('gps_points', 'sum'),
    nodes=('nodes', 'sum'),
    arcs=('arcs', 'sum'),
    route_arcs=('route_arcs', 'sum'),
    missing_values=('missing_gps_values', 'sum'),
    gaps_over_1s=('gaps_over_1s', 'sum'),
)
print('\nDATASET SUMMARY\n', summary.to_string())
print('\nLARGEST GAPS\n', result.nlargest(5, 'largest_gap_s')[
    ['partition', 'trip_id', 'largest_gap_s']].to_string(index=False))
print(f'\nSaved: {OUTPUT / "dataset_inventory.csv"}')
