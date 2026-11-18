#!/usr/bin/env python3
"""Score candidate-feature parquet and (if route-aligned labels exist) evaluate.

Predictions never use reference-route columns. They are used only as targets
for computing metrics when available. This is not full-route reconstruction.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None,
        'log_loss': float(log_loss(y, p, labels=[0, 1])),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True,
                        help='Evaluator candidate Parquet created using existing build_candidates')
    parser.add_argument('--model', default='artifacts/logistic_regression_tuned/model_all_80.joblib')
    parser.add_argument('--output-dir', default='artifacts/evaluator_test_results')
    parser.add_argument('--batch-size', type=int, default=50000)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    package = joblib.load(args.model)
    model = package['model']
    features = package['feature_cols']
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    needed = set(features) | {'trip_id', 'point_idx', 'arc_id'}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f'Evaluator candidate table missing columns: {missing}')
    eval_trips = set(df['trip_id'].astype(str).unique())
    train_trips = set(map(str, package.get('training_trip_ids', [])))
    overlap = eval_trips & train_trips
    if overlap:
        raise ValueError('Evaluator contains training trip IDs: ' + repr(sorted(overlap)))
    print(f'Scoring {len(df):,} candidates from {len(eval_trips)} evaluator trips...', flush=True)
    p = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))
        X = (df.iloc[start:end][features].apply(pd.to_numeric, errors='coerce')
             .replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32))
        p[start:end] = model.predict_proba(X)[:, 1]
    pred = df[['trip_id', 'point_idx', 'arc_id']].copy()
    pred['probability'] = p
    try:
        pred.to_parquet(out / 'all_candidate_scores.parquet', index=False)
        scores_path = out / 'all_candidate_scores.parquet'
    except ImportError:
        scores_path = out / 'all_candidate_scores.csv'
        pred.to_csv(scores_path, index=False)
    winners = pred.loc[pred.groupby(['trip_id', 'point_idx'], sort=False)['probability'].idxmax()]
    winners = winners.sort_values(['trip_id', 'point_idx'])
    try:
        winners.to_parquet(out / 'top1_per_gps_point.parquet', index=False)
        winners_path = out / 'top1_per_gps_point.parquet'
    except ImportError:
        winners_path = out / 'top1_per_gps_point.csv'
        winners.to_csv(winners_path, index=False)
    print('Top-1 predicted GPS points:', len(winners))
    print('Trip IDs:', sorted(eval_trips))
    print('Candidate scores:', scores_path)
    print('Top-1 predictions:', winners_path)

    # Reference alignment must already have happened inside build_candidates.
    if not {'label', 'label_usable'}.issubset(df.columns):
        print('\nNo reference-aligned labels found; scoring completed (no accuracy reported).')
        return
    usable = df['label_usable'].eq(True).to_numpy() & df['label'].notna().to_numpy()
    if not usable.any():
        print('\nNo usable reference-aligned labels found; scoring completed (no accuracy reported).')
        return
    labels = df.loc[usable, 'label'].astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError('Invalid evaluator labels; expected 0/1')
    prob = p[usable]
    sub = df.loc[usable, ['trip_id', 'point_idx']].copy()
    sub['label'] = labels
    sub['probability'] = prob
    positives = sub.groupby(['trip_id', 'point_idx'], sort=False)['label'].transform('sum')
    eligible = sub.loc[positives.eq(1)].copy()
    winners_eligible = eligible.loc[
        eligible.groupby(['trip_id', 'point_idx'], sort=False)['probability'].idxmax()
    ]
    n_positive_points = len(winners_eligible)
    n_total_points = len(winners)
    result = metrics(labels, prob)
    result.update({
        'evaluation_type': 'evaluator_reference_aligned_candidate_level',
        'evaluator_trips': len(eval_trips),
        'total_candidate_rows': len(df),
        'usable_labelled_candidate_rows': int(usable.sum()),
        'all_gps_points': n_total_points,
        'top1_eligible_points': n_positive_points,
        'top1_accuracy_when_positive_present':
            float(winners_eligible['label'].mean()) if n_positive_points else None,
        'eligible_point_fraction': n_positive_points / n_total_points if n_total_points else 0.0,
        'warning': 'Reference routes were aligned to GPS points to derive proxy labels; '
                   'conditional Top-1 is not full-route accuracy.',
    })
    with (out / 'evaluator_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    per_trip = []
    for trip, g in winners_eligible.groupby('trip_id'):
        per_trip.append({'trip_id': trip, 'top1': float(g['label'].mean()),
                         'eligible_points': len(g)})
    pd.DataFrame(per_trip).to_csv(out / 'per_trip_top1.csv', index=False)
    print('\n========== EVALUATOR METRICS ==========', flush=True)
    for k in ('accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
              'roc_auc', 'log_loss', 'top1_accuracy_when_positive_present',
              'top1_eligible_points', 'eligible_point_fraction'):
        print(f'{k:42}: {result[k]}')
    print('Metrics saved:', out / 'evaluator_metrics.json')
    print('NOTE: This is candidate-level/reference-aligned accuracy, not connected-route accuracy.')


if __name__ == '__main__':
    main()
