#!/usr/bin/env python3
"""PathSnap trip-grouped Logistic Regression hyperparameter tuning.

Fit selected model on ALL 80 training trips after 5-fold grouped CV.
Never reads the evaluator data. Every metric from CV is out-of-fold.
"""
import argparse
import json
import time
import warnings
from pathlib import Path
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FORBIDDEN = {
    'label', 'label_usable', 'label_arc_id', 'label_route_pos',
    'label_dist_m', 'point_has_positive', 'trip_id', 'point_idx',
    'arc_id', 'partition'
}


def feature_list(path):
    with open(path, encoding='utf-8') as f:
        cfg = json.load(f)
    cols = cfg if isinstance(cfg, list) else cfg.get('feature_cols', cfg.get('features'))
    if not isinstance(cols, list) or not cols or len(cols) != len(set(cols)):
        raise ValueError('feature_config.json lacks a unique feature_cols list')
    if not all(isinstance(c, str) for c in cols):
        raise ValueError('Feature names must be strings')
    if FORBIDDEN.intersection(cols):
        raise ValueError('Reference/identifier columns would leak labels: ' +
                         repr(sorted(FORBIDDEN.intersection(cols))))
    return cols


def pipeline(C, class_weight, max_iter):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=float(C), class_weight=class_weight, solver='lbfgs',
            max_iter=max_iter, random_state=42,
        )),
    ])


def point_top1(y, proba, codes):
    """Conditional Top-1 on points with exactly one positive label.

    Works with candidate arrays containing multiple rows per GPS point.
    """
    if len(y) == 0:
        return float('nan'), 0, 0
    # Lexicographic sort: GPS point first, probability descending.
    order = np.lexsort((-proba, codes))
    sorted_codes = codes[order]
    starts = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    winners = order[starts]
    positive_counts = np.bincount(codes, weights=y)
    eligible = (positive_counts[codes[winners]] == 1)
    n_eligible = int(eligible.sum())
    score = float(y[winners[eligible]].mean()) if n_eligible else float('nan')
    return score, n_eligible, int(len(winners))


def measure(y, proba, codes):
    pred = (proba >= 0.5).astype(np.int8)
    top1, eligible, total_points = point_top1(y, proba, codes)
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, proba)) if np.unique(y).size == 2 else None,
        'log_loss': float(log_loss(y, proba, labels=[0, 1])),
        'top1_accuracy_when_positive_present': top1,
        'top1_eligible_points': eligible,
        'scored_points': total_points,
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='outputs/features/features_train.parquet')
    p.add_argument('--feature-config', default='outputs/features/feature_config.json')
    p.add_argument('--output-dir', default='artifacts/logistic_regression_tuned')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--C-values', type=float, nargs='+',
                   default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
    p.add_argument('--max-iter', type=int, default=1000)
    p.add_argument('--allow-fewer-trips', action='store_true',
                   help='Only for synthetic smoke tests; never use for final submission')
    args = p.parse_args()
    start_all = time.perf_counter()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = feature_list(args.feature_config)
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    needed = set(cols) | {'trip_id', 'point_idx', 'label', 'label_usable'}
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f'Missing data columns: {missing}')
    n_input = len(df)
    df = df.loc[df['label_usable'].eq(True)].copy()
    if df.empty or df['label'].isna().any():
        raise ValueError('No usable labels, or labels contain nulls')
    trip_ids = sorted(df['trip_id'].astype(str).unique().tolist())
    n_trips = len(trip_ids)
    if n_trips != 80 and not args.allow_fewer_trips:
        raise ValueError(f'Expected 80 labelled training trips, got {n_trips}')
    if args.folds < 2 or args.folds > n_trips:
        raise ValueError('Fold count must be between 2 and number of trips')
    X = (df[cols].apply(pd.to_numeric, errors='coerce')
         .replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32))
    y = df['label'].astype(np.int8).to_numpy()
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Binary labels 0 and 1 must both be present')
    trips = df['trip_id'].astype(str).to_numpy()
    point_idx = df['point_idx'].to_numpy()
    point_keys = pd.MultiIndex.from_arrays([trips, point_idx])
    codes, unique_points = pd.factorize(point_keys, sort=False)
    if np.any(codes < 0):
        raise ValueError('Missing trip IDs or GPS point identifiers')
    codes = codes.astype(np.int32)

    cv = GroupKFold(n_splits=args.folds)
    splits = list(cv.split(X, y, groups=trips))
    c_values = sorted(set(args.C_values))
    if not c_values or any(c <= 0 for c in c_values):
        raise ValueError('C values must be positive numbers')
    grid = list(product(c_values, [None, 'balanced']))
    print(f'\nPathSnap LOGISTIC REGRESSION tuning: {n_trips} trips, '
          f'{len(df):,} usable candidate rows ({n_input:,} total), '
          f'{len(cols)} features', flush=True)
    print(f'Grouped CV: {args.folds} folds; combinations: {len(grid)}; '
          f'total model fits in search: {len(grid) * args.folds}', flush=True)
    print('Evaluator test data is NOT used in this script.\n', flush=True)
    print('Fold validation trips:')
    for j, (_, va) in enumerate(splits, 1):
        print(f'  Fold {j}: {sorted(set(trips[va]))}', flush=True)
    rows = []
    warnings.simplefilter('once', ConvergenceWarning)
    for idx, (C, cw) in enumerate(grid, 1):
        t0 = time.perf_counter()
        fold_scores, fold_losses, fold_eligible = [], [], []
        for fold_index, (tr, va) in enumerate(splits, 1):
            model = pipeline(C, cw, args.max_iter)
            model.fit(X[tr], y[tr])
            probs = model.predict_proba(X[va])[:, 1]
            top1, n_eligible, _ = point_top1(y[va], probs, codes[va])
            if not np.isfinite(top1):
                raise ValueError(f'Fold {fold_index} has no eligible positive GPS points')
            fold_scores.append(top1)
            fold_eligible.append(n_eligible)
            fold_losses.append(float(log_loss(y[va], probs, labels=[0, 1])))
        row = {
            'C': C,
            'class_weight': 'balanced' if cw else 'none',
            'mean_cv_top1': float(np.mean(fold_scores)),
            'std_cv_top1': float(np.std(fold_scores)),
            'mean_cv_log_loss': float(np.mean(fold_losses)),
            'eligible_points_cv': int(sum(fold_eligible)),
            'seconds': round(time.perf_counter() - t0, 2),
        }
        row.update({f'fold_{i}_top1': s for i, s in enumerate(fold_scores, 1)})
        rows.append(row)
        print(f'[{idx:02d}/{len(grid)}] C={C:<8g} weight={row["class_weight"]:<8} '
              f'CV Top1={row["mean_cv_top1"]:.4%}  '
              f'logloss={row["mean_cv_log_loss"]:.4f}  '
              f'time={row["seconds"]:.1f}s', flush=True)
        pd.DataFrame(rows).to_csv(out / 'cv_results.csv', index=False)

    cv_results = pd.DataFrame(rows).sort_values(
        ['mean_cv_top1', 'mean_cv_log_loss'], ascending=[False, True]
    ).reset_index(drop=True)
    cv_results.to_csv(out / 'cv_results_ranked.csv', index=False)
    best = cv_results.iloc[0]
    winner_C = float(best['C'])
    winner_weight = None if best['class_weight'] == 'none' else 'balanced'

    # Re-evaluate selected configuration with OOF predictions for ALL 80 trips.
    # These predictions are from models that did not train on the corresponding trips.
    print('\nGenerating out-of-fold predictions for selected configuration...', flush=True)
    oof = np.full(len(y), np.nan, dtype=np.float32)
    fold_id = np.zeros(len(y), dtype=np.int8)
    for fold_index, (tr, va) in enumerate(splits, 1):
        model = pipeline(winner_C, winner_weight, args.max_iter)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
        fold_id[va] = fold_index
    if not np.isfinite(oof).all():
        raise RuntimeError('Some out-of-fold predictions were not generated')
    oof_metrics = measure(y, oof, codes)
    oof_metrics.update({
        'evaluation_type': '5_fold_grouped_oof' if args.folds == 5 else f'{args.folds}_fold_grouped_oof',
        'trips': n_trips,
        'candidate_rows': len(y),
        'available_rows_before_label_filter': n_input,
        'features': len(cols),
        'best_C': winner_C,
        'best_class_weight': 'balanced' if winner_weight else 'none',
        'folds': args.folds,
    })
    oof_df = pd.DataFrame({
        'trip_id': trips, 'point_idx': point_idx, 'label': y,
        'probability': oof, 'fold': fold_id,
    })
    try:
        oof_df.to_parquet(out / 'oof_predictions.parquet', index=False)
    except ImportError:
        oof_df.to_csv(out / 'oof_predictions.csv', index=False)
        print('Parquet engine not installed: saved OOF predictions as CSV instead.', flush=True)

    print('Fitting final deployment model on all 80 trips...', flush=True)
    final_model = pipeline(winner_C, winner_weight, args.max_iter)
    final_start = time.perf_counter()
    final_model.fit(X, y)
    final_fit_seconds = time.perf_counter() - final_start
    package = {
        'model': final_model,
        'feature_cols': cols,
        'model_name': 'logistic_regression_tuned',
        'training_trips': n_trips,
        'training_rows': len(y),
        'training_trip_ids': trip_ids,
        'cv_oof_metrics': oof_metrics,
        'params': {'C': winner_C, 'class_weight': winner_weight,
                   'solver': 'lbfgs', 'penalty': 'l2', 'max_iter': args.max_iter},
    }
    joblib.dump(package, out / 'model_all_80.joblib')
    with open(out / 'best_params.json', 'w', encoding='utf-8') as f:
        json.dump({
            'C': winner_C,
            'class_weight': winner_weight,
            'mean_cv_top1': float(best['mean_cv_top1']),
            'mean_cv_log_loss': float(best['mean_cv_log_loss']),
            'cv_folds': args.folds,
            'features': len(cols),
        }, f, indent=2)
    with open(out / 'cv_oof_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(oof_metrics, f, indent=2)
    with open(out / 'training_info.json', 'w', encoding='utf-8') as f:
        json.dump({
            'training_trip_ids': trip_ids,
            'n_original_candidate_rows': n_input,
            'n_usable_candidate_rows': len(y),
            'n_features': len(cols),
            'fit_seconds_all_80': round(final_fit_seconds, 3),
            'total_seconds': round(time.perf_counter() - start_all, 3),
            'source_data': args.data,
            'feature_config': args.feature_config,
        }, f, indent=2)
    print('\n========== FINISHED ==========', flush=True)
    print('Best C:', winner_C)
    print('Best class_weight:', winner_weight)
    print('5-fold CV mean Top1:' if args.folds == 5 else 'CV mean Top1:',
          f'{best["mean_cv_top1"]:.4%}')
    print('OOF Top1 (all 80 trips):', f'{oof_metrics["top1_accuracy_when_positive_present"]:.4%}')
    print('OOF Accuracy:', f'{oof_metrics["accuracy"]:.4%}')
    print('OOF F1:', f'{oof_metrics["f1"]:.4%}')
    print('OOF ROC-AUC:', f'{oof_metrics["roc_auc"]:.4f}')
    print('Final model:', out / 'model_all_80.joblib')
    print('CV metrics:', out / 'cv_oof_metrics.json')
    print('These are GROUPED CROSS-VALIDATION metrics, not evaluator-test results.')


if __name__ == '__main__':
    main()
