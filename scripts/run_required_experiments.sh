#!/usr/bin/env bash
# Real experiments on TRAIN+STUDENT only. Resumable; stops on errors.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PATHSNAP_PYTHON:-python}"
mkdir -p artifacts/guideline_reports artifacts/reconstruction/student_test
"$PY" -c 'import joblib,optuna,pyarrow,sklearn,matplotlib,xgboost,folium; print("Core experiment dependencies OK")'

if [ ! -f outputs/features/features_train.parquet ]; then
  echo 'ERROR: missing outputs/features/features_train.parquet'; exit 1
fi
if [ "$(find Data/processed/student_test -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)" -ne 10 ]; then
  echo 'ERROR: missing 10 student-test processed trip folders'; exit 1
fi

"$PY" scripts/guideline_metrics.py --mode validation --output artifacts/guideline_reports/validation
"$PY" scripts/export_hyperparameters_history.py

STUDENT=outputs/features/features_student_test_labeled.parquet
if [ ! -s "$STUDENT" ]; then
  "$PY" scripts/build_student_test_labels.py --root Data/processed/student_test \
      --config outputs/features/feature_config.json --output "$STUDENT"
fi
"$PY" scripts/guideline_metrics.py --mode student --features "$STUDENT" \
    --output artifacts/guideline_reports/student_test

# Grouped CV (MLP/KNN may be very slow): reuse existing fold result CSVs when present.
if [ ! -s artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv ] || \
   [ "$(($(wc -l < artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv 2>/dev/null || echo 0)-1))" -lt 5 ]; then
  "$PY" -u scripts/cv_all_models.py --folds 3 --out artifacts/guideline_reports/all_models_cv \
    2>&1 | tee artifacts/guideline_reports/all_models_cv.log
fi

# Optuna/TPE never reads student or evaluator labels during optimisation.
if [ ! -s artifacts/logistic_regression_bayesian/model_all_80.joblib ] || \
   [ ! -s artifacts/logistic_regression_bayesian/optuna_trials.csv ] || \
   [ "$(($(wc -l < artifacts/logistic_regression_bayesian/optuna_trials.csv 2>/dev/null || echo 0)-1))" -lt 24 ]; then
  "$PY" -u scripts/tune_lr_bayesian.py --data outputs/features/features_train.parquet \
    --config outputs/features/feature_config.json --trials 24 --folds 5 \
    --out artifacts/logistic_regression_bayesian \
    2>&1 | tee artifacts/guideline_reports/bayesian_training.log
fi
"$PY" scripts/guideline_metrics.py --mode student --features "$STUDENT" \
    --output artifacts/guideline_reports/student_test

# Reconstruct student trips using model selected WITHOUT evaluator-tuned parameters.
if [ ! -s artifacts/reconstruction/student_test/comparison.json ]; then
  "$PY" -u scripts/reconstruct_routes.py \
    --model artifacts/logistic_regression_tuned/model_all_80.joblib \
    --features "$STUDENT" --data-root Data/processed/student_test \
    --output-dir artifacts/reconstruction/student_test --trip all \
    2>&1 | tee artifacts/guideline_reports/student_reconstruction.log
fi

"$PY" scripts/render_folium_guidelines.py --trip 00000000 \
  --data-root Data/processed/student_test \
  --reconstruction-dir artifacts/reconstruction/student_test --tiles none --show-reference
"$PY" scripts/guideline_audit_final.py
"$PY" scripts/build_report_evidence.py
printf '\nCOMPLETE: inspect artifacts/guideline_reports/guideline_audit.md\n'
