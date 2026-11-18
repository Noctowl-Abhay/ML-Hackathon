#!/usr/bin/env bash
# Full guideline experiments on permitted TRAIN + STUDENT data only.
# Does not touch evaluator data, publish private labels, or push to GitHub.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts/guideline_reports artifacts/reconstruction/student_test outputs/features
PY="${PATHSNAP_PYTHON:-python}"
"$PY" - <<'PY'
import joblib,optuna,pyarrow,sklearn,matplotlib,xgboost,folium,fastapi,uvicorn,pytest,shapely,pyproj
print('All required Python packages import successfully.')
PY
"$PY" -m py_compile \
  scripts/build_student_test_labels.py scripts/cv_all_models.py \
  scripts/guideline_metrics.py scripts/tune_lr_bayesian.py \
  src/reconstruction/api_compatible.py
"$PY" -m pytest -q tests/test_ui_contract.py
"$PY" scripts/final_guideline_audit.py --phase preflight

if [ ! -s outputs/features/features_train.parquet ]; then
  echo 'ERROR: missing training candidate table outputs/features/features_train.parquet' >&2
  echo 'Run: python -m features.build_candidates --data-root Data/processed --partition train' >&2
  exit 3
fi
TRIP_COUNT=$(find Data/processed/student_test -mindepth 1 -maxdepth 1 -type d -name '????????' 2>/dev/null | wc -l)
if [ "$TRIP_COUNT" != 10 ]; then echo "ERROR: expected 10 student-test trips, found $TRIP_COUNT" >&2; exit 4; fi

"$PY" scripts/guideline_metrics.py --mode validation --output artifacts/guideline_reports/validation
"$PY" scripts/export_hyperparameters_history.py
STUDENT="outputs/features/features_student_test_labeled.parquet"
if [ ! -s "$STUDENT" ]; then
  "$PY" -u scripts/build_student_test_labels.py --root Data/processed/student_test \
      --config outputs/features/feature_config.json --output "$STUDENT" \
      2>&1 | tee artifacts/guideline_reports/student_label_build.log
fi
"$PY" scripts/guideline_metrics.py --mode student --features "$STUDENT" \
      --output artifacts/guideline_reports/student_test

CV_SUMMARY=artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv
if [ -s "$CV_SUMMARY" ]; then
  CV_ROWS=$(tail -n +2 "$CV_SUMMARY" | wc -l)
else
  CV_ROWS=0
fi
if [ "$CV_ROWS" -lt 5 ]; then
  "$PY" -u scripts/cv_all_models.py --folds 3 \
      --out artifacts/guideline_reports/all_models_cv \
      2>&1 | tee artifacts/guideline_reports/all_models_cv.log
fi
TRIAL_ROWS=$("$PY" - <<'PY2'
import csv
from pathlib import Path
p=Path('artifacts/logistic_regression_bayesian/optuna_trials.csv')
if not p.is_file(): print(0)
else:
    with p.open(newline='') as f:
        print(sum(row.get('state')=='COMPLETE' for row in csv.DictReader(f)))
PY2
)
if [ ! -s artifacts/logistic_regression_bayesian/model_all_80.joblib ] || \
   [ "$TRIAL_ROWS" -lt 24 ]; then
  "$PY" -u scripts/tune_lr_bayesian.py \
      --data outputs/features/features_train.parquet \
      --config outputs/features/feature_config.json \
      --trials 24 --folds 5 --out artifacts/logistic_regression_bayesian \
      2>&1 | tee artifacts/guideline_reports/bayesian_training.log
fi
# Update real student test comparison to include newly fitted Bayesian LR.
"$PY" scripts/guideline_metrics.py --mode student --features "$STUDENT" \
      --output artifacts/guideline_reports/student_test
# Mandatory: untuned and Bayesian comparison on IDENTICAL GroupKFold splits.
if [ ! -s artifacts/guideline_reports/bayesian_comparison/cv_same_folds_comparison.csv ]; then
  "$PY" -u scripts/compare_untuned_bayesian_cv.py --folds 5 \
      2>&1 | tee artifacts/guideline_reports/bayesian_same_folds.log
fi
# Compare model-family route continuity on the 10 permitted student trips.
CONTINUITY=artifacts/guideline_reports/model_route_continuity/model_continuity_summary.csv
if [ ! -s "$CONTINUITY" ] || [ "$(($(wc -l < "$CONTINUITY")-1))" -lt 5 ]; then
  "$PY" -u scripts/compare_student_route_continuity.py --features "$STUDENT" \
      2>&1 | tee artifacts/guideline_reports/student_all_models_continuity.log
fi

if [ ! -s deployment/selected_model.joblib ]; then
  echo 'Generating Top-1 based deployment recommendation; team must review selection.'
  "$PY" scripts/select_final_model.py --model auto
fi
SELECTED_SHA=$(sha256sum deployment/selected_model.joblib | awk '{print $1}')
PREVIOUS_SHA="$(cat artifacts/reconstruction/student_test/selected_model.sha256 2>/dev/null || true)"
if [ "$SELECTED_SHA" != "$PREVIOUS_SHA" ] || [ ! -s artifacts/reconstruction/student_test/comparison.json ]; then
  "$PY" -u scripts/reconstruct_routes.py --model deployment/selected_model.joblib \
      --features "$STUDENT" --data-root Data/processed/student_test \
      --output-dir artifacts/reconstruction/student_test --trip all \
      2>&1 | tee artifacts/guideline_reports/student_hmm_reconstruction.log
  printf '%s\n' "$SELECTED_SHA" > artifacts/reconstruction/student_test/selected_model.sha256
fi
# Use actual student-test trip ID rather than assuming the first is 00000000.
FIRST_TRIP=$(find Data/processed/student_test -mindepth 1 -maxdepth 1 -type d -name '????????' -printf '%f\n' | sort | head -1)
"$PY" scripts/render_folium_guidelines.py --trip "$FIRST_TRIP" \
  --data-root Data/processed/student_test \
  --reconstruction-dir artifacts/reconstruction/student_test \
  --tiles none --show-reference
"$PY" scripts/build_report_evidence.py
"$PY" scripts/final_guideline_audit.py --phase development
"$PY" scripts/prepare_public_evidence.py
printf '\nDevelopment pipeline finished; check the audit and review selected model.\n'
printf 'Evidence: artifacts/guideline_reports/final_submission_audit.md\n'
printf 'Working report: artifacts/guideline_reports/REPORT_WORKING_DRAFT.md\n'
