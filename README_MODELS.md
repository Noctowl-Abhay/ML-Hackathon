# PathSnap — Five candidate-level classifiers

The scripts **do not** train on GPS-only preprocessed CSVs. They require a candidate-level feature table with `trip_id`, `label`, and `label_usable`, produced by candidate generation and reference-route alignment.

Run from the repository root:

```bash
python -m pip install -r requirements-models.txt
python scripts/inspect_dataset.py --root Data
python src/models/train_logistic_regression.py --data outputs/features/features_train.parquet
python src/models/train_knn.py --data outputs/features/features_train.parquet --max-train-rows 100000
python src/models/train_random_forest.py --data outputs/features/features_train.parquet
python src/models/train_xgboost.py --data outputs/features/features_train.parquet
python src/models/train_mlp.py --data outputs/features/features_train.parquet
python scripts/compare_models.py
```

Change `--data` to the actual candidate feature table path printed by inspection. For a controlled comparison, remove `--max-train-rows` or apply the same training cap to **all** models. All scripts use seed 42 and trip-level holdout (20%).

Default features are 11 known candidate features. For the team's complete 25-feature list, use `--feature-config outputs/features/feature_config.json` if its JSON contains `feature_cols` or `FEATURE_COLS`. All models must use the same list. Reference-route labels and IDs are excluded from inputs.

Models and validation predictions are saved to `artifacts/<model>/`. Never upload `student_test` ground-truth labels or train on them. Candidate top-1 is measured only on points with a positive candidate; also inspect candidate recall. Final route quality still requires HMM/Viterbi evaluation.

For large datasets, KNN may require substantial RAM/time; start with a smaller training cap for a smoke test, then use matched training subsets for fair comparisons.
