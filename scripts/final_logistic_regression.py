#!/usr/bin/env python3

"""
PathSnap - Final Logistic Regression Model

Commands:
    train : Train on all usable labelled training candidates.
    score : Score candidate road segments using a saved model.

The model uses only the input features listed in feature_config.json.
Route-derived labels are never included as input features.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = "outputs/features/features_train.parquet"
DEFAULT_CONFIG = "outputs/features/feature_config.json"
DEFAULT_MODEL = "artifacts/logistic_regression_final/model.joblib"


def load_features(config_path):
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    if isinstance(config, list):
        features = config
    elif isinstance(config, dict):
        features = config.get("feature_cols")

        if features is None:
            features = config.get("features")
    else:
        raise ValueError("Unsupported feature configuration format.")

    if not isinstance(features, list) or not features:
        raise ValueError(
            "Could not locate a feature list in feature_config.json"
        )

    if not all(isinstance(name, str) for name in features):
        raise ValueError("Feature names must be strings.")

    if len(features) != len(set(features)):
        raise ValueError("Duplicate feature names detected.")

    forbidden = {
        "label",
        "label_usable",
        "label_arc_id",
        "label_route_pos",
        "label_dist_m",
        "point_has_positive",
        "trip_id",
        "point_idx",
        "arc_id",
        "partition",
    }

    leaked = forbidden.intersection(features)

    if leaked:
        raise ValueError(
            f"Potential data leakage: {sorted(leaked)}"
        )

    return features


def load_table(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Expected a .parquet or .csv file.")


def get_input_matrix(df, features):
    missing = [name for name in features if name not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    values = df.loc[:, features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return (
        values.replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float32)
    )


def train(args):
    start = time.perf_counter()

    print("\n" + "=" * 60)
    print("PATHSNAP - FINAL LOGISTIC REGRESSION TRAINING")
    print("=" * 60)

    features = load_features(args.feature_config)
    df = load_table(args.data)

    required = {"trip_id", "label", "label_usable"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Training data missing required columns: {sorted(missing)}"
        )

    total_trips = df["trip_id"].nunique()

    print(f"\nAvailable trips: {total_trips}")
    print(f"Total candidate rows: {len(df):,}")
    print(f"Input features: {len(features)}")

    if total_trips != 80:
        raise ValueError(
            f"Expected 80 training trips, found {total_trips}. "
            "Regenerate the complete training features first."
        )

    # Only rows with usable reference labels enter training.
    usable = df.loc[df["label_usable"].eq(True)].copy()

    if usable.empty:
        raise ValueError("No usable labelled candidates found.")

    if usable["label"].isna().any():
        raise ValueError("Missing labels in usable training rows.")

    y = usable["label"].astype(np.int8).to_numpy()

    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Labels must be binary: 0 or 1.")

    if len(np.unique(y)) != 2:
        raise ValueError("Both positive and negative labels are required.")

    usable_trips = usable["trip_id"].nunique()

    print(f"Usable labelled rows: {len(usable):,}")
    print(f"Trips with usable labels: {usable_trips}")
    print(f"Positive candidate rate: {y.mean():.4f}")

    if usable_trips != 80:
        raise ValueError(
            f"Only {usable_trips} trips contain usable labels. "
            "Check the feature generation and reference alignment."
        )

    print("\nBuilding training matrix...")
    X = get_input_matrix(usable, features)

    print(f"Training matrix: {X.shape}")

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining Logistic Regression on all 80 trips...")
    fit_start = time.perf_counter()

    model.fit(X, y)

    fit_seconds = time.perf_counter() - fit_start

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    package = {
        "model": model,
        "feature_cols": features,
        "model_name": "logistic_regression",
        "training_trips": int(usable_trips),
        "training_rows": int(len(usable)),
        "positive_rate": float(y.mean()),
    }

    joblib.dump(package, model_path)

    metadata = {
        "model": "logistic_regression",
        "training_trips": int(usable_trips),
        "training_rows": int(len(usable)),
        "input_features": len(features),
        "feature_cols": features,
        "positive_rate": float(y.mean()),
        "fit_seconds": float(fit_seconds),
        "total_seconds": float(time.perf_counter() - start),
        "class_weight": "balanced",
        "C": 1.0,
        "max_iter": 2000,
        "solver": "lbfgs",
        "seed": 42,
        "note": (
            "Final model fitted on all 80 training trips. "
            "No independent final-model validation metrics."
        ),
    }

    metadata_path = model_path.parent / "training_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Training trips: {usable_trips}")
    print(f"Training candidates: {len(usable):,}")
    print(f"Training time: {fit_seconds:.2f} seconds")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")


def score(args):
    print("\n" + "=" * 60)
    print("PATHSNAP - LOGISTIC REGRESSION CANDIDATE SCORING")
    print("=" * 60)

    package = joblib.load(args.model)

    model = package["model"]
    features = package["feature_cols"]

    df = load_table(args.data)

    print(f"\nCandidate rows: {len(df):,}")
    print(f"Input features: {len(features)}")

    required = {"trip_id", "point_idx", "arc_id"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing candidate identifiers: {sorted(missing)}"
        )

    scores = np.empty(len(df), dtype=np.float32)

    print("\nPredicting candidate probabilities...")

    for start in range(0, len(df), args.batch_size):
        end = min(start + args.batch_size, len(df))

        chunk = df.iloc[start:end]
        X = get_input_matrix(chunk, features)

        scores[start:end] = model.predict_proba(X)[:, 1]

    output = df[
        ["trip_id", "point_idx", "arc_id"]
    ].copy()

    if "partition" in df.columns:
        output.insert(0, "partition", df["partition"].to_numpy())

    output["probability"] = scores

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_parquet(output_path, index=False)

    group_cols = ["trip_id", "point_idx"]

    if "partition" in output.columns:
        group_cols.insert(0, "partition")

    # Select the highest-scoring candidate for each GPS observation.
    top_indices = output.groupby(
        group_cols,
        sort=False,
        observed=True,
    )["probability"].idxmax()

    best = output.loc[top_indices].copy()

    best = best.sort_values(group_cols).reset_index(drop=True)

    best_path = output_path.with_name(
        output_path.stem + "_top1.parquet"
    )

    best.to_parquet(best_path, index=False)

    print("\nSCORING COMPLETED")
    print(f"All candidate scores: {output_path}")
    print(f"Top-1 candidate predictions: {best_path}")
    print(f"GPS observations scored: {len(best):,}")

    print(
        "\nNOTE: These are independent per-point candidate "
        "predictions, not yet a connected route reconstruction."
    )


def main():
    parser = argparse.ArgumentParser(
        description="PathSnap final Logistic Regression model"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Train on all 80 trips",
    )

    train_parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )

    train_parser.add_argument(
        "--feature-config",
        default=DEFAULT_CONFIG,
    )

    train_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    score_parser = subparsers.add_parser(
        "score",
        help="Score candidate road segments",
    )

    score_parser.add_argument(
        "--data",
        required=True,
    )

    score_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    score_parser.add_argument(
        "--output",
        default="outputs/predictions/candidate_scores.parquet",
    )

    score_parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
    )

    args = parser.parse_args()

    if args.command == "train":
        train(args)

    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
