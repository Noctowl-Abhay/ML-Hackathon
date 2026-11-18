
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw"
PROCESSED = ROOT / "Data" / "processed"
REPORTS = ROOT / "outputs" / "results"

REPORTS.mkdir(parents=True, exist_ok=True)

summary = []

for split in ("train", "student_test"):
    split_dir = RAW / split
    output_dir = PROCESSED / split

    output_dir.mkdir(parents=True, exist_ok=True)

    for trip_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        trip_id = trip_dir.name

        track = np.loadtxt(
            trip_dir / f"{trip_id}.track", ndmin=2
        )

        nodes = np.loadtxt(
            trip_dir / f"{trip_id}.nodes", ndmin=2
        )

        arcs = np.loadtxt(
            trip_dir / f"{trip_id}.arcs",
            dtype=int, ndmin=2
        )

        route = np.loadtxt(
            trip_dir / f"{trip_id}.route",
            dtype=int, ndmin=1
        )

        # Create a GPS DataFrame
        df = pd.DataFrame(
            track,
            columns=["longitude", "latitude", "timestamp"]
        )

        # Check GPS coordinate validity
        valid_coords = (
            df["longitude"].between(-180, 180)
            & df["latitude"].between(-90, 90)
            & np.isfinite(df[["longitude", "latitude", "timestamp"]]).all(axis=1)
        )

        # Calculate time differences
        df["time_difference_s"] = df["timestamp"].diff()

        # Flag GPS gaps greater than one second
        df["gap_over_1s"] = (
            df["time_difference_s"] > 1
        ).fillna(False)

        # Flag exact duplicate GPS observations
        df["duplicate_observation"] = df.duplicated(
            subset=["longitude", "latitude", "timestamp"],
            keep="first"
        )

        # Preserve every original GPS observation
        df["valid_coordinates"] = valid_coords

        # Save processed trip
        trip_output = output_dir / trip_id
        trip_output.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            trip_output / f"{trip_id}_track_processed.csv",
            index=False
        )

        # Copy original road-network and reference files unchanged
        for suffix in ("nodes", "arcs", "route"):
            shutil.copy2(
                trip_dir / f"{trip_id}.{suffix}",
                trip_output / f"{trip_id}.{suffix}"
            )

        summary.append({
            "partition": split,
            "trip_id": trip_id,
            "gps_points": len(df),
            "invalid_coordinates": int((~valid_coords).sum()),
            "duplicate_observations": int(
                df["duplicate_observation"].sum()
            ),
            "gaps_over_1s": int(df["gap_over_1s"].sum()),
            "non_increasing_times": int(
                (df["time_difference_s"] <= 0).sum()
            ),
            "road_nodes": len(nodes),
            "road_arcs": len(arcs),
            "reference_route_arcs": len(route)
        })

result = pd.DataFrame(summary)

result.to_csv(
    REPORTS / "preprocessing_report.csv",
    index=False
)

print("\nPREPROCESSING COMPLETED")
print(result.groupby("partition").agg(
    trips=("trip_id", "count"),
    gps_points=("gps_points", "sum"),
    invalid_coordinates=("invalid_coordinates", "sum"),
    duplicate_observations=("duplicate_observations", "sum"),
    gaps_over_1s=("gaps_over_1s", "sum")
).to_string())

print("\nProcessed data saved to:", PROCESSED)
print("Report saved to:", REPORTS / "preprocessing_report.csv")
