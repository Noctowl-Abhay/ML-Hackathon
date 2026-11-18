"""Generate candidate-level GPS map-matching features and derived labels."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


FEATURE_COLS = [
    "dist_m",
    "cand_rank",
    "dist_minus_min_m",
    "dist_ratio_min",
    "n_cands_in_radius",
    "frac_along",
    "frac_to_end",
    "arc_length_m",
    "heading_diff_deg",
    "heading_diff_undirected_deg",
    "heading_valid",
    "speed_mps",
    "gps_step_m",
    "dt_s",
    "prev_step_m",
    "next_step_m",
    "prev_dt_s",
    "next_dt_s",
    "arc_bearing_deg",
    "degree_from",
    "degree_to",
    "is_fallback",
    "endpoint_dist_m",
    "movement_valid",
    "projection_outside",
]


def segment_distance(point, starts, ends):
    delta = ends - starts
    length_sq = np.einsum("ij,ij->i", delta, delta)

    fraction_raw = (
        np.einsum("ij,ij->i", point - starts, delta)
        / np.maximum(length_sq, 1e-9)
    )

    fraction = np.clip(fraction_raw, 0, 1)
    closest = starts + fraction[:, None] * delta
    distance = np.linalg.norm(closest - point, axis=1)

    return distance, fraction, fraction_raw


def distances_per_point(points, starts, ends):
    delta = ends - starts
    length_sq = np.einsum("ij,ij->i", delta, delta)

    fraction = np.clip(
        np.einsum("ij,ij->i", points - starts, delta)
        / np.maximum(length_sq, 1e-9),
        0,
        1,
    )

    closest = starts + fraction[:, None] * delta
    return np.linalg.norm(closest - points, axis=1)


def align_route(points, arc_starts, arc_ends, route):
    """
    Monotonic dynamic-programming alignment.

    Finds a nondecreasing sequence of reference-route positions.
    Labels are derived estimates, NOT independently verified
    GPS-to-road ground truth.
    """
    n = len(points)
    m = len(route)

    if n * m > 60_000_000:
        raise ValueError(
            f"Route alignment too large: {n} GPS points x {m} arcs"
        )

    starts = arc_starts[route]
    ends = arc_ends[route]

    previous = np.zeros(m, dtype=np.float64)
    backtrack = np.empty((n, m), dtype=np.int32)

    for i, point in enumerate(points):
        distances, _, _ = segment_distance(point, starts, ends)
        distances = np.minimum(distances, 100.0)

        best_costs = np.minimum.accumulate(previous)
        best_indices = np.empty(m, dtype=np.int32)

        best = 0
        for j in range(m):
            if previous[j] < previous[best]:
                best = j
            best_indices[j] = best

        previous = distances + best_costs
        backtrack[i] = best_indices

    aligned = np.empty(n, dtype=np.int32)
    state = int(np.argmin(previous))

    for i in range(n - 1, -1, -1):
        aligned[i] = state
        state = int(backtrack[i, state])

    return aligned


def process_trip(folder, partition, k, radius, label_cutoff):
    trip_id = folder.name

    gps_file = folder / f"{trip_id}_track_processed.csv"
    nodes_file = folder / f"{trip_id}.nodes"
    arcs_file = folder / f"{trip_id}.arcs"
    route_file = folder / f"{trip_id}.route"

    df = pd.read_csv(gps_file).reset_index(drop=True)

    required = {"longitude", "latitude", "timestamp"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{gps_file}: missing columns {missing}")

    lon = df["longitude"].to_numpy(dtype=float)
    lat = df["latitude"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=float)

    nodes = np.loadtxt(nodes_file, dtype=float, ndmin=2)
    arcs = np.loadtxt(arcs_file, dtype=np.int64, ndmin=2)

    if nodes.shape[1] != 2 or arcs.shape[1] != 2:
        raise ValueError("Nodes and arcs must each have two columns.")

    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError(f"{trip_id}: invalid node indices in arcs")

    zone = int(
        np.clip(np.floor((float(np.nanmean(lon)) + 180) / 6) + 1, 1, 60)
    )

    epsg = (32600 if np.nanmean(lat) >= 0 else 32700) + zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        CRS.from_epsg(epsg),
        always_xy=True,
    )

    gps_x, gps_y = transformer.transform(lon, lat)
    gps = np.column_stack((gps_x, gps_y))

    node_x, node_y = transformer.transform(nodes[:, 0], nodes[:, 1])
    node_xy = np.column_stack((node_x, node_y))

    arc_starts = node_xy[arcs[:, 0]]
    arc_ends = node_xy[arcs[:, 1]]

    lengths = np.linalg.norm(arc_ends - arc_starts, axis=1)

    valid_arc_ids = np.where(lengths > 1e-6)[0]

    lines = [
        LineString((arc_starts[i], arc_ends[i]))
        for i in valid_arc_ids
    ]

    tree = STRtree(lines)

    degrees = np.bincount(
        arcs.ravel(),
        minlength=len(nodes),
    )

    bearings = (
        np.degrees(
            np.arctan2(
                arc_ends[:, 1] - arc_starts[:, 1],
                arc_ends[:, 0] - arc_starts[:, 0],
            )
        )
        % 360
    )

    n = len(gps)

    steps = np.full(n, np.nan)
    dt = np.full(n, np.nan)

    if n > 1:
        steps[1:] = np.linalg.norm(np.diff(gps, axis=0), axis=1)
        dt[1:] = np.diff(timestamps)

    speeds = np.divide(
        steps,
        dt,
        out=np.full_like(steps, np.nan),
        where=dt > 0,
    )

    previous_steps = np.roll(steps, 1)
    previous_steps[0] = np.nan

    next_steps = np.roll(steps, -1)
    next_steps[-1] = np.nan

    previous_dt = np.roll(dt, 1)
    previous_dt[0] = np.nan

    next_dt = np.roll(dt, -1)
    next_dt[-1] = np.nan

    headings = np.zeros(n)
    heading_valid = np.zeros(n, dtype=int)

    for i in range(n):
        left = max(0, i - 1)
        right = min(n - 1, i + 1)

        movement = gps[right] - gps[left]

        if (
            np.linalg.norm(movement) >= 1
            and timestamps[right] > timestamps[left]
        ):
            headings[i] = (
                np.degrees(np.arctan2(movement[1], movement[0])) % 360
            )
            heading_valid[i] = 1

    truth = None
    alignment = None
    true_distances = None

    if partition == "train":
        route = np.loadtxt(
            route_file,
            dtype=np.int64,
            ndmin=1,
        )

        if (
            len(route) == 0
            or route.min() < 0
            or route.max() >= len(arcs)
        ):
            raise ValueError(f"{trip_id}: invalid route arc IDs")

        alignment = align_route(
            gps,
            arc_starts,
            arc_ends,
            route,
        )

        truth = route[alignment]

        true_distances = distances_per_point(
            gps,
            arc_starts[truth],
            arc_ends[truth],
        )

    rows = []
    positive_points = 0
    usable_points = 0

    for i, point in enumerate(gps):
        shape_point = Point(point)

        nearby = np.asarray(
            tree.query(
                shape_point.buffer(radius),
                predicate="intersects",
            ),
            dtype=np.int64,
        )

        fallback = len(nearby) == 0

        if fallback:
            nearby = np.array(
                [int(tree.nearest(shape_point))],
                dtype=np.int64,
            )

        candidate_ids = valid_arc_ids[nearby]

        distances, fractions, raw_fractions = segment_distance(
            point,
            arc_starts[candidate_ids],
            arc_ends[candidate_ids],
        )

        order = np.argsort(distances, kind="stable")[:k]

        candidate_ids = candidate_ids[order]
        distances = distances[order]
        fractions = fractions[order]
        raw_fractions = raw_fractions[order]

        minimum_distance = float(distances[0])

        has_positive = bool(
            partition == "train"
            and truth[i] in candidate_ids
        )

        if has_positive:
            positive_points += 1

        label_usable = bool(
            partition == "train"
            and has_positive
            and true_distances[i] <= label_cutoff
        )

        if label_usable:
            usable_points += 1

        for rank, (arc_id, distance, fraction, raw_fraction) in enumerate(
            zip(
                candidate_ids,
                distances,
                fractions,
                raw_fractions,
            ),
            start=1,
        ):
            heading_difference = abs(
                (headings[i] - bearings[arc_id] + 180) % 360 - 180
            )

            endpoint_distance = min(
                np.linalg.norm(point - arc_starts[arc_id]),
                np.linalg.norm(point - arc_ends[arc_id]),
            )

            row = {
                "partition": partition,
                "trip_id": trip_id,
                "point_idx": i,
                "arc_id": int(arc_id),

                "label": (
                    int(arc_id == truth[i])
                    if partition == "train"
                    else np.nan
                ),

                "label_usable": int(label_usable),

                "label_arc_id": (
                    int(truth[i])
                    if partition == "train"
                    else -1
                ),

                "label_route_pos": (
                    int(alignment[i])
                    if partition == "train"
                    else -1
                ),

                "label_dist_m": (
                    float(true_distances[i])
                    if partition == "train"
                    else np.nan
                ),

                "point_has_positive": int(has_positive),

                "dist_m": float(distance),
                "cand_rank": rank,
                "dist_minus_min_m": float(
                    distance - minimum_distance
                ),
                "dist_ratio_min": float(
                    distance / (minimum_distance + 1)
                ),
                "n_cands_in_radius": (
                    int(len(nearby)) if not fallback else 0
                ),
                "frac_along": float(fraction),
                "frac_to_end": float(
                    min(fraction, 1 - fraction)
                ),
                "arc_length_m": float(lengths[arc_id]),

                "heading_diff_deg": (
                    float(heading_difference)
                    if heading_valid[i]
                    else 180.0
                ),

                "heading_diff_undirected_deg": (
                    float(
                        min(
                            heading_difference,
                            180 - heading_difference,
                        )
                    )
                    if heading_valid[i]
                    else 90.0
                ),

                "heading_valid": int(heading_valid[i]),
                "speed_mps": float(speeds[i]),
                "gps_step_m": float(steps[i]),
                "dt_s": float(dt[i]),
                "prev_step_m": float(previous_steps[i]),
                "next_step_m": float(next_steps[i]),
                "prev_dt_s": float(previous_dt[i]),
                "next_dt_s": float(next_dt[i]),
                "arc_bearing_deg": float(bearings[arc_id]),

                "degree_from": int(
                    degrees[arcs[arc_id, 0]]
                ),

                "degree_to": int(
                    degrees[arcs[arc_id, 1]]
                ),

                "is_fallback": int(fallback),
                "endpoint_dist_m": float(endpoint_distance),

                "movement_valid": int(
                    np.isfinite(speeds[i]) and speeds[i] > 0
                ),

                "projection_outside": int(
                    raw_fraction < 0 or raw_fraction > 1
                ),
            }

            rows.append(row)

    table = pd.DataFrame(rows)

    statistics = {
        "partition": partition,
        "trip_id": trip_id,
        "gps_points": n,
        "candidate_rows": len(rows),
        "points_with_positive_candidate": positive_points,
        "candidate_recall": (
            positive_points / n if partition == "train" else None
        ),
        "usable_label_points": usable_points,
    }

    return table, statistics


def main():
    parser = argparse.ArgumentParser(
        description="Generate PathSnap candidate-level features"
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/processed"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/features"),
    )

    parser.add_argument(
        "--partition",
        choices=["train", "student_test", "all"],
        default="train",
    )

    parser.add_argument(
        "--max-trips",
        type=int,
        default=0,
    )

    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--radius", type=float, default=50.0)
    parser.add_argument("--label-cutoff", type=float, default=30.0)

    args = parser.parse_args()

    if args.k < 1 or args.radius <= 0 or args.label_cutoff <= 0:
        parser.error("Invalid k, radius, or label cutoff")

    partitions = (
        ["train", "student_test"]
        if args.partition == "all"
        else [args.partition]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "feature_cols": FEATURE_COLS,
        "k": args.k,
        "radius_m": args.radius,
        "label_cutoff_m": args.label_cutoff,
        "label_notes": (
            "Labels are derived by monotonic route alignment. "
            "They are not independently verified GPS-point truth."
        ),
    }

    config_path = args.output_dir / "feature_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    for partition in partitions:
        partition_dir = args.data_root / partition

        if not partition_dir.is_dir():
            raise FileNotFoundError(
                f"Missing dataset directory: {partition_dir}"
            )

        trip_folders = sorted(
            path
            for path in partition_dir.iterdir()
            if path.is_dir()
        )

        if args.max_trips:
            trip_folders = trip_folders[:args.max_trips]

        if not trip_folders:
            raise FileNotFoundError(
                f"No trips found in {partition_dir}"
            )

        tables = []
        statistics = []

        for index, folder in enumerate(trip_folders, start=1):
            start_time = time.monotonic()

            table, stats = process_trip(
                folder,
                partition,
                args.k,
                args.radius,
                args.label_cutoff,
            )

            tables.append(table)
            statistics.append(stats)

            elapsed = time.monotonic() - start_time

            print(
                f"[{index}/{len(trip_folders)}] "
                f"{partition}/{folder.name}: "
                f"{stats['candidate_rows']:,} candidates, "
                f"{stats['usable_label_points']} usable labelled points, "
                f"{elapsed:.1f}s",
                flush=True,
            )

        combined = pd.concat(tables, ignore_index=True)

        output = args.output_dir / f"features_{partition}.parquet"

        try:
            combined.to_parquet(output, index=False)

        except ImportError:
            output = (
                args.output_dir / f"features_{partition}.csv.gz"
            )

            combined.to_csv(
                output,
                index=False,
                compression="gzip",
            )

        stats_output = (
            args.output_dir / f"candidate_stats_{partition}.csv"
        )

        pd.DataFrame(statistics).to_csv(
            stats_output,
            index=False,
        )

        print(f"\nSaved: {output}", flush=True)


if __name__ == "__main__":
    main()
