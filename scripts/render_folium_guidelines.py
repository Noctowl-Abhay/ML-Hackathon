#!/usr/bin/env python3
"""Folium map: GPS, relevant road network, predicted route, optional STUDENT reference.

Default no basemap tiles: suitable when tile.openstreetmap.org returns 403.
Reference is allowed ONLY for student_test report visualizations, never UI.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import folium
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trip", required=True)
    p.add_argument("--data-root", type=Path, default=Path("Data/processed/student_test"))
    p.add_argument("--reconstruction-dir", type=Path,
                   default=Path("artifacts/reconstruction/student_test"))
    p.add_argument("--show-reference", action="store_true")
    p.add_argument("--tiles", choices=["none", "osm"], default="none")
    p.add_argument("--max-road-arcs", type=int, default=2500)
    args = p.parse_args()
    if args.show_reference and args.data_root.name != "student_test":
        p.error("Reference drawing is strictly permitted for student_test report maps only")
    trip_id = str(args.trip).zfill(8)
    data = args.data_root / trip_id
    reconstruction = args.reconstruction_dir / trip_id
    geo = json.loads((reconstruction / "route.geojson").read_text())
    features = {f.get("properties", {}).get("layer"): f for f in geo["features"]}
    gps = np.asarray(features["raw_gps"]["geometry"]["coordinates"], dtype=float)
    lines = features["predicted_route"]["geometry"]["coordinates"]
    if not len(gps):
        raise ValueError("GPS track empty")
    map_ = folium.Map(location=[float(np.median(gps[:,1])), float(np.median(gps[:,0]))],
                      zoom_start=14, control_scale=True,
                      tiles="OpenStreetMap" if args.tiles == "osm" else None)
    folium.Marker([float(gps[0,1]), float(gps[0,0])],
                  tooltip="START", icon=folium.Icon(color="green")).add_to(map_)
    folium.Marker([float(gps[-1,1]), float(gps[-1,0])],
                  tooltip="END", icon=folium.Icon(color="red")).add_to(map_)

    nodes = np.loadtxt(data / f"{trip_id}.nodes", ndmin=2)
    arcs = np.loadtxt(data / f"{trip_id}.arcs", dtype=np.int64, ndmin=2)
    lo = gps.min(axis=0)
    hi = gps.max(axis=0)
    # Pad by ~250-350m to show surrounding roads around GPS trajectory.
    pad = np.array([0.0035, 0.0030])
    inside = ((nodes[:, 0] >= lo[0]-pad[0]) & (nodes[:, 0] <= hi[0]+pad[0]) &
              (nodes[:, 1] >= lo[1]-pad[1]) & (nodes[:, 1] <= hi[1]+pad[1]))
    selected = np.flatnonzero(inside[arcs[:,0]] & inside[arcs[:,1]])
    if len(selected) > args.max_road_arcs:
        selected = selected[np.linspace(0, len(selected)-1, args.max_road_arcs, dtype=int)]
    network = folium.FeatureGroup(name="Road network (sample of relevant arcs)", show=True)
    for aid in selected:
        a, b = nodes[arcs[aid]]
        folium.PolyLine([[float(a[1]), float(a[0])], [float(b[1]), float(b[0])]],
                        color="#929cab", weight=1.1, opacity=.4).add_to(network)
    network.add_to(map_)

    raw = folium.FeatureGroup(name="Raw GPS track", show=True)
    folium.PolyLine([[float(lat), float(lon)] for lon, lat in gps],
                    color="#3067df", weight=2, opacity=.6).add_to(raw)
    raw.add_to(map_)

    predicted = folium.FeatureGroup(name="Predicted HMM/Viterbi route", show=True)
    for line in lines:
        if len(line) > 1:
            folium.PolyLine([[float(lat), float(lon)] for lon, lat in line],
                            color="#00a873", weight=4, opacity=.95).add_to(predicted)
    predicted.add_to(map_)

    if args.show_reference:
        route_path = data / f"{trip_id}.route"
        reference = np.loadtxt(route_path, dtype=np.int64, ndmin=1)
        official = folium.FeatureGroup(name="Student-test reference (report ONLY)", show=False)
        for aid in reference:
            u, v = arcs[int(aid)]
            a, b = nodes[int(u)], nodes[int(v)]
            folium.PolyLine([[float(a[1]),float(a[0])],[float(b[1]),float(b[0])]],
                            color="#dd9200", weight=2.8, opacity=.8).add_to(official)
        official.add_to(map_)
    folium.LayerControl(collapsed=False).add_to(map_)
    map_.fit_bounds([[float(lo[1]),float(lo[0])],[float(hi[1]),float(hi[0])]])
    out = reconstruction / "road_network_map.html"
    map_.save(out)
    print(f"SAVED {out}: {len(gps)} GPS samples, {len(selected)} road arcs, "
          f"{len(lines)} predicted polylines, reference={args.show_reference}")
    if args.show_reference:
        print("WARNING: reference included. For internal report only; never serve via evaluator UI.")


if __name__ == "__main__":
    main()
