#!/usr/bin/env python3
"""Check LIVE API/UI contract; saves test evidence, never relies on stale files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(url, method="GET", payload=None, timeout=180):
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=body, method=method,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        message = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {message}") from error
    except URLError as error:
        raise RuntimeError(f"Cannot reach {url}: {error}. Start the API first!") from error


def validate(d, trip):
    required = {"trip_id", "gps", "predicted_route", "metrics", "latency_ms"}
    assert required <= d.keys(), f"Missing UI contract fields: {sorted(required - d.keys())}"
    assert d["trip_id"] == trip, f"Wrong trip: {d['trip_id']} != {trip}"
    assert isinstance(d["gps"], list) and len(d["gps"]) > 0, "No GPS geometry!"
    assert isinstance(d["predicted_route"], list) and len(d["predicted_route"]) > 0, "No route!"
    assert isinstance(d["metrics"], dict), "Metrics must be an object"
    assert isinstance(d["latency_ms"], (int, float)) and d["latency_ms"] >= 0
    for key in ("gps",):
        assert all(isinstance(p, list) and len(p) == 2 for p in d[key])
    for line in d["predicted_route"]:
        assert isinstance(line, dict) and len(line.get("coords", [])) >= 2
    raw = json.dumps(d).lower()
    for secret in ('"expanded_route_arc_ids"', '"reference_route_arcs"', '"ground_truth"', '"label_arc_id"'):
        assert secret not in raw, f"Forbidden raw reference field: {secret}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--all-trips", action="store_true")
    parser.add_argument("--final-evaluator", action="store_true",
                        help="Explicit acknowledgement: reserved evaluator trips are for FINAL testing only")
    parser.add_argument("--trip", help="Run a specific trip (otherwise the first returned trip)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/api_checks"))
    args = parser.parse_args()
    if (args.trip or args.all_trips) and not args.final_evaluator:
        parser.error("Explicit trip evaluation requires --final-evaluator: NEVER tune on evaluator trips")
    base = args.base.rstrip("/")
    health = request_json(base + "/health")
    print("HEALTH:", health, flush=True)
    assert all(health.get(x) for x in ("model_present", "features_present", "trip_root_present")), "Missing backend inference assets"
    trips = request_json(base + "/trips").get("trips", [])
    trip_ids = [str(t.get("trip_id", t.get("id"))) if isinstance(t, dict) else str(t) for t in trips]
    print("TRIPS:", len(trip_ids), trip_ids, flush=True)
    assert len(trip_ids) == 10, f"Guidelines require all 10 reserved trips; found {len(trip_ids)}"
    if not args.final_evaluator:
        print("STRUCTURAL CHECK PASS: 10 trips + inference assets; reserved evaluator predictions not run.")
        return
    selected = trip_ids if args.all_trips else ([args.trip.zfill(8)] if args.trip else trip_ids[:1])
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for trip in selected:
        t0 = time.perf_counter()
        result = request_json(base + "/predict", method="POST", payload={"trip_id": trip}, timeout=300)
        validate(result, trip)
        row = {
            "trip_id": trip,
            "gps_display_points": len(result["gps"]),
            "predicted_polylines": len(result["predicted_route"]),
            "latency_ms": result["latency_ms"],
            "client_round_trip_ms": round((time.perf_counter()-t0)*1000, 2),
            "metrics": result["metrics"],
            "contract_pass": True,
        }
        print("PASS", row, flush=True)
        summary.append(row)
        (args.out / f"{trip}_UI_response.json").write_text(json.dumps(result, indent=2))
    (args.out / "contract_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE: {len(summary)}/{len(selected)} live UI predictions passed; {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
