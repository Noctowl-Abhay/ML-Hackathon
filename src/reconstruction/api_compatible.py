"""PathSnap FastAPI/UI contract adapter.

Preserves the original prediction pipeline; converts its known GeoJSON layers
(raw_gps, predicted_route) to the teammate UI's expected coordinate arrays.
The route reference is never returned to the frontend.
"""
from __future__ import annotations

import logging
import math
import time

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .api import app, trips as original_trips, reconstruct as original_reconstruct

logger = logging.getLogger("pathsnap.ui_api")

# Demo is intended for a private two-laptop LAN, not a publicly exposed API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class PredictBody(BaseModel):
    trip_id: str = Field(min_length=1, max_length=8, pattern=r"^[0-9]{1,8}$")


def _lat_lon(coordinates):
    """GeoJSON longitude/latitude -> teammate UI's latitude/longitude."""
    result = []
    for point in coordinates:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Invalid GeoJSON position")
        longitude, latitude = float(point[0]), float(point[1])
        if not (math.isfinite(longitude) and math.isfinite(latitude)):
            raise ValueError("Non-finite GeoJSON position")
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError("GeoJSON coordinates outside geographic range")
        result.append([latitude, longitude])
    return result


def convert_geometry(geojson):
    """Convert EXACT layers produced by decoder.geometry().

    raw_gps is a LineString of [lon,lat] GPS observations.
    predicted_route is a MultiLineString of [lon,lat] road polylines.
    Do not include the confidential reference route.
    """
    if not isinstance(geojson, dict) or geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected reconstruction FeatureCollection GeoJSON")
    gps = []
    predicted_route = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        kind = geom.get("type")
        coordinates = geom.get("coordinates") or []
        layer = props.get("layer")
        if layer == "raw_gps":
            if kind == "LineString":
                gps.extend(_lat_lon(coordinates))
            elif kind == "MultiPoint":
                gps.extend(_lat_lon(coordinates))
            else:
                raise ValueError(f"Unexpected raw_gps geometry: {kind}")
        elif layer == "predicted_route":
            if kind == "LineString":
                lines = [coordinates]
            elif kind == "MultiLineString":
                lines = coordinates
            else:
                raise ValueError(f"Unexpected predicted_route geometry: {kind}")
            for line in lines:
                if len(line) >= 2:
                    # A single polyline can represent MULTIPLE traversed road arcs.
                    # Do not invent individual arc IDs.
                    predicted_route.append({"coords": _lat_lon(line)})
        else:
            # Unknown layers (including any reference data) must not reach UI.
            logger.warning("Omitting unrecognised GeoJSON layer: %s", layer)
    return gps, predicted_route


def _public_metrics(report, stats):
    base = report.get("baseline") or {}
    hmm = report.get("hmm") or {}
    seq = report.get("sequence") or {}
    connected = stats.get("connected_arc_switches")
    broken = stats.get("disconnected_arc_switches")
    total = connected + broken if isinstance(connected, int) and isinstance(broken, int) else None
    values = {
        "baseline_top1_proxy": base.get("directed"),
        "hmm_top1_proxy": hmm.get("directed"),
        "hmm_undirected_top1_proxy": hmm.get("undirected"),
        "eligible_gps_points": report.get("eligible_points"),
        "eligible_gps_fraction": report.get("eligible_fraction"),
        "ordered_sequence_match_f1": seq.get("ordered_sequence_match_f1"),
        "reference_arc_coverage_aggregate": seq.get("reference_unique_arc_coverage"),
        "connected_arc_switches": connected,
        "disconnected_arc_switches": broken,
        "route_switch_continuity": (connected / total if total else None),
    }
    return {key: value for key, value in values.items() if value is not None}


@app.get("/trips", tags=["Teammate UI"])
def get_trips():
    return original_trips()


@app.post("/predict", tags=["Teammate UI"])
def predict(body: PredictBody, request: Request):
    began = time.perf_counter()
    trip_id = body.trip_id.zfill(8)
    logger.info("UI PREDICT request trip=%s client=%s", trip_id,
                request.client.host if request.client else "unknown")
    result = original_reconstruct(trip_id)  # Live ML scoring and HMM inside original API.
    try:
        gps, road_lines = convert_geometry(result.get("geojson"))
    except (ValueError, TypeError) as exc:
        logger.exception("Invalid reconstruction geometry for trip %s", trip_id)
        raise HTTPException(status_code=500, detail=f"Invalid reconstruction geometry: {exc}") from exc
    if not gps:
        raise HTTPException(status_code=500, detail="Backend returned no raw GPS geometry")
    if not road_lines:
        raise HTTPException(status_code=500, detail="Backend returned no predicted-route geometry")
    output = {
        "trip_id": trip_id,
        "gps": gps,
        "predicted_route": road_lines,
        "metrics": _public_metrics(result.get("metrics") or {}, result.get("stats") or {}),
        "latency_ms": round((time.perf_counter() - began) * 1000, 2),
    }
    logger.info("UI PREDICT complete trip=%s gps=%d polylines=%d ms=%.2f",
                trip_id, len(gps), len(road_lines), output["latency_ms"])
    return output


# Original /health, /api/trips, /api/reconstruct/{trip_id} remain unchanged.
