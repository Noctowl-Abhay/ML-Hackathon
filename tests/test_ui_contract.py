"""Offline contract tests: mock original reconstruction (no evaluator access)."""
from __future__ import annotations

from fastapi.testclient import TestClient
import src.reconstruction.api_compatible as adapter

SAMPLE = {
    "trip_id": "00000023",
    "metrics": {
        "available": True,
        "baseline": {"directed": 0.9},
        "hmm": {"directed": 0.95, "undirected": 0.96},
        "eligible_points": 2,
        "sequence": {"ordered_sequence_match_f1": 0.6,
                     "reference_route_arcs": 200},
    },
    "stats": {"connected_arc_switches": 2, "disconnected_arc_switches": 1,
              "expanded_route_arc_ids": [45, 46]},
    "geojson": {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"layer": "raw_gps"},
         "geometry": {"type": "LineString", "coordinates": [[77.60, 12.90], [77.61, 12.91]]}},
        {"type": "Feature", "properties": {"layer": "predicted_route"},
         "geometry": {"type": "MultiLineString", "coordinates": [
             [[77.60, 12.90], [77.605, 12.905]],
             [[77.605, 12.905], [77.61, 12.91]]] }},
        {"type": "Feature", "properties": {"layer": "ground_truth"},
         "geometry": {"type": "LineString", "coordinates": [[77.6, 12.9], [77.8, 13.2]]}}
    ]}
}


def test_ui_predict_contract(monkeypatch):
    monkeypatch.setattr(adapter, "original_reconstruct", lambda trip: SAMPLE)
    client = TestClient(adapter.app)
    response = client.post("/predict", json={"trip_id": "23"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result) == {"trip_id", "gps", "predicted_route", "metrics", "latency_ms"}
    assert result["trip_id"] == "00000023"
    assert result["gps"] == [[12.9, 77.6], [12.91, 77.61]]
    assert len(result["predicted_route"]) == 2
    assert result["predicted_route"][0]["coords"][0] == [12.9, 77.6]
    assert result["metrics"]["hmm_top1_proxy"] == 0.95
    assert abs(result["metrics"]["route_switch_continuity"] - 2/3) < 1e-7
    assert result["latency_ms"] >= 0
    assert "reference_route_arcs" not in response.text
    assert "expanded_route_arc_ids" not in response.text
    assert "ground_truth" not in response.text


def test_ui_trips_contract(monkeypatch):
    monkeypatch.setattr(adapter, "original_trips", lambda: {"trips": [
        {"id": "00000023", "gps_points": 2}]})
    response = TestClient(adapter.app).get("/trips")
    assert response.status_code == 200
    assert response.json()["trips"][0]["id"] == "00000023"


def test_incorrect_input_is_rejected():
    client = TestClient(adapter.app)
    for bad in ("../../etc/passwd", "not-a-trip", "9999999999"):
        response = client.post("/predict", json={"trip_id": bad})
        assert response.status_code == 422


def test_post_cors_preflight():
    client = TestClient(adapter.app)
    response = client.options(
        "/predict", headers={"Origin": "http://localhost:5500",
                             "Access-Control-Request-Method": "POST",
                             "Access-Control-Request-Headers": "content-type"})
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
