# PathSnap UI (Laptop 2)

A static front end. It only talks to the API and never sees ground-truth labels.

## Run
1. On Laptop 1 start the API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
2. On Laptop 2 serve this folder: `cd ui && python -m http.server 5500`
3. Open http://localhost:5500, type Laptop 1's address (e.g. `http://192.168.1.20:8000`) in the header box, press Connect.
   The address is remembered, or set it in `config.js`, or use `?api=http://IP:8000`.

No internet needed: there are no external libraries or map tiles.

## API contract the UI expects
`GET /trips` returns `{"trips": ["12", "37", ...]}` (a plain list, or objects with `trip_id`, also works).

`POST /predict` with body `{"trip_id": "12"}` returns:
```json
{
  "trip_id": "12",
  "gps": [[lat, lon], ...],
  "predicted_route": [ {"arc_id": 5, "coords": [[lat, lon], [lat, lon]]}, ... ],
  "metrics": {"top1_accuracy": 0.0, "route_continuity": 0.0},
  "latency_ms": 0
}
```
`metrics` is computed on the server against its private reference data. Every key/value there is shown automatically, so add or rename metrics without touching the UI. Do not return the reference route.

## CORS (required, UI and API are on different machines)
```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```
Also allow port 8000 through Laptop 1's firewall, and put both laptops on the same Wi-Fi or hotspot.

## Repo placement
Copy the `ui/` folder to the repo root (matches the suggested structure).
