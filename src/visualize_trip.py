from pathlib import Path
import folium
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRIP_ID = '00000005'
TRIP = ROOT / 'data' / 'raw' / 'train' / TRIP_ID
OUTPUT = ROOT / 'outputs' / 'maps' / f'{TRIP_ID}_exploration.html'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

track = np.loadtxt(TRIP / f'{TRIP_ID}.track', ndmin=2)
nodes = np.loadtxt(TRIP / f'{TRIP_ID}.nodes', ndmin=2)
arcs = np.loadtxt(TRIP / f'{TRIP_ID}.arcs', dtype=int, ndmin=2)
route = np.loadtxt(TRIP / f'{TRIP_ID}.route', dtype=int, ndmin=1)

m = folium.Map(
    location=[float(track[:, 1].mean()), float(track[:, 0].mean())],
    zoom_start=13,
    tiles="CartoDB positron"
)
road_layer = folium.FeatureGroup(name='Road network', show=True)
for start_node, end_node in arcs:
    a, b = nodes[start_node], nodes[end_node]
    folium.PolyLine([(a[1], a[0]), (b[1], b[0])],
                    color='gray', weight=1, opacity=0.3).add_to(road_layer)
road_layer.add_to(m)

gps_layer = folium.FeatureGroup(name='GPS observations', show=True)
for lon, lat, _ in track:
    folium.CircleMarker([lat, lon], radius=2, color='blue',
                        fill=True, fill_opacity=0.7).add_to(gps_layer)
gps_layer.add_to(m)

route_layer = folium.FeatureGroup(name='Reference route', show=True)
for arc_id in route:
    a, b = nodes[arcs[arc_id]]
    folium.PolyLine([(a[1], a[0]), (b[1], b[0])],
                    color='green', weight=4, opacity=0.8).add_to(route_layer)
route_layer.add_to(m)

folium.LayerControl().add_to(m)
m.fit_bounds([(float(track[:, 1].min()), float(track[:, 0].min())),
              (float(track[:, 1].max()), float(track[:, 0].max()))])
m.save(OUTPUT)
print(f'Saved map: {OUTPUT}')
