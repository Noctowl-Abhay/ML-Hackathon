#!/usr/bin/env python3
"""Generate required Folium map from reconstruction output; optional server-only reference."""
import argparse
import json
from pathlib import Path
import folium
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trip',required=True)
    p.add_argument('--reconstruction-dir',default='artifacts/reconstruction')
    p.add_argument('--data-root',default='Data/processed/evaluator_test')
    p.add_argument('--show-reference',action='store_true',help='SERVER-SIDE REPORT ONLY; do not send to UI')
    args=p.parse_args()
    trip=str(args.trip).zfill(8)
    root=Path(args.reconstruction_dir)/trip
    content=json.loads((root/'route.geojson').read_text())
    gps=np.asarray(content['features'][0]['geometry']['coordinates'],float)
    route_lines=content['features'][1]['geometry']['coordinates']
    if not len(gps): raise ValueError('No GPS points')
    mapa=folium.Map(location=[float(np.median(gps[:,1])),float(np.median(gps[:,0]))],
                   zoom_start=14,control_scale=True,tiles='OpenStreetMap')
    raw=folium.FeatureGroup(name='Raw GPS',show=True)
    folium.PolyLine([[float(lat),float(lon)] for lon,lat in gps],
                    color='#529cf6',weight=2,opacity=.7,tooltip='Raw GPS trace').add_to(raw)
    raw.add_to(mapa)
    pred=folium.FeatureGroup(name='Viterbi predicted route',show=True)
    for line in route_lines:
        if len(line)>=2:
            folium.PolyLine([[float(lat),float(lon)] for lon,lat in line],
                            color='#15a977',weight=5,opacity=.9,tooltip='Predicted route').add_to(pred)
    pred.add_to(mapa)
    if args.show_reference:
        folder=Path(args.data_root)/trip
        # Reference route only included in explicitly requested OFFLINE map.
        nodes=np.loadtxt(folder/f'{trip}.nodes',dtype=float,ndmin=2)
        arcs=np.loadtxt(folder/f'{trip}.arcs',dtype=int,ndmin=2)
        reference=np.loadtxt(folder/f'{trip}.route',dtype=int,ndmin=1)
        group=folium.FeatureGroup(name='REFERENCE - server only',show=True)
        for aid in reference:
            u,v=arcs[int(aid)]
            a,b=nodes[int(u)],nodes[int(v)]
            folium.PolyLine([[float(a[1]),float(a[0])], [float(b[1]),float(b[0])]],
                            color='#f2a73e',weight=3,opacity=.8,tooltip='Server-only reference').add_to(group)
        group.add_to(mapa)
    folium.LayerControl().add_to(mapa)
    out=root/('internal_reference_comparison.html' if args.show_reference else 'route_map.html')
    mapa.save(out)
    print('Saved:',out)
    if args.show_reference:
        print('CONFIDENTIAL: contains ground-truth geometry; do not ship to UI')

if __name__=='__main__':main()
