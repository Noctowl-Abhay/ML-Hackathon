"""FastAPI evaluation API: live model scoring, live Viterbi, server-only reference."""
from __future__ import annotations
import os
import time
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib
import json
from features.build_candidates import process_trip
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .decoder import RoadNetwork,Config,score_candidates,decode_trip,evaluation_metrics,geometry,load_gps,sequence_metrics

MODEL_FILE=Path(os.getenv('PATHSNAP_MODEL','artifacts/logistic_regression_tuned/model_all_80.joblib'))
FEATURES_FILE=Path(os.getenv('PATHSNAP_FEATURES','outputs/features/features_evaluator_test.parquet'))
CONFIG_FILE=Path(os.getenv('PATHSNAP_CONFIG','outputs/features/feature_config.json'))
TRIP_ROOT=Path(os.getenv('PATHSNAP_TRIPS','Data/processed/evaluator_test'))

app=FastAPI(title='PathSnap — Route Reconstruction API', version='1.0')
# For private hackathon LAN. Restrict CORS origins for deployment on a public network.
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['GET'],allow_headers=['*'])

@lru_cache(maxsize=1)
def resources():
    if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
        raise RuntimeError(f'Missing model/features: {MODEL_FILE} or {FEATURES_FILE}')
    package=joblib.load(MODEL_FILE)
    feats=(pd.read_parquet(FEATURES_FILE) if str(FEATURES_FILE).endswith('.parquet')
           else pd.read_csv(FEATURES_FILE))
    feats['trip_id']=feats['trip_id'].astype(str).str.zfill(8)
    return package,feats

@lru_cache(maxsize=10)
def network(tid):
    return RoadNetwork(TRIP_ROOT/tid)

@app.get('/health')
def health():
    return {'status':'ok','model_present':MODEL_FILE.exists(),
            'features_present':FEATURES_FILE.exists(), 'trip_root_present':TRIP_ROOT.exists()}

@app.get('/api/trips')
def trips():
    _,feats=resources()
    items=[]
    for tid,group in feats.groupby('trip_id'):
        if (TRIP_ROOT/tid).is_dir():
            items.append({'id':tid,'gps_points':int(group['point_idx'].nunique())})
    return {'trips':items}

@app.get('/api/reconstruct/{trip_id}')
def reconstruct(trip_id:str):
    start=time.perf_counter()
    tid=str(trip_id).zfill(8)
    package, feats=resources()
    df=feats.loc[feats['trip_id'].eq(tid)].copy()
    if df.empty or not (TRIP_ROOT/tid).is_dir():
        raise HTTPException(status_code=404,detail='Unknown evaluator trip ID')
    try:
        # Full live inference from processed GPS+nodes+arcs. 'student_test'
        # branch never reads .route or assigns reference labels.
        config=json.loads(CONFIG_FILE.read_text())
        candidates,_=process_trip(TRIP_ROOT/tid, 'student_test',
                                  k=int(config.get('k',8)),
                                  radius=float(config.get('radius_m',50)),
                                  label_cutoff=float(config.get('label_cutoff_m',30)))
        if candidates.empty:
            raise ValueError('Feature extraction produced no candidates')
        scored=score_candidates(candidates,package)
        net=network(tid)
        best,stats=decode_trip(scored,net,Config())
        eval_result=evaluation_metrics(df,best,net)
        eval_result['sequence']=sequence_metrics(net,stats,TRIP_ROOT/tid)
        # Do not expose reference route lengths or IDs to evaluator-facing UI.
        eval_result['sequence'].pop('reference_route_arcs',None)
        gps=load_gps(TRIP_ROOT/tid)
        geo=geometry(net,best,stats,gps)
        # NO reference arc IDs, route, alignment positions or truth geometry sent.
        print(f'[PathSnap API] reconstructed {tid}: {len(best)} GPS points, '
              f"HMM direct={eval_result.get('hmm',{}).get('directed')}; "
              f"broken={stats['disconnected_arc_switches']}",flush=True)
        return {'trip_id':tid,
                'metrics':eval_result,
                'stats':{k:v for k,v in stats.items() if k!='expanded_route_arc_ids'},
                'geojson':geo,
                'elapsed_seconds':round(time.perf_counter()-start,3)}
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'Reconstruction failed: {e}') from e
