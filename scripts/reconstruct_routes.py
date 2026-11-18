#!/usr/bin/env python3
"""CLI reconstruction and reference-aligned comparisons for PathSnap."""
import argparse
import json
from pathlib import Path
import sys
import time
import joblib
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.reconstruction.decoder import (RoadNetwork, Config, score_candidates,
                                        decode_trip, evaluation_metrics, geometry, load_gps, sequence_metrics)


def run_trip(trip_id, features, model_package, data_root, out_root, cfg):
    folder=Path(data_root)/trip_id
    df=features.loc[features['trip_id'].astype(str).str.zfill(8).eq(trip_id)].copy()
    if df.empty:
        raise ValueError('No candidates for '+trip_id)
    if not folder.exists():
        raise FileNotFoundError(folder)
    print(f'[{trip_id}] scoring {len(df):,} candidates...', flush=True)
    t0=time.perf_counter()
    scored=score_candidates(df,model_package)
    network=RoadNetwork(folder)
    chosen,stats=decode_trip(scored,network,cfg)
    evaluation=evaluation_metrics(df,chosen,network)
    evaluation['sequence']=sequence_metrics(network,stats,folder)
    trip_out=Path(out_root)/trip_id
    trip_out.mkdir(parents=True,exist_ok=True)
    chosen.to_csv(trip_out/'gps_point_predictions.csv',index=False)
    pd.DataFrame({'arc_id':stats['expanded_route_arc_ids']}).to_csv(
        trip_out/'expanded_route_arcs.csv',index=False)
    track=load_gps(folder)
    map_data=geometry(network,chosen,stats,track)
    with (trip_out/'route.geojson').open('w') as f:
        json.dump(map_data,f)
    # Never save original .route or reference per-GPS labels to UI outputs.
    stats_to_json={k:v for k,v in stats.items() if k!='expanded_route_arc_ids'}
    summary={'trip_id':trip_id, 'reconstruction':stats_to_json,
             'evaluation':evaluation, 'elapsed_seconds':round(time.perf_counter()-t0,3)}
    with (trip_out/'summary.json').open('w') as f:
        json.dump(summary,f,indent=2)
    print(f"[{trip_id}] baseline direct={evaluation.get('baseline',{}).get('directed','N/A')}, "
          f"Viterbi direct={evaluation.get('hmm',{}).get('directed','N/A')}; "
          f"broken switches={stats['disconnected_arc_switches']}; "
          f"time={summary['elapsed_seconds']}s",flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='artifacts/logistic_regression_tuned/model_all_80.joblib')
    parser.add_argument('--features',default='outputs/features/features_evaluator_test.parquet')
    parser.add_argument('--data-root',default='Data/processed/evaluator_test')
    parser.add_argument('--output-dir',default='artifacts/reconstruction')
    parser.add_argument('--trip',default='all',help='all or zero-padded eight-digit trip id')
    parser.add_argument('--beta',type=float,default=30)
    parser.add_argument('--emission-weight',type=float,default=1)
    parser.add_argument('--disconnected-penalty',type=float,default=12)
    parser.add_argument('--gap-seconds',type=float,default=30)
    args=parser.parse_args()
    package=joblib.load(args.model)
    data=(pd.read_parquet(args.features) if str(args.features).endswith('.parquet')
          else pd.read_csv(args.features))
    trips=sorted(data['trip_id'].astype(str).str.zfill(8).unique())
    if args.trip!='all':
        tid=str(args.trip).zfill(8)
        if tid not in trips: raise ValueError(f'{tid} not found: {trips}')
        trips=[tid]
    cfg=Config(emission_weight=args.emission_weight,distance_beta_m=args.beta,
               disconnected_penalty=args.disconnected_penalty,gap_seconds=args.gap_seconds)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for tid in trips:
        summaries.append(run_trip(tid,data,package,args.data_root,out,cfg))
    (out/'comparison.json').write_text(json.dumps(summaries,indent=2))
    print('\nCOMPLETE:',out/'comparison.json')

if __name__=='__main__': main()
