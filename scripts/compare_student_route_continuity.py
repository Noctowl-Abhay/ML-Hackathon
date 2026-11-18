#!/usr/bin/env python3
"""Real student-test classifier/HMM continuity comparison for ALL five families.

Runs 10 student trips per saved baseline model. Does NOT use evaluator trips.
Resumes safely from model-specific JSON results. Only publishes numeric summaries.
"""
from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib,numpy as np,pandas as pd
from src.reconstruction.decoder import RoadNetwork,Config,score_candidates,decode_trip,evaluation_metrics,sequence_metrics

NAMES=['logistic_regression','distance_weighted_knn','random_forest','xgboost','deep_mlp']
ALT={'distance_weighted_knn':'knn','deep_mlp':'mlp'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',default='outputs/features/features_student_test_labeled.parquet')
    p.add_argument('--root',default='Data/processed/student_test')
    p.add_argument('--out',default='artifacts/guideline_reports/model_route_continuity')
    p.add_argument('--only',choices=NAMES)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=(pd.read_parquet(a.features) if str(a.features).endswith('.parquet')
        else pd.read_csv(a.features,dtype={'trip_id':str}))
    df['trip_id']=df.trip_id.astype(str).str.zfill(8)
    tids=sorted(df.trip_id.unique())
    if len(tids)!=10:raise ValueError('Exactly 10 student-testing trips required')
    for name in ([a.only] if a.only else NAMES):
        source=Path('artifacts')/name/'model.joblib'
        if not source.exists(): source=Path('artifacts')/ALT.get(name,name)/'model.joblib'
        if not source.is_file():raise FileNotFoundError(source)
        package=joblib.load(source)
        if 'pipeline' in package:
            package={'model':package['pipeline'],'feature_cols':package['features']}
        if 'model' not in package or 'feature_cols' not in package:
            raise ValueError('Unexpected model serialization for '+name)
        rows=[]
        for tid in tids:
            start=time.perf_counter()
            local=df.loc[df.trip_id.eq(tid)].copy()
            scored=score_candidates(local,package)
            net=RoadNetwork(Path(a.root)/tid)
            best,stats=decode_trip(scored,net,Config())
            evaluation=evaluation_metrics(local,best,net)
            sequence=sequence_metrics(net,stats,Path(a.root)/tid)
            linked=stats.get('connected_arc_switches',0)
            broken=stats.get('disconnected_arc_switches',0)
            rows.append({'model':name,'trip_id':tid,'eligible_points':evaluation.get('eligible_points',0),
                'baseline_top1':(evaluation.get('baseline') or {}).get('directed'),
                'hmm_top1':(evaluation.get('hmm') or {}).get('directed'),
                'connected_arc_switches':linked,'disconnected_arc_switches':broken,
                'route_continuity':linked/(linked+broken) if (linked+broken)>0 else None,
                'sequence_f1':sequence.get('ordered_sequence_match_f1'),
                'elapsed_seconds':round(time.perf_counter()-start,3)})
            print(f'{name} {tid}: HMM={rows[-1]["hmm_top1"]} connectivity={rows[-1]["route_continuity"]}',flush=True)
        (out/f'{name}_student_trips.json').write_text(json.dumps(rows,indent=2)+'\n')
    joined=[];summaries=[]
    for name in NAMES:
        file=out/f'{name}_student_trips.json'
        if not file.exists():continue
        r=json.loads(file.read_text());joined.extend(r)
        if len(r)!=10:continue
        n=sum(x['eligible_points'] for x in r)
        cont=sum(x['connected_arc_switches'] for x in r)
        dis=sum(x['disconnected_arc_switches'] for x in r)
        summaries.append({'model':name,'trips':len(r),'eligible_points':n,
            'baseline_top1_weighted':sum(x['eligible_points']*x['baseline_top1'] for x in r if x['baseline_top1'] is not None)/n if n else None,
            'hmm_top1_weighted':sum(x['eligible_points']*x['hmm_top1'] for x in r if x['hmm_top1'] is not None)/n if n else None,
            'route_continuity':cont/(cont+dis) if cont+dis else None,
            'connected_arc_switches':cont,'disconnected_arc_switches':dis,
            'mean_sequence_f1':sum((x['sequence_f1'] or 0) for x in r)/len(r),
            'elapsed_seconds':sum(x['elapsed_seconds'] for x in r)})
    pd.DataFrame(joined).to_csv(out/'model_continuity_all_trips.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'model_continuity_summary.csv',index=False)
    print('SAVED',out/'model_continuity_summary.csv')

if __name__=='__main__':main()
