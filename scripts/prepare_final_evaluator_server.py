#!/usr/bin/env python3
"""FINAL DEMONSTRATION ONLY: create SERVER-PRIVATE evaluator reference metrics.

The source .route is used ONLY to derive offline evaluation labels after the
model has been frozen. Inference in api.py independently calls process_trip()
with partition='student_test', so the classifier never receives .route.
NEVER run this script to tune or select a model, candidate radius, or HMM.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from features.build_candidates import process_trip


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final-evaluator',action='store_true',help='Explicit acknowledgement: model is frozen')
    p.add_argument('--force',action='store_true',help='Explicitly rebuild existing private reference features')
    p.add_argument('--root',type=Path,default=Path('Data/processed/evaluator_test'))
    p.add_argument('--output',type=Path,default=Path('outputs/features/features_evaluator_test.parquet'))
    args=p.parse_args()
    if not args.final_evaluator:p.error('This may be run ONLY for final evaluator demo, after the model is frozen. Pass --final-evaluator.')
    if not Path('deployment/selected_model_manifest.json').is_file():p.error('Select and freeze final model first')
    if args.output.exists() and not args.force:
        print('EXISTS: existing evaluator reference table was NOT overwritten:',args.output)
        return
    if args.root.name != 'evaluator_test' or args.output.name != 'features_evaluator_test.parquet':
        p.error('Refusing unexpected evaluator paths')
    cfg=json.loads(Path('deployment/feature_config.json').read_text())
    trips=sorted(d for d in args.root.iterdir() if d.is_dir() and d.name.isdigit())
    if len(trips)!=10:raise ValueError(f'Expected exactly 10 reserved evaluator trip folders, found {len(trips)}')
    tables=[]
    for folder in trips:
        tid=folder.name
        for name in [f'{tid}.route',f'{tid}.arcs',f'{tid}.nodes',f'{tid}_track_processed.csv']:
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
        # label alignment is ONLY for the server-side metric reference dataset.
        frame,_=process_trip(folder,'train',k=int(cfg['k']),radius=float(cfg['radius_m']),
                             label_cutoff=float(cfg['label_cutoff_m']))
        frame['partition']='evaluator_reference_final_server_only'
        tables.append(frame)
        print('FINAL SERVER REFERENCE:',tid,'candidate rows',len(frame),flush=True)
    all_df=pd.concat(tables,ignore_index=True)
    assert all_df.trip_id.nunique()==10
    assert set(cfg['feature_cols'])<=set(all_df.columns)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    all_df.to_parquet(args.output,index=False)
    print('SAVED SERVER-PRIVATE EVALUATION REFERENCE:',args.output)
    print('Do NOT publish this file; the server predicts without reference inputs, then scores privately.')

if __name__=='__main__':main()
