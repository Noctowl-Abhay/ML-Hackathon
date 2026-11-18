#!/usr/bin/env python3
"""Create student-test candidate features with ROUTE-ALIGNED PROXY LABELS for evaluation ONLY.

Never fit or tune a model on these labels. For live inference the model uses
only the configured 25 features and never accesses .route.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from features.build_candidates import process_trip


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('Data/processed/student_test'))
    p.add_argument('--config',type=Path,default=Path('outputs/features/feature_config.json'))
    p.add_argument('--output',type=Path,default=Path('outputs/features/features_student_test_labeled.parquet'))
    a=p.parse_args()
    cfg=json.loads(a.config.read_text())
    trips=sorted(p for p in a.root.iterdir() if p.is_dir())
    if len(trips)!=10:raise ValueError(f'Expected 10 student-test trips, found {len(trips)}')
    tables=[];stats=[]
    for trip in trips:
        if not (trip/f'{trip.name}.route').exists():
            raise FileNotFoundError(f'{trip}: needs reference route for student-test metrics')
        frame,stat=process_trip(trip,'train',k=int(cfg.get('k',8)),
                                radius=float(cfg.get('radius_m',50)),
                                label_cutoff=float(cfg.get('label_cutoff_m',30)))
        frame['partition']='student_test'
        tables.append(frame);stats.append(stat)
        print(trip.name,'candidate rows',len(frame),'usable label points',stat['usable_label_points'],flush=True)
    all_df=pd.concat(tables,ignore_index=True)
    expected=cfg['feature_cols']
    if not set(expected).issubset(all_df.columns):raise ValueError('Feature mismatch')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.resolve()==Path('outputs/features/features_train.parquet').resolve():
        raise ValueError('Refusing to overwrite train')
    all_df.to_parquet(a.output,index=False)
    pd.DataFrame(stats).to_csv(a.output.with_suffix('.stats.csv'),index=False)
    print('Saved labelled STUDENT-TEST candidates for EVALUATION ONLY:',a.output)

if __name__=='__main__':main()
