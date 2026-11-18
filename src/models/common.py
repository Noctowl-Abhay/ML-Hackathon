"""Shared, leakage-aware candidate-level training and evaluation."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES = ['dist_m','cand_rank','dist_minus_min_m','dist_ratio_min','n_cands_in_radius','frac_along','frac_to_end','arc_length_m','heading_diff_deg','heading_diff_undirected_deg','heading_valid']
FORBIDDEN = {'label','label_usable','label_arc_id','label_route_pos','label_dist_m','point_has_positive','trip_id','arc_id','candidate_arc_id','point_idx','gps_idx','point_id','partition','route_id','is_correct'}

def parser_for(name):
    p=argparse.ArgumentParser(description=f'Train {name} on PathSnap candidate-level features')
    p.add_argument('--data',required=True,help='Candidate-level features_train.parquet, .csv or .csv.gz')
    p.add_argument('--features',nargs='*',help='Explicit feature columns; otherwise use feature_config.json or safe defaults')
    p.add_argument('--feature-config',help='JSON containing feature_cols or FEATURE_COLS list')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-size',type=float,default=0.2)
    p.add_argument('--max-train-rows',type=int,default=0,help='Optional row cap for faster KNN/MLP experiments; 0=all')
    return p

def load_data(path):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(f'Feature table missing: {path}. GPS-only preprocessed CSVs are NOT sufficient. Generate candidate-level features and labels first.')
    if path.suffix=='.parquet': return pd.read_parquet(path)
    if path.name.endswith(('.csv','.csv.gz')): return pd.read_csv(path,low_memory=False,dtype={'trip_id':str})
    raise ValueError('Use .parquet, .csv, or .csv.gz')

def choose_features(df,args):
    if args.features:
        cols=args.features
    elif args.feature_config:
        cfg=json.loads(Path(args.feature_config).read_text(encoding='utf-8'))
        cols=cfg.get('feature_cols',cfg.get('FEATURE_COLS'))
        if not isinstance(cols,list): raise ValueError('Config must contain a feature_cols or FEATURE_COLS list')
    else:
        cols=[c for c in DEFAULT_FEATURES if c in df.columns]
    if not cols: raise ValueError('No usable feature columns found. Supply --features or --feature-config from the candidate generator.')
    banned=[c for c in cols if c in FORBIDDEN or c.startswith(('label_','target_'))]
    if banned: raise ValueError(f'Potential target leakage in selected features: {banned}')
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError(f'Missing selected feature columns: {missing}')
    non_numeric=[c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric: raise ValueError(f'Non-numeric feature columns: {non_numeric}')
    return cols

def prepare(df,args):
    required={'trip_id','label','label_usable'}
    missing=required-set(df.columns)
    if missing: raise ValueError(f'Missing columns {sorted(missing)}. This is probably a GPS-only preprocessing table, not candidate-level labelled features.')
    df=df.copy()
    df=df.loc[pd.to_numeric(df['label_usable'],errors='coerce').eq(1)].copy()
    df['label']=pd.to_numeric(df['label'],errors='coerce')
    df=df[df['label'].isin([0,1]) & df['trip_id'].notna()].copy()
    if df.empty: raise ValueError('No usable labelled candidate rows.')
    df['trip_id']=df['trip_id'].astype(str).str.zfill(8)
    if df.trip_id.nunique()<5: raise ValueError('Need at least 5 distinct labelled trips for a meaningful trip-level holdout.')
    cols=choose_features(df,args)
    for c in cols: df[c]=pd.to_numeric(df[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    if not 0<args.test_size<1: raise ValueError('--test-size must be between 0 and 1')
    splitter=GroupShuffleSplit(n_splits=1,test_size=args.test_size,random_state=args.seed)
    tr,va=next(splitter.split(df,df.label,groups=df.trip_id))
    train,val=df.iloc[tr].copy(),df.iloc[va].copy()
    if train.label.nunique()!=2 or val.label.nunique()!=2: raise ValueError('One split contains only one class. Try another --seed or inspect labels.')
    if args.max_train_rows and len(train)>args.max_train_rows:
        # Stratified sampling within the training trips only (validation stays untouched).
        from sklearn.model_selection import train_test_split
        chosen,_=train_test_split(np.arange(len(train)),train_size=args.max_train_rows,stratify=train.label,random_state=args.seed)
        train=train.iloc[chosen].copy()
    return train,val,cols

def make_pipeline(model,scale=True):
    steps=[('imputer',SimpleImputer(strategy='median',add_indicator=False))]
    if scale: steps.append(('scaler',StandardScaler()))
    steps.append(('classifier',model))
    return Pipeline(steps)

def run(name,model,scale=True):
    args=parser_for(name).parse_args()
    out=Path(args.output_dir or ROOT/'artifacts'/name)
    df=load_data(args.data)
    print(f'Loaded {len(df):,} candidate rows; columns: {list(df.columns)}')
    train,val,cols=prepare(df,args)
    print(f'Train: {len(train):,} rows / {train.trip_id.nunique()} trips; validation: {len(val):,} rows / {val.trip_id.nunique()} trips')
    pipe=make_pipeline(model,scale)
    start=time.perf_counter()
    pipe.fit(train[cols],train.label.astype(int))
    fit_sec=time.perf_counter()-start
    start=time.perf_counter()
    probs=pipe.predict_proba(val[cols])[:,1]
    infer_sec=time.perf_counter()-start
    pred=(probs>=0.5).astype(int)
    precision,recall,f1,_=precision_recall_fscore_support(val.label.astype(int),pred,average='binary',zero_division=0)
    metrics={'model':name,'train_rows':len(train),'validation_rows':len(val),'train_trips':train.trip_id.nunique(),'validation_trips':val.trip_id.nunique(),'positive_rate_validation':float(val.label.mean()),'accuracy':float(accuracy_score(val.label,pred)),'balanced_accuracy':float(balanced_accuracy_score(val.label,pred)),'precision':float(precision),'recall':float(recall),'f1':float(f1),'log_loss':float(log_loss(val.label,np.clip(probs,1e-7,1-1e-7))),'roc_auc':float(roc_auc_score(val.label,probs)),'fit_seconds':fit_sec,'inference_seconds':infer_sec,'seed':args.seed,'features':cols,'validation_trip_ids':sorted(val.trip_id.unique().tolist())}
    results=val[['trip_id','label']].copy()
    for col in ['point_idx','gps_idx','point_id','arc_id','candidate_arc_id']:
        if col in val: results[col]=val[col].values
    results['probability']=probs
    results['prediction']=pred
    # Top-1 is only valid when a GPS observation identifier is present.
    point_col=next((c for c in ['point_idx','gps_idx','point_id'] if c in results.columns),None)
    if point_col:
        groups=['trip_id',point_col]
        ranked=results.sort_values('probability',ascending=False).drop_duplicates(groups)
        valid_points=results.groupby(groups)['label'].max().reset_index(name='has_positive')
        ranked=ranked.merge(valid_points,on=groups)
        eligible=ranked[ranked.has_positive.eq(1)]
        metrics['top1_accuracy_when_positive_present']=float(eligible.label.mean()) if len(eligible) else None
        metrics['top1_eligible_points']=int(len(eligible))
        metrics['candidate_recall_points']=float(valid_points.has_positive.mean())
    else:
        print('WARNING: no point identifier found; skipping top-1 candidate ranking metric.')
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'pipeline':pipe,'features':cols,'model':name},out/'model.joblib')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,default=str),encoding='utf-8')
    results.to_csv(out/'validation_predictions.csv',index=False)
    if hasattr(pipe.named_steps['classifier'],'coef_'):
        pd.DataFrame({'feature':cols,'coefficient':pipe.named_steps['classifier'].coef_[0]}).to_csv(out/'coefficients.csv',index=False)
    if hasattr(pipe.named_steps['classifier'],'feature_importances_'):
        pd.DataFrame({'feature':cols,'importance':pipe.named_steps['classifier'].feature_importances_}).sort_values('importance',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['features','validation_trip_ids']},indent=2,default=str))
    print(f'Saved: {out}')
