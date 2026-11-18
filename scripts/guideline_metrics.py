#!/usr/bin/env python3
"""PathSnap report: reproducible model comparison + macro-F1, confusion matrices.

--mode validation reads the FIVE models' saved holdout predictions.
--mode student reads student-test-labelled candidate table and scores the saved
    FIVE models plus baseline/tuned full-80 models if they exist.
Ground-truth proxy labels are used ONLY by evaluation. No model retraining.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, log_loss, roc_auc_score,
    confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=['logistic_regression','distance_weighted_knn','random_forest','xgboost','deep_mlp']
EXTRA=[('untuned_full80',Path('artifacts/logistic_regression_final/model.joblib')),
       ('tuned_full80',Path('artifacts/logistic_regression_tuned/model_all_80.joblib')),
       ('bayesian_full80',Path('artifacts/logistic_regression_bayesian/model_all_80.joblib'))]


def plot_cm(cm, model_name, folder):
    fig,ax=plt.subplots(figsize=(5.5,4.8))
    im=ax.imshow(cm)
    fig.colorbar(im,ax=ax)
    ax.set_xticks([0,1],labels=['Negative','Positive'])
    ax.set_yticks([0,1],labels=['Negative','Positive'])
    ax.set_xlabel('Predicted label');ax.set_ylabel('Proxy reference label')
    ax.set_title(f'{model_name} | Confusion matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',
                    color='white' if cm[i,j]>cm.max()*.5 else 'black',fontsize=11)
    fig.tight_layout()
    fig.savefig(folder/f'{model_name}_confusion_matrix.png',dpi=180)
    plt.close(fig)


def plots(table,folder,subtitle):
    for metric, title in [('macro_f1','Macro F1'),('top1_accuracy','Conditional Top-1'),('log_loss','Log Loss')]:
        t=table.dropna(subset=[metric]);
        if t.empty: continue
        fig,ax=plt.subplots(figsize=(max(7,len(t)*1.1),4.6))
        ax.bar(t['model'],t[metric]);ax.set_title(f'{subtitle}: {title}')
        ax.set_ylabel(title);ax.tick_params(axis='x',labelrotation=25)
        fig.tight_layout();fig.savefig(folder/f'{metric}_comparison.png',dpi=180);plt.close(fig)


def calc_metrics(frame, y, p, seconds, *, all_points=None):
    if np.isnan(p).any() or not np.isfinite(p).all():
        raise ValueError('Nonfinite probabilities')
    pred=(p>=.5).astype(int)
    cm=confusion_matrix(y,pred,labels=[0,1])
    result={
        'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),
        'binary_f1':float(f1_score(y,pred,zero_division=0)),
        'macro_f1':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
        'log_loss':float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
        'roc_auc':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'confusion_matrix':cm.tolist(), 'inference_seconds':float(seconds),
        'candidate_rows':int(len(frame)),
        'trips':int(frame['trip_id'].nunique()),
    }
    tmp=frame[['trip_id','point_idx','arc_id']].copy()
    tmp['label']=y;tmp['probability']=p
    groups=['trip_id','point_idx']
    # The reference segment must be represented by exactly one candidate.
    pos=tmp.groupby(groups,observed=True)['label'].transform('sum')
    elig=tmp.loc[pos.eq(1)]
    if not elig.empty:
        idx=elig.groupby(groups,observed=True)['probability'].idxmax()
        best=elig.loc[idx]
        result['top1_accuracy']=float(best.label.mean())
        result['top1_eligible_points']=int(len(best))
    else:
        result['top1_accuracy']=None;result['top1_eligible_points']=0
    if all_points is not None:
        result['candidate_miss_rate']=all_points
    return result


def load_model(path):
    obj=joblib.load(path)
    if 'pipeline' in obj:
        return obj['pipeline'],obj['features']
    if 'model' in obj and 'feature_cols' in obj:
        return obj['model'],obj['feature_cols']
    raise ValueError(f'Unrecognised model format: {path}')


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    records=[]
    if args.mode=='validation':
        print('Reporting EXISTING 64/16 holdout (NOT cross-validation or student test).',flush=True)
        for model in MODELS:
            pth=Path('artifacts')/model/'validation_predictions.csv'
            if not pth.exists() and model=='distance_weighted_knn':
                pth=Path('artifacts/knn/validation_predictions.csv')
            if not pth.exists() and model=='deep_mlp':
                pth=Path('artifacts/mlp/validation_predictions.csv')
            if not pth.exists():
                print('MISSING:',pth,flush=True);continue
            d=pd.read_csv(pth,dtype={'trip_id':str})
            if not {'trip_id','point_idx','arc_id','label','probability'}.issubset(d.columns):
                print('SKIPPING (missing columns):',pth,flush=True);continue
            r=calc_metrics(d,d.label.to_numpy(dtype=int),d.probability.to_numpy(float),0.)
            metrics_path=pth.parent/'metrics.json'
            if metrics_path.exists():
                original=json.loads(metrics_path.read_text())
                r['inference_seconds']=float(original.get('inference_seconds',0.))
                r['fit_seconds']=float(original.get('fit_seconds',0.))
            r['model']=model;r['partition']='grouped_holdout_16';r['metric_source']=str(pth)
            records.append(r);plot_cm(np.asarray(r['confusion_matrix']),model,out)
    else:
        path=Path(args.features)
        if not path.exists():
            raise FileNotFoundError(f'{path} missing: generate labelled student-test candidates first')
        all_df=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path,dtype={'trip_id':str})
        needed={'trip_id','point_idx','arc_id','label','label_usable','point_has_positive'}
        if needed-set(all_df.columns):raise ValueError('Missing '+str(needed-set(all_df.columns)))
        if all_df.trip_id.nunique()!=10:raise ValueError('Expected exactly 10 student-test trips')
        point_coverage=all_df.groupby(['trip_id','point_idx'],observed=True)['point_has_positive'].max()
        miss_rate=float(1-point_coverage.mean())
        stats={'partition':'student_test','all_gps_points':int(len(point_coverage)),
               'candidate_miss_rate_before_label_filter':miss_rate,
               'candidate_recall_before_label_filter':float(point_coverage.mean()),
               'definition':'Reference-aligned road arc absent among generated candidates. BEFORE label_usable filtering.'}
        (out/'candidate_coverage.json').write_text(json.dumps(stats,indent=2))
        df=all_df.loc[all_df['label_usable'].eq(True)].copy().reset_index(drop=True)
        if df.empty:raise ValueError('No usable labels')
        y=df['label'].to_numpy(dtype=int)
        candidates=[(name,Path('artifacts')/name/'model.joblib') for name in MODELS]+EXTRA
        candidates=[(name,Path('artifacts/knn/model.joblib') if name=='distance_weighted_knn' and not path.exists() else path) for name,path in candidates]
        candidates=[(name,Path('artifacts/mlp/model.joblib') if name=='deep_mlp' and not path.exists() else path) for name,path in candidates]
        for name,path in candidates:
            if not path.exists():
                print('MISSING:',path,flush=True);continue
            print('Scoring',name,'rows',len(df),flush=True)
            pipeline,features=load_model(path)
            missing=set(features)-set(df.columns)
            if missing:raise ValueError(f'{name} missing features: {sorted(missing)}')
            start=time.perf_counter();parts=[]
            for i in range(0,len(df),args.batch_size):
                chunk=df.iloc[i:i+args.batch_size][features]
                chunk=chunk.apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
                parts.append(pipeline.predict_proba(chunk)[:,1])
            p=np.concatenate(parts)
            seconds=time.perf_counter()-start
            r=calc_metrics(df,y,p,seconds,all_points=miss_rate)
            r['model']=name;r['partition']='student_test_10';r['metric_source']=str(path)
            records.append(r);plot_cm(np.asarray(r['confusion_matrix']),name,out)
    if not records:raise SystemExit('No results produced; check model/prediction files')
    with (out/'full_metrics.json').open('w') as f:json.dump(records,f,indent=2)
    tab=pd.DataFrame(records)
    tab.drop(columns=['confusion_matrix'],errors='ignore').to_csv(out/'comparison.csv',index=False)
    cols=['model','accuracy','precision','recall','binary_f1','macro_f1','log_loss','top1_accuracy','inference_seconds']
    print(tab[[k for k in cols if k in tab]].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
    plots(tab,out,args.mode)
    print('Saved metrics, CSV, confusion plots, comparison plots to',out)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['validation','student'],required=True)
    ap.add_argument('--features',default='outputs/features/features_student_test_labeled.parquet')
    ap.add_argument('--output',default='artifacts/guideline_reports')
    ap.add_argument('--batch-size',type=int,default=30000)
    run(ap.parse_args())
