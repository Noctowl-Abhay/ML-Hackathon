#!/usr/bin/env python3
"""Report-required GROUPED CV Macro-F1 on all five previously fitted model families.
Uses each model's SAVED pipeline as a template, refits one new clone per fold.
This is computationally expensive (especially KNN/MLP). Never uses evaluator.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import f1_score,log_loss
from sklearn.model_selection import GroupKFold
MODEL_DIRS=['logistic_regression','distance_weighted_knn','random_forest','xgboost','deep_mlp']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='outputs/features/features_train.parquet')
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--only',choices=MODEL_DIRS,default=None)
    p.add_argument('--out',default='artifacts/guideline_reports/all_models_cv')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.folds<2:raise ValueError('Need at least two CV folds')
    df=pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data,dtype={'trip_id':str})
    if df.trip_id.nunique()!=80:raise ValueError('Expected 80 training trips')
    df=df.loc[df.label_usable.eq(True)].copy().reset_index(drop=True)
    df['trip_id']=df.trip_id.astype(str).str.zfill(8)
    y=df.label.to_numpy(int);groups=df.trip_id.to_numpy()
    split=list(GroupKFold(n_splits=args.folds).split(df,y,groups))
    for name in (MODEL_DIRS if args.only is None else [args.only]):
        saved=Path('artifacts')/name/'model.joblib'
        if not saved.exists() and name in ('deep_mlp','distance_weighted_knn'):
            saved=Path('artifacts')/('mlp' if name=='deep_mlp' else 'knn')/'model.joblib'
        if not saved.exists():
            print('SKIP: model not found:',saved,flush=True);continue
        pkg=joblib.load(saved)
        if 'pipeline' not in pkg or 'features' not in pkg:
            raise ValueError(f'Expected saved original pipeline in {saved}')
        features=pkg['features'];model=pkg['pipeline']
        X=(df[features].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan))
        records=[]
        for i,(tr,va) in enumerate(split,1):
            start=time.perf_counter();est=clone(model)
            est.fit(X.iloc[tr],y[tr]);fit=time.perf_counter()-start
            start=time.perf_counter();p=est.predict_proba(X.iloc[va])[:,1];infer=time.perf_counter()-start
            pred=(p>=.5).astype(int)
            r={'model':name,'fold':i,'train_trips':len(set(groups[tr])),
               'validation_trips':len(set(groups[va])), 'macro_f1':float(f1_score(y[va],pred,average='macro',labels=[0,1],zero_division=0)),
               'log_loss':float(log_loss(y[va],np.clip(p,1e-7,1-1e-7))),
               'fit_seconds':float(fit),'inference_seconds':float(infer)}
            records.append(r);print(r,flush=True)
        pd.DataFrame(records).to_csv(out/f'{name}_cv.csv',index=False)
        print('SAVED',out/f'{name}_cv.csv',flush=True)
    results=[]
    for name in MODEL_DIRS:
        path=out/f'{name}_cv.csv'
        if path.exists():
            scores=pd.read_csv(path)
            results.append({'model':name,'n_folds':len(scores),
                'cv_macro_f1_mean':float(scores.macro_f1.mean()),
                'cv_macro_f1_std':float(scores.macro_f1.std(ddof=0)),
                'cv_log_loss_mean':float(scores.log_loss.mean()),
                'cv_fit_seconds_total':float(scores.fit_seconds.sum()),
                'cv_inference_seconds_total':float(scores.inference_seconds.sum())})
    if results:
        summary=pd.DataFrame(results)
        summary.to_csv(out/'cv_all_models_summary.csv',index=False)
        fig,ax=plt.subplots(figsize=(max(7,len(summary)*1.3),4.7))
        ax.bar(summary.model,summary.cv_macro_f1_mean,yerr=summary.cv_macro_f1_std,capsize=4)
        ax.set_ylabel("Macro F1 (mean ± std across grouped folds)")
        ax.set_title("PathSnap 80-trip GroupKFold Macro-F1")
        ax.tick_params(axis="x",labelrotation=30)
        fig.tight_layout();fig.savefig(out/"cv_macro_f1_comparison.png",dpi=180);plt.close(fig)
        print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':main()
