#!/usr/bin/env python3
"""Fair train-only grouped-CV comparison: untuned vs Bayesian LR on identical folds."""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,log_loss

BASE={'untuned_full80':'artifacts/logistic_regression_final/model.joblib',
      'bayesian_full80':'artifacts/logistic_regression_bayesian/model_all_80.joblib'}

def normalize_model(path):
    pkg=joblib.load(path)
    if 'model' in pkg and 'feature_cols' in pkg: return pkg['model'],pkg['feature_cols']
    if 'pipeline' in pkg and 'features' in pkg: return pkg['pipeline'],pkg['features']
    raise ValueError(f'Unknown model package format: {path}')

def top1(y,prob,group,point):
    frame=pd.DataFrame({'trip':group,'point':point,'label':y,'prob':prob})
    ok=frame.groupby(['trip','point'],sort=False)['label'].transform('sum').eq(1)
    frame=frame.loc[ok]
    if frame.empty:return float('nan')
    winners=frame.loc[frame.groupby(['trip','point'],sort=False)['prob'].idxmax()]
    return float(winners.label.mean())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',default='outputs/features/features_train.parquet')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--out',default='artifacts/guideline_reports/bayesian_comparison')
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    df=(pd.read_parquet(a.features) if str(a.features).endswith('.parquet')
        else pd.read_csv(a.features,dtype={'trip_id':str}))
    df['trip_id']=df.trip_id.astype(str).str.zfill(8)
    if df.trip_id.nunique()!=80:raise ValueError('Train-only CV requires exactly 80 trips')
    df=df.loc[df.label_usable.eq(True)].reset_index(drop=True)
    y=df.label.to_numpy(int);groups=df.trip_id.to_numpy();points=df.point_idx.to_numpy()
    cv=list(GroupKFold(n_splits=a.folds).split(df,y,groups))
    results=[]
    for name,file in BASE.items():
        model,cols=normalize_model(file)
        if len(cols)!=25 or any(x.startswith('label') for x in cols):raise ValueError('Unsafe features')
        X=df[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
        for k,(tr,va) in enumerate(cv,1):
            est=clone(model);start=time.perf_counter();est.fit(X[tr],y[tr]);fit=time.perf_counter()-start
            start=time.perf_counter();prob=est.predict_proba(X[va])[:,1];infer=time.perf_counter()-start
            m={'model':name,'fold':k,'train_trips':len(set(groups[tr])),
               'validation_trips':len(set(groups[va])),
               'cv_macro_f1':float(f1_score(y[va],prob>=.5,average='macro',labels=[0,1],zero_division=0)),
               'cv_top1':top1(y[va],prob,groups[va],points[va]),
               'cv_log_loss':float(log_loss(y[va],np.clip(prob,1e-7,1-1e-7),labels=[0,1])),
               'fit_seconds':fit,'inference_seconds':infer}
            results.append(m);print(m,flush=True)
        pd.DataFrame([r for r in results if r['model']==name]).to_csv(out/f'{name}_folds.csv',index=False)
    pd.DataFrame(results).to_csv(out/'cv_same_folds_detail.csv',index=False)
    summ=pd.DataFrame(results).groupby('model').agg({'cv_macro_f1':['mean','std'],
          'cv_top1':['mean','std'],'cv_log_loss':['mean','std'],
          'fit_seconds':'sum','inference_seconds':'sum'})
    summ.columns=['_'.join(x) for x in summ.columns]
    summ.reset_index().to_csv(out/'cv_same_folds_comparison.csv',index=False)
    st=Path('artifacts/guideline_reports/student_test/comparison.csv')
    if not st.exists():raise FileNotFoundError('Student-test comparison missing: '+str(st))
    student=pd.read_csv(st)
    student=student.loc[student.model.isin(BASE) & student.partition.eq('student_test_10')]
    if set(student.model)!=set(BASE):raise ValueError('Missing same-partition student scores for both LR variants')
    student.to_csv(out/'student_same_partition_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4))
    grouped=pd.DataFrame(results).groupby('model')['cv_macro_f1']
    vals=grouped.mean();errs=grouped.std(ddof=0)
    ax.bar(vals.index,vals.values,yerr=errs.values,capsize=5)
    ax.set_ylabel('Grouped CV Macro F1');ax.set_title('Untuned vs Bayesian LR - same 5 grouped folds')
    fig.tight_layout();fig.savefig(out/'cv_same_folds_macro_f1.png',dpi=170);plt.close(fig)
    print('SAVED:',out)

if __name__=='__main__':main()
