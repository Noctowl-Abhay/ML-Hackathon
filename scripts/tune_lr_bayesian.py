#!/usr/bin/env python3
"""True Bayesian/TPE tuning of Logistic Regression on 80 TRAIN trips only.

Trip-grouped CV objective is observation-level conditional Top-1.
The 10 student-test trips are not touched until final frozen model evaluation.
The 10 evaluator trips are never read here.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, log_loss, accuracy_score


def make_model(C,class_weight):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),
                     ('scaler',StandardScaler()),
                     ('classifier',LogisticRegression(C=C, class_weight=class_weight,
                                                  solver='lbfgs',max_iter=1500,random_state=42))])


def top1_score(y,prob,trip_ids,point_ids):
    v=pd.DataFrame({'trip_id':trip_ids,'point_idx':point_ids,'y':y,'p':prob})
    good=v.groupby(['trip_id','point_idx'],sort=False,observed=True)['y'].transform('sum').eq(1)
    v=v.loc[good]
    if v.empty:return float('nan')
    selected=v.loc[v.groupby(['trip_id','point_idx'],sort=False,observed=True)['p'].idxmax()]
    return float(selected.y.mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='outputs/features/features_train.parquet')
    p.add_argument('--config',default='outputs/features/feature_config.json')
    p.add_argument('--trials',type=int,default=24)
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--out',default='artifacts/logistic_regression_bayesian')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(args.config).read_text())
    cols=cfg['feature_cols']
    illegal={'label','label_usable','label_arc_id','label_route_pos','label_dist_m','point_has_positive','trip_id','arc_id','point_idx','partition'}
    if illegal.intersection(cols):raise ValueError('Label leakage found in feature set')
    if args.trials<1:raise ValueError('--trials must be positive')
    frame=pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data,dtype={'trip_id':str})
    if frame.trip_id.nunique()!=80:raise ValueError('Expected EXACTLY 80 training trips')
    frame=frame.loc[frame.label_usable.eq(True)].reset_index(drop=True)
    if len(frame)==0:raise ValueError('No labelled training data')
    frame['trip_id']=frame.trip_id.astype(str).str.zfill(8)
    X=(frame[cols].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan).to_numpy(np.float32))
    y=frame.label.to_numpy(dtype=int)
    groups=frame.trip_id.to_numpy()
    point=frame.point_idx.to_numpy()
    folds=list(GroupKFold(n_splits=args.folds).split(X,y,groups))
    print(f'Bayesian TPE optimisation: {args.trials} trials x {args.folds} grouped folds; '
          f'{len(frame):,} candidate rows; 80 TRAIN trips only',flush=True)
    base_params={'C':1.0,'class_weight':'balanced'}
    def score(C,weight):
        scores=[];losses=[];seconds=0
        for train_ids,valid_ids in folds:
            est=make_model(C,weight)
            start=time.perf_counter()
            est.fit(X[train_ids],y[train_ids]);seconds+=time.perf_counter()-start
            prob=est.predict_proba(X[valid_ids])[:,1]
            scores.append(top1_score(y[valid_ids],prob,groups[valid_ids],point[valid_ids]))
            losses.append(log_loss(y[valid_ids],np.clip(prob,1e-7,1-1e-7)))
        return float(np.mean(scores)),float(np.mean(losses)),seconds
    def objective(trial):
        C=trial.suggest_float('C',1e-3,1e2,log=True)
        weight=trial.suggest_categorical('class_weight',['none','balanced'])
        score_v,loss,sec=score(C,None if weight=='none' else 'balanced')
        trial.set_user_attr('cv_log_loss',loss)
        trial.set_user_attr('fit_seconds',sec)
        print(f'[TPE {trial.number+1:02d}] C={C:.5g}, weight={weight}, '
              f'mean Top1={score_v:.5%}, logloss={loss:.5f}, fit={sec:.1f}s',flush=True)
        return score_v
    storage='sqlite:///'+str((out/'study.db').resolve())
    study=optuna.create_study(study_name='pathsnap_lr_train80_top1',storage=storage,
        load_if_exists=True,direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42,n_startup_trials=8))
    study.optimize(objective,n_trials=args.trials,show_progress_bar=False)
    best=study.best_params
    params={'C':float(best['C']),'class_weight':None if best['class_weight']=='none' else best['class_weight'],
            'objective':f'mean conditional Top-1 across {args.folds} grouped training-only folds',
            'cv_mean_top1':float(study.best_value),'trials_total':len(study.trials),
            'seed':42,'n_folds':args.folds,'feature_cols':cols,'search':{'C':'log-uniform [0.001,100]',
                  'class_weight':['none','balanced']},'training_partition':'train (80 trips)'}
    (out/'best_params.json').write_text(json.dumps(params,indent=2))
    study.trials_dataframe().to_csv(out/'optuna_trials.csv',index=False)
    print('Cross-validating BEST config for confusion matrix and macro-F1...',flush=True)
    oof=np.full(len(y),np.nan)
    start=time.perf_counter()
    for train_ids,valid_ids in folds:
        fitted=make_model(params['C'],params['class_weight'])
        fitted.fit(X[train_ids],y[train_ids])
        oof[valid_ids]=fitted.predict_proba(X[valid_ids])[:,1]
    assert np.isfinite(oof).all()
    pred=(oof>=.5).astype(int)
    report={'cv_top1_oof':top1_score(y,oof,groups,point),
            'cv_accuracy_oof':float(accuracy_score(y,pred)),
            'cv_macro_f1_oof':float(f1_score(y,pred,average='macro',labels=[0,1],zero_division=0)),
            'cv_binary_f1_oof':float(f1_score(y,pred,zero_division=0)),
            'cv_log_loss_oof':float(log_loss(y,np.clip(oof,1e-7,1-1e-7))),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
            'oof_inference_and_fit_seconds':float(time.perf_counter()-start),
            'note':'Grouped OOF performance on training trips; model chosen using these same grouped folds.'}
    (out/'cv_oof_metrics.json').write_text(json.dumps(report,indent=2))
    print('Fitting final model on all 80 TRAIN trips...',flush=True)
    final=make_model(params['C'],params['class_weight'])
    fitstart=time.perf_counter();final.fit(X,y)
    joblib.dump({'model':final,'feature_cols':cols,'model_name':'logistic_regression_bayesian',
                 'training_trips':80},out/'model_all_80.joblib')
    report['final_fit_seconds']=time.perf_counter()-fitstart
    (out/'cv_oof_metrics.json').write_text(json.dumps(report,indent=2))
    print('BEST',params,'\nOOF metrics:',report,flush=True)
    print('Output:',out,'\nIMPORTANT: Bayesian search never reads student-test/evaluator-test.',flush=True)

if __name__=='__main__':main()
