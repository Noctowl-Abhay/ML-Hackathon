#!/usr/bin/env python3
"""Select a DEPLOYMENT model based ONLY on recorded student-test metrics.

No evaluator data may be used to choose the model. Explicit --model allows
team-owned justification; --auto is an optional Top-1/Macro-F1 recommendation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil
import joblib
from datetime import datetime, timezone
from pathlib import Path

CHOICES = {
    'logistic_regression': 'artifacts/logistic_regression/model.joblib',
    'distance_weighted_knn': 'artifacts/distance_weighted_knn/model.joblib',
    'random_forest': 'artifacts/random_forest/model.joblib',
    'xgboost': 'artifacts/xgboost/model.joblib',
    'deep_mlp': 'artifacts/deep_mlp/model.joblib',
    'untuned_full80': 'artifacts/logistic_regression_final/model.joblib',
    'tuned_full80': 'artifacts/logistic_regression_tuned/model_all_80.joblib',
    'bayesian_full80': 'artifacts/logistic_regression_bayesian/model_all_80.joblib',
}
ALT = {'distance_weighted_knn':'artifacts/knn/model.joblib', 'deep_mlp':'artifacts/mlp/model.joblib'}


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['auto',*CHOICES])
    p.add_argument('--reason',default='Team selection based on student-test Top-1, Macro-F1, CV stability, and efficiency; review before publishing.')
    args=p.parse_args()
    results=Path('artifacts/guideline_reports/student_test/comparison.csv')
    if not results.is_file():p.error('Run 10-trip student-test model evaluation first; missing '+str(results))
    with results.open(newline='') as f:rows=list(csv.DictReader(f))
    valid={r['model']:r for r in rows if r.get('partition')=='student_test_10' and r.get('model') in CHOICES}
    if args.model=='auto':
        if len(set(CHOICES)-set(valid)):
            p.error('Auto selection requires metrics for all eight model variants, missing '+str(set(CHOICES)-set(valid)))
        lr_variants=['untuned_full80','tuned_full80','bayesian_full80']
        model=max(lr_variants,key=lambda name:(float(valid[name]['top1_accuracy']),
            float(valid[name]['macro_f1']),-float(valid[name]['log_loss'])))
        strongest=max(valid,key=lambda name:(float(valid[name]['top1_accuracy']),float(valid[name]['macro_f1'])))
        if strongest not in lr_variants:
            print('REVIEW NEEDED: another model family has higher student-test Top-1:',strongest)
            print('Auto recommends best fully trained LR variant because the Bayesian search was for LR.')
            print('Team must justify LR based on CV stability, inference latency and HMM compatibility.')
    else:model=args.model
    if model not in valid:p.error(f'No student-test evaluation for {model}')
    source=Path(CHOICES[model]);source=source if source.exists() else Path(ALT.get(model,str(source)))
    if not source.is_file():p.error('Missing trained model: '+str(source))
    cfg=Path('outputs/features/feature_config.json')
    if not cfg.is_file():p.error('Missing training feature config: '+str(cfg))
    config=json.loads(cfg.read_text());cols=config.get('feature_cols',[])
    if len(cols)!=25 or any(c.startswith('label') for c in cols):
        p.error('Expected 25 non-label features from training configuration')
    out=Path('deployment');out.mkdir(exist_ok=True)
    target=out/'selected_model.joblib'
    stored=joblib.load(source)
    if isinstance(stored,dict) and 'pipeline' in stored and 'features' in stored:
        # Decoder/API require the standard {model, feature_cols} package.
        stored={'model':stored['pipeline'],'feature_cols':stored['features'],
                'model_name':model,'source_artifact':str(source)}
        joblib.dump(stored,target)
    elif isinstance(stored,dict) and 'model' in stored and 'feature_cols' in stored:
        shutil.copy2(source,target)
    else:
        p.error('Unsupported saved model package structure: '+str(source))
    shutil.copy2(cfg,out/'feature_config.json')
    if target.stat().st_size>95*1024*1024:
        target.unlink();p.error('Model >95MiB; GitHub rejects individual files >=100MiB. Arrange Git LFS if allowed.')
    manifest={'selected_model':model,'source_artifact':str(source),
        'selected_from':'student_test_10 (NOT evaluator_test)',
        'primary_selection_metric':'conditional observation-level Top-1',
        'student_metrics':{k:valid[model].get(k) for k in ['top1_accuracy','macro_f1','log_loss','inference_seconds','candidate_rows','trips']},
        'team_selection_rationale':args.reason,'feature_count':len(cols),
        'model_sha256':sha256(target),'feature_config_sha256':sha256(out/'feature_config.json'),
        'selected_utc':datetime.now(timezone.utc).isoformat()}
    (out/'selected_model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('SELECTED:',model)
    print('Student-test Top-1:',manifest['student_metrics']['top1_accuracy'])
    print('Model:',target,'size:',target.stat().st_size,'bytes')
    print('Review manifest rationale and STUDENT-test ranking before GitHub submission.')

if __name__=='__main__':main()
