#!/usr/bin/env python3
"""Evidence audit explicitly mapped to all five official PathSnap PDFs.

Existence of an artifact is NOT proof of correct scientific methodology or
working two-laptop networking. Reports both automated and manual gates.
"""
from __future__ import annotations
import argparse,csv,json,os,re,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path.cwd(); checks=[]
FIVE={'logistic_regression','distance_weighted_knn','random_forest','xgboost','deep_mlp'}

def add(group,title,passed,evidence,action,manual=False):
    checks.append({'group':group,'checkpoint':title,'status':('MANUAL' if manual else ('PASS' if passed else 'PENDING')),
      'evidence':str(evidence),'next_action':action if not passed else ''})

def exists(p):return (ROOT/p).is_file()
def json_file(p):
    try:return json.loads((ROOT/p).read_text())
    except (OSError,ValueError):return None

def csv_rows(p):
    try:
        with (ROOT/p).open(newline='') as f:return list(csv.DictReader(f))
    except (OSError,csv.Error):return []

def folders(name):
    p=ROOT/'Data/processed'/name
    return sorted(x.name for x in p.iterdir() if x.is_dir() and re.fullmatch(r'\d{8}',x.name)) if p.is_dir() else []

def finite(v):
    try:
        f=float(v);return bool(f==f and abs(f)!=float('inf'))
    except (ValueError,TypeError):return False

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['preflight','development','submission'],default='development')
    args=ap.parse_args()
    parts={p:folders(p) for p in ('train','student_test','evaluator_test')}
    for p,n in [('train',80),('student_test',10),('evaluator_test',10)]:
        add('Dataset',f'{p}: {n} distinct trips',len(parts[p])==n,
            f'Data/processed/{p}: {len(parts[p])} trips',f'Restore {n} processed {p} trips')
    collision=set(parts['train']) & set(parts['student_test']) | set(parts['train']) & set(parts['evaluator_test']) | set(parts['student_test']) & set(parts['evaluator_test'])
    add('Dataset','Trip IDs do not overlap across partitions',all(parts.values()) and not collision,f'collisions={sorted(collision)}','Fix overlapping trip partitions')
    data=ROOT/'outputs/features/features_train.parquet';cfg=json_file('outputs/features/feature_config.json')
    add('Features','Training candidate table exists and is nonempty',data.is_file() and data.stat().st_size>0,
        data,'Generate 80 training-trip features')
    cols=cfg.get('feature_cols',[]) if isinstance(cfg,dict) else []
    bad=[x for x in cols if re.search(r'(?:^label|ground_truth|reference|route_pos|point_has_positive)',x,re.I)]
    add('Features','Exactly 25 inference features without reference fields',len(cols)==25 and not bad,
        f'feature_count={len(cols)}, forbidden={bad}','Fix saved feature config to exclude any labels')
    add('Features','Preprocessing source and feature generator',exists('features/build_candidates.py') and
        (exists('src/preprocess_dataset.py') or exists('src/preprocessing.py') or exists('src/inspect_dataset.py')),
        'features/build_candidates.py + src preprocessing source','Include original processing code')
    for model in sorted(FIVE):
        alt={'distance_weighted_knn':'knn','deep_mlp':'mlp'}.get(model,model)
        found=exists(f'artifacts/{model}/model.joblib') or exists(f'artifacts/{alt}/model.joblib')
        add('Five models',f'{model}: saved fitted model',found,f'artifacts/{model}/model.joblib','Complete training and save model')
    val=csv_rows('artifacts/guideline_reports/validation/comparison.csv')
    add('Five models','Five-model 64/16 internal holdout',FIVE <= {r.get('model') for r in val},
        f'{len(val)} rows; validation/comparison.csv','Run guideline_metrics.py --mode validation')
    cv=csv_rows('artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv')
    cv_by={r.get('model'):r for r in cv}
    validcv=all(name in cv_by and int(float(cv_by[name].get('n_folds',0)))>=3 and finite(cv_by[name].get('cv_macro_f1_mean')) for name in FIVE)
    add('Five models','Grouped CV Macro-F1 for every model (>=3 folds)',validcv,
        'artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv',
        'Run cv_all_models.py --folds 3; verify trip-level groups')
    student=csv_rows('artifacts/guideline_reports/student_test/comparison.csv')
    students={r.get('model'):r for r in student if r.get('partition')=='student_test_10'}
    def has_student(n):
        v=students.get(n)
        return bool(v and int(float(v.get('trips',0)))==10 and all(finite(v.get(k)) for k in ['top1_accuracy','macro_f1','log_loss','inference_seconds']))
    add('Student tests','All FIVE models evaluated on ten student trips',all(has_student(n) for n in FIVE),
        f'{sum(has_student(n) for n in FIVE)} of 5 complete','Build labelled student features and run student evaluation')
    add('Student tests','Untuned/full80 LR and grid-tuned/full80 LR compared on same split',
        has_student('untuned_full80') and has_student('tuned_full80'),
        'student_test/comparison.csv','Score both frozen full80 LR artifacts on student test')
    add('Student tests','Bayesian/full80 LR evaluated on same student split',has_student('bayesian_full80'),
        'student_test/comparison.csv','Complete Optuna and rerun student comparison')
    coverage=json_file('artifacts/guideline_reports/student_test/candidate_coverage.json')
    cov=coverage.get('candidate_miss_rate_before_label_filter') if isinstance(coverage,dict) else None
    add('Student tests','Candidate miss rate measured before label filter',finite(cov) and 0<=float(cov)<=1 if finite(cov) else False,
        f'candidate_miss_rate={cov}','Generate original unfiltered student candidate coverage')
    cm=all(exists(f'artifacts/guideline_reports/student_test/{n}_confusion_matrix.png') for n in FIVE)
    add('Plots','Actual student-test confusion matrices for all five models',cm,
        'student_test/*_confusion_matrix.png','Run student-test guideline_metrics.py')
    add('Plots','CV Macro-F1, student Top-1/Macro-F1, and log loss plots',
        all(exists(p) for p in ['artifacts/guideline_reports/all_models_cv/cv_macro_f1_comparison.png',
          'artifacts/guideline_reports/student_test/top1_accuracy_comparison.png',
          'artifacts/guideline_reports/student_test/macro_f1_comparison.png',
          'artifacts/guideline_reports/student_test/log_loss_comparison.png']),
        'all_models_cv/*.png + student_test/*.png','Generate real measured comparison figures')
    trials=csv_rows('artifacts/logistic_regression_bayesian/optuna_trials.csv')
    finished=[t for t in trials if t.get('state')=='COMPLETE' and finite(t.get('value'))]
    bayes=json_file('artifacts/logistic_regression_bayesian/best_params.json')
    add('Bayesian','TPE search log with at least 24 COMPLETED trials',len(finished)>=24,
        f'completed={len(finished)} of {len(trials)} trials','Run train-only 24-trial Optuna Bayesian search')
    add('Bayesian','Recorded search parameters, space, folds, objective',bool(bayes and bayes.get('n_folds',0)>=3 and bayes.get('search') and bayes.get('objective') and bayes.get('training_partition','').startswith('train')),
        'artifacts/logistic_regression_bayesian/best_params.json','Complete Bayesian tuning and saved params')
    add('Bayesian','Refitted Bayesian model and grouped OOF metrics',
        exists('artifacts/logistic_regression_bayesian/model_all_80.joblib') and isinstance(json_file('artifacts/logistic_regression_bayesian/cv_oof_metrics.json'),dict),
        'artifacts/logistic_regression_bayesian/model_all_80.joblib + cv_oof_metrics.json',
        'Train final Bayesian model on 80 permitted training trips')
    fair=csv_rows('artifacts/guideline_reports/bayesian_comparison/cv_same_folds_comparison.csv')
    fairnames={r.get('model') for r in fair}
    add('Bayesian','Untuned-vs-Bayesian on SAME five training CV folds',
        {'untuned_full80','bayesian_full80'}<=fairnames and all(finite(r.get('cv_macro_f1_mean')) for r in fair),
        'artifacts/guideline_reports/bayesian_comparison/cv_same_folds_comparison.csv',
        'Run compare_untuned_bayesian_cv.py --folds 5')
    same_student=csv_rows('artifacts/guideline_reports/bayesian_comparison/student_same_partition_comparison.csv')
    add('Bayesian','Untuned-vs-Bayesian on SAME 10 student-testing trips',
        {r.get('model') for r in same_student}=={'untuned_full80','bayesian_full80'},
        'artifacts/guideline_reports/bayesian_comparison/student_same_partition_comparison.csv',
        'Run fair Bayesian comparison script after scoring student test')
    family_cont=csv_rows('artifacts/guideline_reports/model_route_continuity/model_continuity_summary.csv')
    add('Five models','Student route continuity recorded for all five model families',
        FIVE<={r.get('model') for r in family_cont} and all(finite(r.get('route_continuity')) for r in family_cont),
        'artifacts/guideline_reports/model_route_continuity/model_continuity_summary.csv',
        'Run compare_student_route_continuity.py on permitted student trips')
    add('Bayesian','Hyperparameter export',exists('artifacts/guideline_reports/parameters/model_hyperparameters.json'),
        'artifacts/guideline_reports/parameters/model_hyperparameters.json','Run export_hyperparameters_history.py')
    summ=json_file('artifacts/reconstruction/student_test/comparison.json')
    valid_summ=isinstance(summ,list) and len(summ)==10 and all(
        isinstance(x.get('evaluation',{}).get('baseline'),dict) and isinstance(x.get('evaluation',{}).get('hmm'),dict)
        and finite(x['evaluation']['hmm'].get('directed')) for x in summ)
    add('HMM','Classifier vs HMM recorded across all ten student trips',valid_summ,
        'artifacts/reconstruction/student_test/comparison.json','Run student route reconstruction using explicitly selected model')
    checksum=ROOT/'artifacts/reconstruction/student_test/selected_model.sha256'
    import hashlib
    modelbytes=ROOT/'deployment/selected_model.joblib'
    matches=False
    if checksum.is_file() and modelbytes.is_file():
        h=hashlib.sha256()
        with modelbytes.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):
                h.update(b)
        matches=(checksum.read_text().strip()==h.hexdigest())
    add('HMM','Student HMM results use deployed model weights',matches,
        'artifacts/reconstruction/student_test/selected_model.sha256',
        'Re-run student HMM after changing the final selected model')
    add('HMM','Decoder code present',exists('src/reconstruction/decoder.py'),
        'src/reconstruction/decoder.py','Install trained route decoder and validate transitions')
    add('HMM','Route continuity/sequence figures recorded',
        bool(valid_summ and all(('disconnected_arc_switches' in (x.get('reconstruction') or {})) and
            ('ordered_sequence_match_f1' in (x.get('evaluation',{}).get('sequence') or {})) for x in summ)),
        'student_test/comparison.json sequence and connected arc metrics',
        'Record disconnected switches and sequence metric per student trip')
    add('Folium','Original GPS + road network + prediction generator and actual map',
        exists('scripts/render_folium_guidelines.py') and bool(list((ROOT/'artifacts/reconstruction/student_test').glob('*/road_network_map.html'))),
        'scripts/render_folium_guidelines.py + generated student map','Run render_folium_guidelines.py on permitted student trip')
    add('API + UI','FastAPI adapter and UI files present',
        exists('src/reconstruction/api_compatible.py') and exists('src/reconstruction/api.py') and
        all(exists('ui/'+x) for x in ['index.html','app.js','config.js','style.css']),
        'src/reconstruction/*.py + ui/', 'Install API and pull teammate UI')
    add('API + UI','UI contract offline tests present',exists('tests/test_ui_contract.py'),
        'tests/test_ui_contract.py','Run python -m pytest -q tests/test_ui_contract.py')
    add('API + UI','Reserved evaluator LIVE 10-trip smoke evidence',
        isinstance(json_file('artifacts/api_checks/contract_summary.json'),list) and
        len(json_file('artifacts/api_checks/contract_summary.json'))==10 if isinstance(json_file('artifacts/api_checks/contract_summary.json'),list) else False,
        'artifacts/api_checks/contract_summary.json',
        'ONLY AFTER FREEZE: run api_smoke_check.py --all-trips --final-evaluator')
    add('API + UI','Two physical laptops, Windows UI, Ubuntu API screenshot + access log',False,
        'submission/screenshots/ui_windows.png + api_ubuntu_logs.png',
        'Capture real two-laptop screenshots; check same-network connection',manual=True)
    manifest=json_file('deployment/selected_model_manifest.json')
    model_file=ROOT/'deployment/selected_model.joblib'
    add('Deployment','Selected model weights and inference config bundled',
        bool(isinstance(manifest,dict) and manifest.get('selected_from','').startswith('student_test') and
             model_file.is_file() and 0<model_file.stat().st_size<95*1024*1024 and exists('deployment/feature_config.json')),
        'deployment/selected_model.joblib + feature_config.json + manifest.json',
        'Run select_final_model.py --model <student-validated-variant>')
    add('Submission','Complete source README and requirements',exists('README.md') and
        exists('README_SUBMISSION.md') and exists('requirements-pathsnap-final.txt'),
        'README.md, README_SUBMISSION.md, requirements-pathsnap-final.txt','Add exact setup and deployment commands')
    report=ROOT/'submission/PathSnap_Final_Technical_Report.pdf'
    add('Report','ONE consolidated non-draft final PDF',report.is_file() and report.stat().st_size>15_000 if report.exists() else False,
        'submission/PathSnap_Final_Technical_Report.pdf',
        'Team must finish individual model analysis, insert REAL graphs/maps/screenshots, export PDF')
    add('Report','All mandatory report sections and figures verified by the team',False,
        'Human review of PDF sections 2-7 / five model subsections / Folium / screenshots',
        'Review actual PDF against Deliverables.pdf; cannot be certified from filenames',manual=True)
    add('Integrity','No evaluator labels in model inputs or team-to-team code sharing',False,
        'Human review + repo history and implementation ownership',
        'Verify independence, explainable code, no hardcoded predictions or data leakage',manual=True)
    add('GitHub','Public GitHub repo reviewed, screenshots, credentials and weights checked',False,
        'github.com/Noctowl-Abhay/ML-Hackathon',
        'Stage whitelisted files with safe_stage_git.sh; inspect before committing',manual=True)
    counts={s:sum(c['status']==s for c in checks) for s in ['PASS','PENDING','MANUAL']}
    output=ROOT/'artifacts/guideline_reports';output.mkdir(parents=True,exist_ok=True)
    report_data={'generated_utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,
                 'counts':counts,'checks':checks}
    (output/'final_submission_audit.json').write_text(json.dumps(report_data,indent=2)+'\n')
    lines=['# PathSnap - Evidence-based final guideline audit','',
      f"**Automated: {counts['PASS']} PASS, {counts['PENDING']} PENDING. Manual reviews: {counts['MANUAL']}.**",'',
      'A PASS only means that the requested evidence passed this script\'s structural checks; it does not certify scientific accuracy.',
      'Actual evaluation must use 80 training trips / 10 student-testing trips; evaluator only for FINAL frozen inference.','',
      '| State | Category | Checkpoint | Evidence / required action |','|---|---|---|---|']
    for c in checks:lines.append('| '+ ' | '.join([c['status'],c['group'],c['checkpoint'],
        (c['evidence'] if c['status']=='PASS' else c['next_action']).replace('|','/')])+' |')
    lines+=['','Source PDFs: Main Guidelines, Dataset Exploration, Models and Training Guidelines, Deliverables, Assessment and Plagiarism Policy.','']
    (output/'final_submission_audit.md').write_text('\n'.join(lines))
    print('\n'.join(lines))
    if args.phase=='submission' and counts['PENDING']>0:
        print('NOT READY: complete pending tests. MANUAL checkpoints require separate human verification.',file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
