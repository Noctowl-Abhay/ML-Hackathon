#!/usr/bin/env python3
"""Copy publication-safe aggregate experiment evidence only; never copy raw labels."""
from __future__ import annotations
from pathlib import Path
import json,shutil

ROOT=Path('submission/evidence')
COPIES={
 'artifacts/guideline_reports/validation/comparison.csv':'validation/comparison.csv',
 'artifacts/guideline_reports/bayesian_comparison/cv_same_folds_comparison.csv':'bayesian/cv_same_folds_comparison.csv',
 'artifacts/guideline_reports/bayesian_comparison/student_same_partition_comparison.csv':'bayesian/student_same_partition_comparison.csv',
 'artifacts/guideline_reports/model_route_continuity/model_continuity_summary.csv':'student_test/model_continuity_summary.csv',
 'artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv':'cv/cv_all_models_summary.csv',
 'artifacts/guideline_reports/student_test/comparison.csv':'student_test/comparison.csv',
 'artifacts/guideline_reports/student_test/candidate_coverage.json':'student_test/candidate_coverage.json',
 'artifacts/logistic_regression_bayesian/optuna_trials.csv':'bayesian/optuna_trials.csv',
 'artifacts/logistic_regression_bayesian/best_params.json':'bayesian/best_params.json',
 'artifacts/logistic_regression_bayesian/cv_oof_metrics.json':'bayesian/cv_oof_metrics.json',
 'artifacts/guideline_reports/parameters/model_hyperparameters.json':'model_hyperparameters.json',
 'artifacts/guideline_reports/guideline_audit.md':'audit/guideline_audit.md',
}

def main():
    count=0
    for src,dst in COPIES.items():
        p=Path(src)
        if not p.exists():print('MISSING',src);continue
        target=ROOT/dst;target.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix.lower()=='.csv':
            # Git's whitespace checker treats CRLF as trailing whitespace.
            # Preserve data while normalising CSV line endings for the public repo.
            target.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
        else:
            shutil.copy2(p,target)
        count+=1
        print('COPIED',target)
    for folder,targetdir in [('all_models_cv','cv'),('student_test','student_test'),('bayesian_comparison','bayesian')]:
        source=Path('artifacts/guideline_reports')/folder
        if source.exists():
            for png in source.glob('*.png'):
                target=ROOT/targetdir/png.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(png,target);count+=1
    # Reconstructed routes may contain student-test reference identifiers. Publish
    # only an aggregate table with numeric metrics, not predictions, .route, GeoJSON,
    # or route-reference coordinates.
    comp=Path('artifacts/reconstruction/student_test/comparison.json')
    if comp.exists():
        rows=json.loads(comp.read_text())
        if not isinstance(rows,list) or len(rows)!=10:
            raise ValueError('Expected exactly ten student-test reconstruction summaries')
        sanitized=[]
        for row in rows:
            ev=row.get('evaluation') or {};s=row.get('reconstruction') or {}
            baseline=ev.get('baseline') or {};hmm=ev.get('hmm') or {};sequence=ev.get('sequence') or {}
            sanitized.append({'trip_id':row.get('trip_id'),
              'eligible_points':ev.get('eligible_points'),
              'classifier_top1_proxy':baseline.get('directed'),
              'hmm_top1_proxy':hmm.get('directed'),
              'sequence_match_f1':sequence.get('ordered_sequence_match_f1'),
              'disconnected_arc_switches':s.get('disconnected_arc_switches'),
              'connected_arc_switches':s.get('connected_arc_switches'),
              'elapsed_seconds':row.get('elapsed_seconds')})
        target=ROOT/'student_test/reconstruction_summary_10.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(sanitized,indent=2)+'\n');count+=1
    (ROOT/'README.md').write_text(
      '# Verified experiment evidence\n\nThese are copied summaries from real training/student-test runs. '
      'Per-GPS labels, raw files, evaluator routes, and reference geometry are excluded. '
      'Inspect the reports and raw offline artifacts before submission.\n')
    print('Finished publishing',count,'evidence files under',ROOT)

if __name__=='__main__':main()
