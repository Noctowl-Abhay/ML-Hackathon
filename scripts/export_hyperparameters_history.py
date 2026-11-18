#!/usr/bin/env python3
"""Export genuine saved model hyperparameters, and MLP epoch histories if available."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE={
    'logistic_regression':'artifacts/logistic_regression/model.joblib',
    'distance_weighted_knn':'artifacts/distance_weighted_knn/model.joblib',
    'random_forest':'artifacts/random_forest/model.joblib',
    'xgboost':'artifacts/xgboost/model.joblib',
    'deep_mlp':'artifacts/deep_mlp/model.joblib',
    'untuned_full80':'artifacts/logistic_regression_final/model.joblib',
    'tuned_full80':'artifacts/logistic_regression_tuned/model_all_80.joblib',
    'bayesian_full80':'artifacts/logistic_regression_bayesian/model_all_80.joblib',
}
out=Path('artifacts/guideline_reports/parameters');out.mkdir(parents=True,exist_ok=True)
results={}
for name,path in SOURCE.items():
    p=Path(path)
    if not p.exists() and name in ('distance_weighted_knn','deep_mlp'):
        p=Path('artifacts')/('knn' if name=='distance_weighted_knn' else 'mlp')/'model.joblib'
    if not p.exists():
        print('SKIP missing',p);continue
    package=joblib.load(p)
    pipe=package.get('pipeline',package.get('model'))
    if not hasattr(pipe,'named_steps'):
        print('SKIP unknown pipeline',p);continue
    classifier=pipe.named_steps['classifier']
    info={'model_name':name,'artifact':str(p),'algorithm_class':type(classifier).__name__,
          'features':list(package.get('features',package.get('feature_cols',[]))),
          'hyperparameters':classifier.get_params(deep=False)}
    results[name]=info
    print(name,type(classifier).__name__,classifier.get_params(deep=False))
    if name=='deep_mlp':
        train=getattr(classifier,'loss_curve_',None)
        valid=getattr(classifier,'validation_scores_',None)
        if train is not None and len(train)>0:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot(range(1,len(train)+1),train,label='Training loss')
            ax.set_xlabel('Epoch');ax.set_ylabel('Training loss')
            ax.set_title('Deep MLP - Actual training loss history')
            ax.legend();fig.tight_layout()
            fig.savefig(out/'deep_mlp_training_loss.png',dpi=180);plt.close(fig)
        if valid is not None and len(valid)>0:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.plot(range(1,len(valid)+1),valid,label='Validation score')
            ax.set_xlabel('Epoch');ax.set_ylabel('Validation score (NOT validation loss)')
            ax.set_title('Deep MLP - Actual early-stopping validation score')
            ax.legend();fig.tight_layout()
            fig.savefig(out/'deep_mlp_validation_score.png',dpi=180);plt.close(fig)
(out/'model_hyperparameters.json').write_text(json.dumps(results,indent=2,default=str))
print('Saved',out/'model_hyperparameters.json')
