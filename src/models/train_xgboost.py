from xgboost import XGBClassifier
try:
    from .common import run
except ImportError:
    from common import run

if __name__ == '__main__':
    run('xgboost',XGBClassifier(n_estimators=350,max_depth=5,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,tree_method='hist',eval_metric='logloss',n_jobs=-1,random_state=42),scale=False)
