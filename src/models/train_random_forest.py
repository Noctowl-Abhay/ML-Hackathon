from sklearn.ensemble import RandomForestClassifier
try:
    from .common import run
except ImportError:
    from common import run

if __name__ == '__main__':
    run('random_forest',RandomForestClassifier(n_estimators=200,min_samples_leaf=3,max_depth=18,class_weight='balanced_subsample',n_jobs=-1,random_state=42),scale=False)
