from sklearn.linear_model import LogisticRegression
try:
    from .common import run
except ImportError:
    from common import run

if __name__ == '__main__':
    run('logistic_regression',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=42),scale=True)
