from sklearn.neighbors import KNeighborsClassifier
try:
    from .common import run
except ImportError:
    from common import run

if __name__ == '__main__':
    run('distance_weighted_knn',KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    algorithm="auto",
    leaf_size=40,
    metric="euclidean",
    n_jobs=4
),scale=True)
