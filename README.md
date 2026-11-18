# ML-Hackathon
```
pathsnap/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── models.yaml
│   ├── bayes_opt.yaml
│   └── hmm.yaml
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   └── student_test/
│   └── processed/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessing.py
│   │   ├── projection.py
│   │   └── splits.py
│   │
│   ├── candidates/
│   │   ├── __init__.py
│   │   └── generate.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── extract.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── gradient_boosting.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plots.py
│   │
│   ├── tuning/
│   │   ├── __init__.py
│   │   └── optuna_tuning.py
│   │
│   ├── hmm/
│   │   ├── __init__.py
│   │   ├── emission.py
│   │   ├── transition.py
│   │   └── viterbi.py
│   │
│   ├── viz/
│   │   ├── __init__.py
│   │   └── folium_map.py
│   │
│   └── pipeline.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_hmm_analysis.ipynb
│
├── results/
│   ├── logs/
│   ├── tables/
│   ├── figures/
│   ├── maps/
│   └── optuna/
│
├── artifacts/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   └── evaluator.py
│
├── ui/
│   ├── app.py
│   └── requirements.txt
│
├── scripts/
│   ├── run_all.sh
│   └── export_artifacts.py
│
├── tests/
│   ├── test_candidates.py
│   ├── test_features.py
│   └── test_viterbi.py
│
└── report/
```
