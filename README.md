# ML-Hackathon
pathsnap/
├── README.md                  # install, run, deploy (2 laptops)
├── requirements.txt
├── .gitignore                 # exclude secrets, raw data if large
├── configs/
│   ├── data.yaml              # paths, CRS, K, radius
│   ├── models.yaml            # per-model configs + seeds
│   ├── bayes_opt.yaml         # search space, n_trials
│   └── hmm.yaml               # decoder params
├── data/
│   ├── raw/                   # train / student_test (NOT evaluator labels in UI repo)
│   └── processed/             # candidates, features
├── src/
│   ├── data/                  # loaders, cleaning, projection, splits
│   ├── candidates/            # candidate generation
│   ├── features/              # feature engineering
│   ├── models/                # one training script per model
│   ├── evaluation/            # metrics, confusion matrices, plots
│   ├── tuning/                # Bayesian optimisation (Optuna etc.)
│   ├── hmm/                   # emission, transition, viterbi
│   ├── pipeline.py            # trip_id -> predicted route (no labels)
│   └── viz/                   # Folium map builders
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_hmm_analysis.ipynb
├── results/
│   ├── logs/                  # experiment logs, seeds
│   ├── tables/                # CSVs for the report
│   ├── figures/
│   ├── maps/                  # exported Folium HTML
│   └── optuna/                # trial history
├── artifacts/                 # saved model, scaler, hmm config
├── api/
│   ├── main.py                # FastAPI app, model loading
│   ├── routes.py              # /trips, /predict
│   └── evaluator.py           # server-side ground truth scoring only
├── ui/                        # frontend for Laptop 2
├── scripts/                   # run_all.sh, export_artifacts.py
└── report/                    # final PDF + screenshots
