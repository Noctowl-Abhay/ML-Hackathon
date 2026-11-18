# PathSnap | Final guideline-compliance and submission runbook

Course: UE24CS352A Machine Learning, PES University. Sources checked directly: **Main Guidelines**, **Dataset Exploration**, **Models and Training Guidelines**, **Deliverables**, **Assessment and Plagiarism Policy** (all five original PDFs in `PathSnap_student_materials.zip`).

> **Important:** This is a reproducible *finishing workflow*, not a claim that your computer has already completed every experiment. Real training may take substantial compute. A completed run and independent review are required before GitHub submission. Keep the existing team-created code; the installer backs up replacements and does not overwrite the teammate's UI or the original decoder.

## STEP 0. Use Ubuntu, not the Windows laptop, for experiments

Open a terminal in Ubuntu. Your existing repository is `~/ML-Hackathon`. Do **not** use `git add .` and do not delete or overwrite `Data/` or `artifacts/`.

Use one functioning Python virtualenv consistently. If your older environment is intact, you can use:

```bash
cd ~/ML-Hackathon
source ~/PathSnap/.venv/bin/activate
which python
python -m pip install -r requirements-pathsnap-final.txt
```

If that environment no longer exists, use `source venv2/bin/activate` and install requirements there. Do not randomly alternate environments while running experiments.

## STEP 1. Install this package (once)

Download ZIP into `~/Downloads/PathSnap_PreSubmission_Fix.zip` and run:

```bash
cd ~/ML-Hackathon
unzip -oq ~/Downloads/PathSnap_PreSubmission_Fix.zip -d /tmp
bash /tmp/PathSnap_PreSubmission_Fix/scripts/install_submission_fix.sh "$HOME/ML-Hackathon"
```

Installer copies `scripts/`, `src/reconstruction/api_compatible.py`, contract tests and README; previous versions are backed up in `.pathsnap_backups/`. **It does not replace `ui/app.js`, `ui/index.html`, `ui/style.css`, `src/reconstruction/decoder.py` or the trained model files.**

## STEP 2. Preflight and offline tests

```bash
cd ~/ML-Hackathon
python -m pip install -r requirements-pathsnap-final.txt
python -m pytest -q tests/test_ui_contract.py
python scripts/final_guideline_audit.py --phase preflight
```

If `outputs/features/features_train.parquet` is absent and `Data/processed/train` contains all 80 trips, first inspect `python -m features.build_candidates --help` and then run:

```bash
python -m features.build_candidates --data-root Data/processed --partition train
```

Do not regenerate that file if it already contains the verified training data.

## STEP 3. Actual training + permitted development evaluations

**All model selection and decoder tuning happen only on 80 training and 10 student-testing trips.** The student-testing `.route` files may be read by an offline **evaluation** process to calculate labels; they must never enter the model feature columns or frontend.

```bash
cd ~/ML-Hackathon
bash scripts/run_development_checks.sh 2>&1 | tee pathsnap_development_run.log
```

This runs/resumes: saved 16-trip internal-holdout reports; all-five model student metrics and confusion matrices; student candidate miss rate before label filtering; grouped 3-fold Macro-F1 for all five; 24 Optuna TPE trials with 5-fold trip-grouped objective; an independent untuned-vs-Bayesian comparison on **the same 5 CV folds and same student-test trips**; 10-trip route-continuity comparisons for all five model families; model selection; ten student-trip HMM comparisons; offline Folium map; report evidence draft; and a detailed audit. It does **not** tune using evaluator data.

**Do not report 64/16 holdout as cross-validation**, and do **not** report candidate classification accuracy as route accuracy. Top-1 is computed on *eligible* GPS points with an aligned reference candidate, and sequence F1 is a separate metric.

If the script stops on a real missing prerequisite, fix that specific error; it exits instead of printing a misleading 'done'.

## STEP 4. Review model choice after student testing

```bash
cat artifacts/guideline_reports/student_test/comparison.csv
cat deployment/selected_model_manifest.json
```

The runner generates an **automatic recommendation** from permitted student-test Top-1, Macro-F1 and log loss. The team must independently review it with CV stability, route continuity and compute cost. To explicitly choose a different **already student-evaluated** model:

```bash
python scripts/select_final_model.py --model tuned_full80 --reason 'Our measured student-test and CV evidence supports this model because ...'
```

Or select `bayesian_full80`, `untuned_full80`, `random_forest`, `xgboost`, `deep_mlp`, `distance_weighted_knn`, or `logistic_regression` **only if supported by results**. Choosing a new model requires rebuilding student HMM evidence so it matches the deployed weights.

## STEP 5. Check development audit and generate publication-safe evidence

```bash
python scripts/final_guideline_audit.py --phase development
python scripts/prepare_public_evidence.py
cat artifacts/guideline_reports/final_submission_audit.md
```

A structural PASS proves a file/metric exists and passes minimal consistency checks, **not** that a team member can defend the work. In particular, check the original five PDFs and report quality manually.

## STEP 6. Final evaluation on Ubuntu laptop (Laptop 1)

**Only after you freeze model choice, feature config and HMM parameters.** Put evaluator trip files in `Data/processed/evaluator_test/` on Ubuntu; never push them to GitHub. For the **final demonstration only**, the API requires a server-private reference feature table to compute aggregate accuracy. If `outputs/features/features_evaluator_test.parquet` is missing, and the model is frozen, generate it with:

```bash
python scripts/prepare_final_evaluator_server.py --final-evaluator
```

The generator reads `.route` *only for private reference labels after model freeze*. The API's inference branch recomputes input features separately without `.route`; it then compares the prediction with the private reference server-side. The generated table is ignored by Git and must never be returned to Windows. If the file already exists, the script preserves it rather than overwriting.

Launch the *real* API from the repo root:

```bash
cd ~/ML-Hackathon
PATHSNAP_MODEL=deployment/selected_model.joblib \
PATHSNAP_CONFIG=deployment/feature_config.json \
PATHSNAP_FEATURES=outputs/features/features_evaluator_test.parquet \
PATHSNAP_TRIPS=Data/processed/evaluator_test \
python -m uvicorn src.reconstruction.api_compatible:app --host 0.0.0.0 --port 8000 --access-log
```

**In a second Ubuntu terminal**, check health and 10-trip listing, without inferring accuracy from the UI alone:

```bash
python scripts/api_smoke_check.py
```

When frozen and ready for the final evaluator-only smoke test:

```bash
python scripts/api_smoke_check.py --all-trips --final-evaluator
```

The smoke check creates private responses under `artifacts/api_checks/` which are **not published**. It checks every trip returns non-empty GPS and reconstructed geometry. Do not tune against those evaluator outcomes.

## STEP 7. Windows teammate UI (Laptop 2)

Update the **actual cloned GitHub repo** only after preserving any local UI edits:

```powershell
cd C:\Users\Rakshita\Downloads\ML-Hackathon
git status --short
git pull --ff-only origin main
cd ui
py -m http.server 5500 --bind 127.0.0.1
```

Browse `http://127.0.0.1:5500` and enter `http://<UBUNTU_LAN_IP>:8000` in the UI API field. Do **not** run `uvicorn test_api:app` on Windows. The Ubuntu backend must show GET `/trips` and POST `/predict` access logs **from the Windows IP**. Use the two laptops on the same network; institutional Wi-Fi can prevent peer-to-peer connections.

Take real screenshots and save the files (without private route labels or personal credentials) as:

- `submission/screenshots/ui_windows.png`
- `submission/screenshots/api_ubuntu_logs.png`

## STEP 8. Consolidated PDF report

Your **only report deliverable** is the final consolidated PDF:

`submission/PathSnap_Final_Technical_Report.pdf`

Do **not** use the earlier generated draft unchanged. Complete it in the team's own words with actual graphs, metrics, screenshots and independent explanations, following `REPORT_REQUIREMENTS.md`. The PDF must contain sections 2-7, one subsection **per each of the five models**, experiment evidence, Optuna trial analysis and tuned comparisons, HMM details and comparisons, real Folium map screenshots, and API/UI screenshots. Include AI-assistance disclosure if required by the event's academic policy.

## STEP 9. Safe GitHub preparation and manual push

**Only after a real full run and human review.**

```bash
cd ~/ML-Hackathon
python scripts/final_guideline_audit.py --phase submission
python scripts/prepare_public_evidence.py
bash scripts/safe_stage_git.sh
git status --short
git diff --cached --name-only
git diff --cached --check
```

The audit intentionally marks subjective items `MANUAL`; it cannot certify the two physical laptops, integrity, or the real PDF by file presence alone. Confirm those human gates before proceeding, even if all automated checks pass.

Check if any evaluator-only files **were already tracked** (ignoring them now does not remove past commits):

```bash
git ls-files | grep -E '(^|/)evaluator_test/|\.route$' || true
```

If sensitive evaluator files are tracked, **do not push** until you reconcile them. Do not `git add .`. The staging script refuses restricted staged paths and files that violate GitHub's 100 MiB limit.

Once personally satisfied with the staged paths and the team's independent work:

```bash
git commit -m "Complete PathSnap evaluation, reconstruction, deployment and report"
git push origin main
```

Verify the public GitHub repository is accessible to the evaluators and contains code, tracked results, one final PDF, and a deployable model/config. The ZIP package here does **not** itself upload or push anything.

## What cannot be automatically certified

- Actual run correctness and numerical validity of models trained on your Ubuntu machine.
- Independence/authenticity and other teams' code similarity.
- Successful Windows-Ubuntu networking and evaluator clicking all 10 routes.
- Whether the PDF accurately discusses real patterns or model failures in your own words.
- Previously made use of evaluator labels during development: maintain an honest experiment history and do not use evaluator outcomes to tune or select models further.
