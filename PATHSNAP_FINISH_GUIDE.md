# PathSnap Final Repair — installed on existing code and teammate UI

## What this fixes, and what it does NOT do

- Preserves `ui/app.js`, `ui/index.html`, `ui/config.js` and `ui/style.css` without changes.
- Fixes FastAPI `POST /predict`: converts the ORIGINAL reconstruction's real `raw_gps` and `predicted_route` GeoJSON into UI `[lat,lon]` points and `coords` polylines, returns flattened numeric metrics and latency.
- Adds mocked offline tests for the frontend contract, reference-label secrecy and CORS. No evaluator dataset needed for these tests.
- Adds a live API smoke checker which **reads fresh HTTP responses** instead of reading an old `/tmp` file.
- Adds a student/train-only experiment runner, full factual guideline audit, actual-data Folium map builder, and factual report-working-draft generator.

**It cannot run training on the student's laptop from ChatGPT, fabricate PDF screenshots, complete independent team analysis, or claim a final evaluation without actual execution.** Do not select hyperparameters based on evaluator trips.

## 1. Extract and install

First DOWNLOAD the ZIP through Firefox, and confirm `ls ~/Downloads/PathSnap_Final_Repair.zip`.

```bash
cd ~/ML-Hackathon
unzip -oq ~/Downloads/PathSnap_Final_Repair.zip -d /tmp
bash /tmp/PathSnap_Final_Repair/scripts/install_final_repair.sh /tmp/PathSnap_Final_Repair
```

## 2. Pick ONE working Python environment (important!)

The earlier error `(venv2) ... No module named uvicorn` meant that environment was incomplete. Either activate a previously working environment or install requirements into the **CURRENT active virtual environment**:

```bash
cd ~/ML-Hackathon
python -m pip install -r requirements-pathsnap-final.txt
python -c 'import joblib,uvicorn,optuna,pandas,sklearn,pyproj,pyarrow,xgboost,folium,fastapi,httpx; print("DEPENDENCIES OK")'
python -m pytest -q tests/test_ui_contract.py
```

If `pip` cannot find/build an optional package, STOP and read the exact error. Don't report that the setup is complete.

## 3. Start API on Laptop 1 and UI on Laptop 2

**Terminal 1, Laptop 1** (in `~/ML-Hackathon` with same Python):

```bash
python -m uvicorn src.reconstruction.api_compatible:app --host 0.0.0.0 --port 8000 --access-log
```

**Terminal 2, Laptop 1** (check metadata, does not use evaluator labels):

```bash
python scripts/api_smoke_check.py
```

This checks `/health` and ten trip IDs. To test real predictions only AFTER final model selection and freeze:

```bash
python scripts/api_smoke_check.py --all-trips --final-evaluator
```

This tests all 10 reserved trips, saving fresh result JSON and a summary in `artifacts/api_checks/`. It may be time-consuming for long trips. DO NOT tune from these results.

**Laptop 2** (on same private Wi-Fi):

```bash
cd ~/ML-Hackathon/ui
python -m http.server 5500 --bind 0.0.0.0
```

Open `http://127.0.0.1:5500` on Laptop 2. Set its API box to `http://LAPTOP1_LAN_IP:8000` and click Connect. The UI is unchanged. `localhost:8000` on Laptop 2 refers to Laptop 2, NOT Laptop 1.

If the API port is occupied by the old server, stop the older process in its terminal using Ctrl+C, then run the above. You can inspect the listener via `ss -ltnp '( sport = :8000 )'`.

**For evidence:** screenshot working UI on Laptop 2, and request/response/access logs on Laptop 1. The evaluator needs to see those logs.

## 4. Complete experiment evidence on 80 train + 10 student-testing trips

```bash
cd ~/ML-Hackathon
bash scripts/run_required_experiments.sh
```

This orchestrator validates dependencies and trip folders, then runs existing:

- validation reporting and real hyperparameter export;
- student-test feature/weak proxy label construction (no fitting);
- five-model student-test comparison with confusion matrices, Macro F1, Top-1, log loss, candidate miss rate;
- grouped CV for all five models (potentially VERY expensive for MLP / KNN);
- Bayesian Optuna TPE training on 80 TRAIN trips ONLY, preserving trial records;
- Bayesian tuned-vs-untuned student comparison;
- baseline classifier vs HMM on all 10 STUDENT trips;
- offline Folium HTML map showing GPS, relevant road network, predicted route and STUDENT reference;
- factual guideline audit Markdown/JSON.

The orchestrator stops at the first real error and leaves previous outputs intact. Do NOT claim completion until it finishes.

Manual commands for individual stages can be found in `PATHSNAP_FINISH_GUIDE.md` after install and in the previously installed `README_TERMINAL.md` (extracted from `PathSnap_Guideline_Integration.zip`).

## 5. Inspect results and generate report WORKING DRAFT

```bash
python scripts/guideline_audit_final.py
python scripts/build_report_evidence.py
cat artifacts/guideline_reports/guideline_audit.md
```

The draft is saved at `artifacts/guideline_reports/REPORT_WORKING_DRAFT.md`. Add real screenshots, experimental interpretations, and team-written methods before producing the ONE required PDF.

## 6. GitHub safety: don't `git add .` blindly

Your earlier `git status --short` contained untracked `Data/processed/evaluator_test/`, generated `artifacts/`, and `outputs/`. Ensure those directories and `.route` reference data are excluded from general staging:

```gitignore
/Data/processed/
/outputs/features/
/artifacts/api_checks/
/artifacts/reconstruction/
/artifacts/evaluator_test_results/
*.route
```

Add these patterns if missing, without deleting existing `.gitignore` rules. Carefully track scripts, source, relevant result summaries/plots, dependencies and the final deployable model (watch GitHub's 100MB per-file limit). Do NOT commit raw evaluator reference files, secrets, full internal reference maps, or random old artifacts.

Stage explicit reviewed files, not all untracked directories. For example:

```bash
git add src/reconstruction/api_compatible.py tests/test_ui_contract.py \
  scripts/api_smoke_check.py scripts/guideline_audit_final.py \
  scripts/render_folium_guidelines.py scripts/run_required_experiments.sh \
  scripts/build_report_evidence.py requirements-pathsnap-final.txt \
  PATHSNAP_FINISH_GUIDE.md

git diff --cached --stat
```

Then add actual model scripts and curated CSVs, plot images, README, and model artifacts after checking size and confidentiality.

## Academic integrity / reserved evaluator data

The provided PDFs prohibit tuning or selecting models using evaluator-testing labels/results. The earlier development-stage evaluator runs should not influence any model choice. Student-test labels are used for post-hoc evaluation only, not fitting. Route reference must remain server-side and excluded from frontend API responses. The report must be the team's real analysis, with no invented scores or curves.
