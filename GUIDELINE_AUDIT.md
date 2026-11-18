# PathSnap — exact-PDF compliance checklist (18 Nov 2026)

Sources inspected: `Main Guidelines.pdf`, `Models and Training Guidelines.pdf`, `Deliverables.pdf`, `Assessment and Plagiarism Policy.pdf` from the user's `PathSnap_student_materials.zip`.

**The previous 10-trip evaluator experiments must not be used for tuning or final model/threshold selection.** They can be described transparently as prior diagnostic experiments, but that evaluator partition is supposed to remain untouched during development under the assessment policy. The guideline also requires using the 10 *student-test* trips for development comparison.

## Mandatory items and verification

| Exact requirement | Status before this package | How to meet/verify |
|---|---|---|
| ≥5 different model families trained | DONE | Existing `scripts/compare_models.py`, `artifacts/*/metrics.json` |
| Record configurations and comparative evidence | PARTIAL | Existing five training scripts + report table, config filenames |
| 80 train trips for fitting/group CV | DONE for tuned LR | Existing `cv_oof_metrics.json`; `scripts/cv_all_models.py` for all five models |
| 10 student-test trips for development comparison | MISSING | `scripts/build_student_test_labels.py` then `scripts/guideline_metrics.py --mode student` |
| Accuracy, precision, recall, F1, **Macro F1**, log loss, confusion matrix | PARTIAL | `scripts/guideline_metrics.py` generates confusion PNG + full metrics |
| **Candidate miss rate BEFORE label_usable filtering** | MISSING | Student eval script reports true pre-filter proxy candidate coverage |
| Training and inference times | PARTIAL | Existing model `metrics.json`; CV script logs times |
| Cross-validation Macro-F1 graph for *all* five models | MISSING | `scripts/cv_all_models.py` + graph from actual results |
| Student-test Top-1 and Macro-F1 comparison plots | MISSING | `scripts/guideline_metrics.py --mode student` |
| Log loss comparison plot | MISSING | `scripts/guideline_metrics.py` |
| Training/validation learning curves for models with epoch history | NOT VERIFIED | Only for MLP if `loss_curve_`/validation history were saved; never fabricate |
| **Bayesian** hyperparameter search, trials, search space, objective | MISSING | `scripts/tune_lr_bayesian.py` with Optuna/TPE |
| Tuned vs untuned metrics on **same student-test** trips | MISSING | Student eval script evaluates both saved full80 models and Bayesian model if specified |
| HMM states/emission/transition/gaps/missing/disconnect | IMPLEMENTED; needs REVIEW | `src/reconstruction/decoder.py` and tests |
| Classifier-only vs HMM metrics and sequence performance | RECONSTRUCTION EXISTS | `python scripts/reconstruct_routes.py --trip all`, inspect `comparison.json` |
| Decoder optimisation on *train/student only* | NOT VERIFIED | Record train/student-only CV search, **not evaluator** |
| Return a segment for every applicable GPS observation | IMPLEMENTED, verify | Per-trip `summary.json` and GPS row count |
| Folium: raw GPS, road geometry, reference (report side), predicted | PARTIAL | `render_folium.py`; visually inspect layer coverage and export report screenshot |
| API Laptop 1, UI Laptop 2, ten trip selector | STARTER EXISTS; unverified friend UI | `src/reconstruction/api_compatible.py` supports `GET /trips`, `POST /predict` |
| Reference routes not sent to UI / used for inference | INTENDED in API | Inspect actual API JSON for `reference_*`, `label_*`, `.route` before demo |
| On-site terminal request logs screenshot | NOT DONE | Start uvicorn with `--access-log`, invoke from Laptop 2, capture logs |
| Public GitHub complete code/dependencies/README/configs/reports | NOT YET VERIFIED | `git status`, `git ls-files`, `git push` only after tests |
| ONE consolidated PDF report sections 2–7 | NOT DONE | Include actual tables/figures, every model Q&A, optimisation, HMM, maps, API/UI screenshots |
| Academic integrity, citations, AI disclosure | REQUIRED | Document real experiments and actual independent team contributions |

**Accuracy caveat:** All reported observation Top-1 metrics are conditional on reference-aligned label eligibility, not guaranteed official route reconstruction accuracy. `candidate_recall_points = 1.0` *after label filtering* must not be reported as the real generator candidate recall. Comparing an evaluator trip to the training folds is not an independent development comparison.

**GitHub UI constraint:** The public screenshot confirms there is a `ui/` directory on `main`, but the available browsing environment could not retrieve its source code. This package does not modify its internal markup or assume particular JS selectors. Inspect it locally using `python scripts/ui_integration_audit.py` and connect its actual handlers to `window.PathSnapAPI`.
