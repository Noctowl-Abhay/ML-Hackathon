# Mandatory report checklist (checked against Deliverables.pdf, pages 2-7)

**One consolidated PDF per team.** Actual results only; no fabricated curves, graphs, screenshots, scores, or references. The source code and result files must support every reported number. Write analysis in the team's own words.

## Section 2.2 Dataset understanding / EDA

- 80 training, 10 student testing, 10 evaluator testing (reserved final only). Count actual observation, node and arc rows from inventory.
- Explain `.track` (longitude, latitude, timestamp), `.nodes`, directed `.arcs`, `.route` as ordered *arc IDs*, not one route arc per GPS point.
- Describe non-destructive cleaning and GPS timestamp gaps, missing/invalid values, network connectivity and disconnected components.
- Explain GPS/road coordinate projections, candidate search and fallback (with confirmed config values).
- Justify **each family** of the 25 features (distance, heading, movement/time, topology), label alignment and reference-proxy limitations.
- Include actual raw GPS plots, road-network visualisations, and a **real Folium map**. Label each plot's data partition.

## Section 3.1 Required five-model comparison table

Columns in original PDF: Model, Algorithm Family, Key Configuration, **CV Macro F1**, Top-1 Accuracy, Route Continuity, **Student-Test Macro F1**, Log Loss, Inference Time. Include a separate column showing which partition is used for each metric. Distinguish training 64/16 holdout from train-grouped CV and student test. If route continuity was only measured for a selected model, state `N/A; not measured for this model` and explain that; do not copy the same value across models.

## Section 3.2 Separate questions answered for EACH model

For Logistic Regression, distance-weighted KNN, Random Forest, XGBoost, Deep MLP:

1. Why is it potentially suitable for GPS-to-road candidate classification?
2. Strengths and limitations for this particular data, supported by observed results.
3. Initial hyperparameters (actual fit configuration) and training objective or loss.
4. **10-trip student-test accuracy, Top-1, Macro F1 and log loss**, where applicable.
5. Evidence for overfitting/underfitting: compare grouped CV and student-test, not unverified generalities.
6. Training/inference time and how it affects two-laptop inference.

## Section 3.3 Compulsory actual figures

- All-five model **CV Macro-F1** comparison from GroupKFold.
- All-five **student-test** Top-1 and Macro-F1 comparison.
- Applicable log-loss comparison.
- Actual confusion matrices (the report calls them optional diagnostics, though code should generate them).
- Epoch-wise training/validation loss **only when the model genuinely exposes those values**; do not invent for non-iterative models.
- Additional comparisons supporting the final choice.

## Section 4 Bayesian hyperparameter optimisation

- Explain chosen model, search dimensions/ranges, TPE Bayesian strategy, grouped training-only CV objective, seed, total completed trials, trial log, best parameters, fit cost.
- Compare same-split untuned-vs-Bayesian CV Macro F1, 10-trip student-test Top-1, Macro F1, log loss and inference time.
- Preserve `optuna_trials.csv`, `best_params.json`, `cv_oof_metrics.json` and selected model weights.

## Section 5 HMM / Viterbi

- Hidden states = road-arc candidates per GPS point; explain emissions from classifier scores and road-connected transitions.
- Show Viterbi recurrence (or exactly how implementation differs), large GPS gaps, missing candidates, impossible/disconnected transitions and route expansion.
- Compare classifier-only vs HMM on **permitted student-test data**: pointwise directed Top-1, route continuity, ordered sequence metric, change by trip including failures.
- Explain conditional GPS-point labels and limitations of the derived reference-alignment metric.

## Section 6 Folium map

- Real map of student-test trip with original GPS, road network, predicted route and student reference (only in report/offline).
- Different colors/markers, legend, fit bounds/zoom, layer control and screenshot in PDF.
- Commit **generation code**; do not send private reference geometry to public UI or GitHub evidence.

## Section 7 API/UI/deployment

- FastAPI + model-loading details, live inference, locally available evaluator data, correct HTTP GET `/trips` and POST `/predict`, no hardcoded outputs.
- Ubuntu Laptop 1 and Windows Laptop 2 actual commands, IPs, request logs at API receiving terminal and screenshot of UI result.
- All ten reserved trips selectable; no reference arc ID or per-point ground truth is sent to frontend.

## Academic integrity and conclusion

- Make actual claims only, reference the team repo commit and reproducibility steps.
- Identify genuine limitations, disclose AI assistance as required, and be able to defend every model and design choice.
- Do not use reserved evaluator outcomes for any further model selection/tuning. The UI may display **aggregate accuracy** computed server-side at final evaluation, but never private ground-truth route data.
