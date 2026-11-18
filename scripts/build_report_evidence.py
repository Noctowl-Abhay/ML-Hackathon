#!/usr/bin/env python3
"""Generate a factual Markdown WORKING DRAFT from real results, never fabricated values."""
from __future__ import annotations
import csv
import json
from pathlib import Path

BASE = Path("artifacts/guideline_reports")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def metric(row, field, digits=4):
    raw = row.get(field)
    if raw in (None, "", "nan", "NaN"):
        return "N/A (not measured)"
    try:
        return f"{float(raw):.{digits}f}"
    except (TypeError, ValueError):
        return str(raw)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    stu = read_csv(BASE / "student_test/comparison.csv")
    cvs = read_csv(BASE / "all_models_cv/cv_all_models_summary.csv")
    val = read_csv(BASE / "validation/comparison.csv")
    cv_by = {r["model"]: r for r in cvs}
    continuity=read_csv(BASE / 'model_route_continuity/model_continuity_summary.csv')
    cont_by={r.get('model'):r for r in continuity}
    st_by = {r["model"]: r for r in stu}
    names = ["logistic_regression", "distance_weighted_knn", "random_forest", "xgboost", "deep_mlp"]
    names += [x for x in ["untuned_full80", "tuned_full80", "bayesian_full80"] if x in st_by]
    lines = [
        "# PathSnap — WORKING DRAFT (NOT READY TO SUBMIT)", "",
        "> This file contains only imported numerical results and prompts for the team's own",
        "> analysis. Authors must verify every figure, add real screenshots, and write",
        "> explanations in their own words before exporting a final PDF.", "",
        "## 2. Dataset Understanding and Exploratory Data Analysis", "",
        "[TEAM: explain file formats, 80/10/10 partitions, missing values, GPS time gaps, ",
        "candidate generation, label-proxy limitations, and justify 25 features.]", "",
        "[INSERT: missing-value plots, GPS trajectory plot, road-network plot, Folium screenshot.]", "",
        "## 3. Model Experimentation and Comparison", "",
        "### 3.1 Five-model and tuned-model comparison", "",
        "| Model | Grouped CV Macro-F1 | Student Top-1 | Student Macro-F1 | Student log loss | Student inference seconds | Student route continuity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        a, b = cv_by.get(name, {}), st_by.get(name, {})
        lines.append(f"| {name} | {metric(a,'cv_macro_f1_mean')} | {metric(b,'top1_accuracy')} | "
                     f"{metric(b,'macro_f1')} | {metric(b,'log_loss')} | {metric(b,'inference_seconds',2)} | "
                     f"{metric(cont_by.get(name,{}),'route_continuity')} |")
    lines.extend(["", "**Metric note:** Top-1 is conditional on a reference-aligned eligible candidate set.",
                  "Do not conflate this value with end-to-end connected-route accuracy.", "",
                  "### 3.2 Compulsory individual-model discussion", ""])
    for name in names[:5]:
        lines += [f"#### {name}", "",
                  "[TEAM: Why considered? Strengths and limitations? Important initial hyperparameters?",
                  "Actual loss/optimisation objective? Student-test metrics? Overfitting/underfitting evidence?]", ""]
    lines += ["### 3.3 Plots", "",
              "[INSERT: all-model CV Macro-F1 comparison, student Top-1 and Macro-F1," ,
              "log-loss comparison, confusion matrices, genuine MLP epoch-wise training curve if available.]", "",
              "## 4. Bayesian Hyperparameter Optimisation", ""]
    best = Path("artifacts/logistic_regression_bayesian/best_params.json")
    if best.exists():
        obj = json.loads(best.read_text())
        lines += ["Actual recorded best parameters:", "", "```json", json.dumps(obj, indent=2), "```", ""]
    else:
        lines += ["[PENDING: actual Optuna best_params.json, trial plots and tuned-vs-untuned results]", ""]
    lines += ["[TEAM: justify search space/objective, explain trials, computational cost, ",
              "CV Macro-F1 and student-test comparison on SAME partition.]", "",
              "## 5. HMM and Viterbi Route Reconstruction", ""]
    comparisons = Path("artifacts/reconstruction/student_test/comparison.json")
    if comparisons.exists():
        experiments = json.loads(comparisons.read_text())
        lines += ["| Trip | Classifier directed Top-1 | HMM directed Top-1 | Broken route switches | Sequence F1 |",
                  "|---|---:|---:|---:|---:|"]
        for item in experiments:
            e, r = item.get("evaluation", {}), item.get("reconstruction", {})
            lines.append(f"| {item.get('trip_id','?')} | {metric(e.get('baseline',{}),'directed')} | "
                         f"{metric(e.get('hmm',{}),'directed')} | {r.get('disconnected_arc_switches','N/A')} | "
                         f"{metric(e.get('sequence',{}),'ordered_sequence_match_f1')} |")
    else:
        lines += ["[PENDING: Run HMM reconstruction on all ten STUDENT-test trips.]", ""]
    lines += ["", "[TEAM: explain hidden segment states, emission probability, network shortest-path",
              "transitions, gap resets, invalid transitions, improvements and failures.]", "",
              "## 6. Folium Geographical Visualisation", "",
              "[INSERT: actual screenshots, road network, raw GPS, predicted route, student reference",
              "(offline report only); include interactive HTML generation path.]", "",
              "## 7. FastAPI, UI and Two-Laptop Demo", "",
              "[INSERT: actual UI screenshots, receiving-laptop access logs and endpoint response",
              "screenshots, hardware/IP info, exact run commands.]", "",
              "## Conclusions and Reproducibility", "",
              "[TEAM: actual findings, limitations, citations, GitHub commit, and AI assistance disclosure",
              "as required by event policy.]", ""]
    if val:
        lines += ["## Appendix: earlier 64/16 trip-grouped holdout (NOT CV or student test)", "",
                  "| Model | Holdout Macro-F1 | Holdout Top-1 | Holdout log loss |",
                  "|---|---:|---:|---:|"]
        for row in val:
            lines.append(f"| {row['model']} | {metric(row,'macro_f1')} | {metric(row,'top1_accuracy')} | {metric(row,'log_loss')} |")
    target = BASE / "REPORT_WORKING_DRAFT.md"
    target.write_text("\n".join(lines) + "\n")
    print("CREATED:", target)
    print("WARNING: not a final report; manually complete team explanations and evidence.")


if __name__ == "__main__":
    main()
