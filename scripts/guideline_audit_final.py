#!/usr/bin/env python3
"""Evidence-based audit against PathSnap's actual PDFs. Never invents completion."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path.cwd()
RESULT = ROOT / "artifacts/guideline_reports"
CHECKS = []


def check(name, success, evidence, why=""):
    CHECKS.append({"requirement": name, "pass": bool(success),
                   "evidence": evidence, "next_action": "" if success else why})


def rows(p):
    p = ROOT / p
    if not p.exists():
        return []
    try:
        with p.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def trips(partition):
    p = ROOT / f"Data/processed/{partition}"
    if not p.is_dir():
        return -1
    return len([x for x in p.iterdir() if x.is_dir() and x.name.isdigit()])


def present(path):
    return (ROOT / path).is_file()


def model_paths():
    alternatives = {
        "logistic_regression": ["logistic_regression"],
        "distance_weighted_knn": ["distance_weighted_knn", "knn"],
        "random_forest": ["random_forest"],
        "xgboost": ["xgboost"],
        "deep_mlp": ["deep_mlp", "mlp"],
    }
    return {name: any(present(f"artifacts/{option}/model.joblib") for option in options)
            for name, options in alternatives.items()}


def main():
    check("80 training trips", trips("train") == 80, f"Data/processed/train count={trips('train')}",
          "Restore all 80 processed training trips")
    check("10 student-testing trips", trips("student_test") == 10, f"Data/processed/student_test count={trips('student_test')}",
          "Restore all 10 student-test trips")
    check("10 reserved evaluator trips", trips("evaluator_test") == 10, f"Data/processed/evaluator_test count={trips('evaluator_test')}",
          "Final demo machine must contain 10 evaluator trips")

    for name, available in model_paths().items():
        check(f"Trained model: {name}", available, "artifacts/<model>/model.joblib",
              "Train and save this model")
    holdout = rows("artifacts/guideline_reports/validation/comparison.csv")
    check("Five-model 16-trip holdout reporting", len(holdout) >= 5,
          "artifacts/guideline_reports/validation/comparison.csv",
          "Run guideline_metrics.py --mode validation")

    cv = rows("artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv")
    check("Grouped CV Macro-F1 for all five models", len(cv) >= 5,
          "artifacts/guideline_reports/all_models_cv/cv_all_models_summary.csv",
          "Run cv_all_models.py on 80 training trips")

    student_rows = rows("artifacts/guideline_reports/student_test/comparison.csv")
    model_names = {r.get("model", "") for r in student_rows}
    baseline_names = {"logistic_regression", "distance_weighted_knn", "random_forest", "xgboost", "deep_mlp"}
    check("Five-model STUDENT-test comparison", baseline_names <= model_names,
          "artifacts/guideline_reports/student_test/comparison.csv",
          "Build student-test labelled features; run guideline_metrics.py --mode student")
    check("Untuned vs grid-tuned STUDENT-test", {"untuned_full80", "tuned_full80"} <= model_names,
          "artifacts/guideline_reports/student_test/comparison.csv", "Score both saved full80 LR models on student test")
    check("Bayesian tuned STUDENT-test comparison", "bayesian_full80" in model_names,
          "artifacts/guideline_reports/student_test/comparison.csv", "Complete Optuna run, rerun student eval")
    check("Student candidate miss rate before filtering", present("artifacts/guideline_reports/student_test/candidate_coverage.json"),
          "artifacts/guideline_reports/student_test/candidate_coverage.json", "Run student-test model evaluation")
    check("Actual student-test confusion matrices", all(present(f"artifacts/guideline_reports/student_test/{n}_confusion_matrix.png")
                                                      for n in baseline_names),
          "artifacts/guideline_reports/student_test/*_confusion_matrix.png",
          "Run student evaluation for all five models")

    bayes = ROOT / "artifacts/logistic_regression_bayesian"
    trialfile = bayes / "optuna_trials.csv"
    trialrows = rows("artifacts/logistic_regression_bayesian/optuna_trials.csv")
    check("Optuna TPE trial log (>=24 trials)", len(trialrows) >= 24,
          f"{trialfile} (n={len(trialrows)})", "Run or resume train-only Bayesian tuning")
    check("Bayesian best model + CV metrics", (bayes / "model_all_80.joblib").exists() and
          (bayes / "cv_oof_metrics.json").exists() and (bayes / "best_params.json").exists(),
          "artifacts/logistic_regression_bayesian/", "Complete the final fit and OOF scoring")
    check("Model hyperparameters and real MLP history export",
          present("artifacts/guideline_reports/parameters/model_hyperparameters.json"),
          "artifacts/guideline_reports/parameters/model_hyperparameters.json",
          "Run export_hyperparameters_history.py")

    compare = ROOT / "artifacts/reconstruction/student_test/comparison.json"
    summaries = []
    if compare.exists():
        try:
            v = json.loads(compare.read_text())
            summaries = v if isinstance(v, list) else []
        except Exception:
            pass
    check("Classifier vs HMM on all ten STUDENT trips", len(summaries) == 10 and all(
          (r.get("evaluation") or {}).get("hmm") is not None for r in summaries),
          "artifacts/reconstruction/student_test/comparison.json",
          "Run reconstruct_routes.py with STUDENT-test features, root and output")
    check("HMM segment/gap and continuity code", present("src/reconstruction/decoder.py"),
          "src/reconstruction/decoder.py", "Install reconstruction module")
    check("API UI contract adapter + mock tests", present("src/reconstruction/api_compatible.py") and
          present("tests/test_ui_contract.py"),
          "src/reconstruction/api_compatible.py, tests/test_ui_contract.py", "Install PathSnap final repair")
    check("Teammate UI (unchanged)", all(present(f"ui/{x}") for x in ("index.html", "app.js", "config.js", "style.css")),
          "ui/index.html, ui/app.js, ui/config.js, ui/style.css", "Pull teammate's UI")

    folium_code = present("scripts/render_folium_guidelines.py")
    maps = list((ROOT / "artifacts/reconstruction/student_test").glob("*/road_network_map.html"))
    check("Folium network+GPS+route code & generated map", folium_code and bool(maps),
          "scripts/render_folium_guidelines.py + road_network_map.html", "Generate a student-test map")
    check("Public repository README", present("README.md"), "README.md", "Write install, run, deployment instructions")
    report_files = list(ROOT.glob("*.pdf"))
    check("ONE consolidated final PDF report", bool(report_files),
          ", ".join(str(x.name) for x in report_files) or "none at repo root",
          "Generate and review the actual technical report PDF; include report sections 2-7")
    check("Laptop 2 screenshots + Laptop 1 API log screenshots", bool(list(ROOT.glob("**/*ui*demo*.png")))
          and bool(list(ROOT.glob("**/*api*log*.png"))),
          "Screenshots under repository", "Capture real two-laptop demo and API log screens")

    passing = sum(c["pass"] for c in CHECKS)
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "guideline_audit.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passing, "total": len(CHECKS), "checks": CHECKS}, indent=2))
    lines = ["# PathSnap guideline evidence audit", "",
             f"**{passing} / {len(CHECKS)} evidence checks passed.**", "",
             "This checks saved evidence, NOT qualitative correctness or whether the PDF is well-written.",
             "", "| Status | Requirement | Evidence / next action |", "|---|---|---|"]
    for c in CHECKS:
        detail = c["evidence"] if c["pass"] else c["next_action"]
        lines.append(f"| {'PASS' if c['pass'] else 'PENDING'} | {c['requirement']} | {detail} |")
    lines += ["", "**Important:** The 10 evaluator trips must not be used to choose models, parameters, or decoder weights.",
              "This audit cannot prove independence, no plagiarism, correct pointwise labels, or two physical laptops; team must demonstrate these."]
    (RESULT / "guideline_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
