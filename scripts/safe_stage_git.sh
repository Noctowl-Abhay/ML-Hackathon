#!/usr/bin/env bash
# Stage only reviewed source, deployment config/weights, and curated evidence.
# DOES NOT commit or push. User reviews staged diff, then explicitly commits.
set -euo pipefail
cd "$(dirname "$0")/.."
test -d .git || { echo 'ERROR: Not a git repo' >&2; exit 3; }
# Ignore rules do not remove already-committed evaluator files. Block until reviewed.
TRACKED_PRIVATE=$(git ls-files | grep -E '(^|/)evaluator_test/|^outputs/features/features_evaluator_test|^artifacts/evaluator_test' || true)
if [ -n "$TRACKED_PRIVATE" ]; then
  printf 'BLOCKED: evaluator files are ALREADY tracked in Git:\n%s\n' "$TRACKED_PRIVATE" >&2
  echo 'Review and untrack safely before committing; do not delete original local data.' >&2
  exit 7
fi
for path in deployment/selected_model.joblib deployment/feature_config.json deployment/selected_model_manifest.json \
 submission/PathSnap_Final_Technical_Report.pdf; do
  test -s "$path" || { echo "BLOCKED: missing $path" >&2; exit 4; }
done
# Do not auto-stage the raw Data, outputs, or artifacts, including evaluator references.
for path in .gitignore README.md README_SUBMISSION.md REPORT_REQUIREMENTS.md requirements-pathsnap-final.txt \
  src/ features/ scripts/ tests/ ui/ deployment/ submission/ api/ notebooks/; do
  if [ -e "$path" ]; then git add -- "$path"; fi
done
# Also stage root-level training/evaluation utilities, if present.
for path in ./*.py ./*.ipynb; do
  case "${path##*/}" in test_api.py|mock_*.py|*backup*.py) continue ;; esac
  if [ -f "$path" ]; then git add -- "$path"; fi
done
mapfile -t STAGED < <(git diff --cached --name-only)
if [ "${#STAGED[@]}" -eq 0 ]; then echo 'Nothing staged'; exit 0; fi
invalid=0
for path in "${STAGED[@]}"; do
  case "$path" in
     Data/*|outputs/*|artifacts/*|*evaluator_test*|*.route|*.track|*.arcs|*.nodes|*.env|*.pem|*.key|*.p12|*.sqlite|*.db|*.zip|*.csv.gz)
       echo "BLOCKED: restricted path staged: $path" >&2;invalid=1;;
  esac
  if [ -f "$path" ]; then
    bytes=$(stat -c %s "$path")
    if [ "$bytes" -gt 99614720 ]; then
      echo "BLOCKED: GitHub individual file limit exceeded: $path ($bytes bytes)" >&2;invalid=1
    fi
  fi
done
if [ "$invalid" -ne 0 ];then
  echo 'REVIEW staged paths; to unstage a forbidden file: git restore --staged -- PATH' >&2
  exit 5
fi
# grep only staged text diff, avoid scanning binary weights.
if git diff --cached --unified=0 -- . ':!deployment/*.joblib' | grep -En 'gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY|AKIA[0-9A-Z]{16}' ; then
 echo 'BLOCKED: potential secret in staged content' >&2;exit 6
fi
git diff --cached --check
printf '\nSTAGED FILES (not committed, not pushed):\n'
git diff --cached --stat
printf '\nReview: git diff --cached --name-only && git status --short\n'
printf 'Verify originality and permissions, then commit/push ONLY after the audit and PDF review.\n'
