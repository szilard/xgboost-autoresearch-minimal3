#!/bin/bash
#
# Evaluate every kept experiment in output/results.tsv against the ground truth holdout set.
#
# Scores the saved artifact (model + prepare) of each commit from artifacts/,
# so nothing is retrained and the repo is never modified.
#
# Usage:
#   ./human/run_groundtruth_all.sh [results.tsv] [output.tsv] [timeout_seconds]
#
# Defaults:
#   results.tsv      -> output/results.tsv
#   output.tsv       -> output/groundtruth_all.tsv
#   timeout_seconds  -> 3000
#
# Prerequisites:
#   - results.tsv must exist with columns: commit, Eval_AUC, status, description
#   - human/check_groundtruth.py and harness.py (repo root) must exist
#   - The virtual environment (if any) should be activated before running
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_INPUT="${1:-output/results.tsv}"
OUTPUT_FILE="${2:-output/groundtruth_all.tsv}"
TIMEOUT="${3:-3000}"

if [ ! -f "$RESULTS_INPUT" ]; then
  echo "ERROR: $RESULTS_INPUT not found. Run experiments first."
  exit 1
fi

# Write header
echo -e "commit\tstatus\tdescription\teval_auc\tholdout_auc" > "$OUTPUT_FILE"

LINE_NUM=0
while IFS=$'\t' read -r commit eval_auc status description; do
  LINE_NUM=$((LINE_NUM + 1))

  # Skip header
  [ "$LINE_NUM" -eq 1 ] && continue

  # Skip empty lines
  [ -z "$commit" ] && continue

  echo "=== [$LINE_NUM] $commit | $status | $description ==="

  # Only kept experiments are scored (crashes have no artifact; discards are skipped to save time)
  if [ "$status" != "keep" ]; then
    echo "  SKIP: status $status"
    echo -e "$commit\t$status\t$description\t$eval_auc\tN/A" >> "$OUTPUT_FILE"
    continue
  fi

  GT_LOG=$(mktemp)
  EXIT_CODE=0
  timeout "$TIMEOUT" python3 human/check_groundtruth.py "$commit" > "$GT_LOG" 2>&1 || EXIT_CODE=$?
  if [ "$EXIT_CODE" -ne 0 ]; then
    [ "$EXIT_CODE" -eq 124 ] && echo "  TIMEOUT: exceeded ${TIMEOUT}s" || echo "  CRASH: exit code $EXIT_CODE"
    tail -5 "$GT_LOG" 2>/dev/null || true
    echo -e "$commit\t$status\t$description\t$eval_auc\tCRASH" >> "$OUTPUT_FILE"
    rm -f "$GT_LOG"
    continue
  fi

  HOLDOUT_AUC=$(grep "^Holdout AUC:" "$GT_LOG" | grep -oP '[\d.]+$' || echo "N/A")
  rm -f "$GT_LOG"

  echo "  holdout=$HOLDOUT_AUC"
  echo -e "$commit\t$status\t$description\t$eval_auc\t$HOLDOUT_AUC" >> "$OUTPUT_FILE"
done < "$RESULTS_INPUT"

echo ""
echo "=== DONE: results written to $OUTPUT_FILE ==="
