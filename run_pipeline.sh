#!/usr/bin/env bash
# Download the NHANES files and build the feature table, start to end.
#   ./run_pipeline.sh                 everything
#   ./run_pipeline.sh --skip-download  data already in $NHANES_DATA
# Data go to $NHANES_DATA (default ./data), outputs to $NHANES_WORK (default ./outputs).
# The pipeline notebooks are executed in place, so their saved tables and figures show this run.
set -euo pipefail
cd "$(dirname "$0")"

export THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

if [[ "${1:-}" != "--skip-download" ]]; then
    python -m nhanes.download
fi

for step in labels step1_nonfeature step2_leakage step3_cycle_overlap step4_rename \
            step5_missing step6_gates fetch_codebooks build_features; do
    echo "== $step  $(date +%H:%M:%S)"
    jupyter execute --inplace "pipeline/$step.ipynb"
done
echo "== done: ${NHANES_WORK:-outputs}/features/"
