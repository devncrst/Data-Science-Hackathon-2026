#!/usr/bin/env bash
# FINAL SUBMISSION PREDICTIONS (h4d2).  Usage:
#   H4D_DATA_DIR=/path/to/data scripts/submit_predict.sh <amsterdam_test.parquet> <madrid_test.parquet> <out_dir>
# Uses the committed models in h4d2/artifacts (Stage 1 Madrid + Stage 2 Amsterdam n = 5,25,50,100,200, seed 42).
set -euo pipefail
: "${H4D_DATA_DIR:?set H4D_DATA_DIR to the folder with the two provided parquet files (unlabelled spatial context)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export H4D2_OUT="${H4D2_OUT:-${ROOT}/h4d2/artifacts}"
AMS="$(realpath "$1")"; MAD="$(realpath "$2")"; mkdir -p "$3"; OUTD="$(realpath "$3")"
PY="${PYTHON:-python}"
cd "${ROOT}"
for n in 5 25 50 100 200; do
  "${PY}" -m h4d2 predict --city Amsterdam --n "${n}" --seed 42 --input "${AMS}" --out "${OUTD}/amsterdam_n${n}_predictions.csv"
done
"${PY}" -m h4d2 predict --city Madrid --input "${MAD}" --out "${OUTD}/madrid_predictions.csv"
echo "predictions written to ${OUTD}"
