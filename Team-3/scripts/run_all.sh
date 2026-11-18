#!/usr/bin/env bash
# CHAMPION PIPELINE = h4d2 (the legacy h4d baseline is scripts/run_all_h4d_legacy.sh)
set -euo pipefail
: "${H4D_DATA_DIR:?set H4D_DATA_DIR to the folder with the two parquet files}"
export H4D2_OUT="${H4D2_OUT:-./h4d2/artifacts}"
if [ ! -f "${H4D2_OUT}/stage1_madrid.pt" ]; then python -m h4d2 stage1; fi      # ~1.5 h on 16 CPUs if retraining
python -m h4d2 sweep --seeds 42 --rng-mode notebook                          # headline (organiser NB4 draws)
python -m h4d2 sweep --seeds 42 --rng-mode per_n                             # robustness
for n in 5 25 50 100 200; do python -m h4d2 adapt --n "${n}" --seed 42; done
python -m h4d2.report
echo "DONE -> ${H4D2_OUT}"
