#!/usr/bin/env bash
set -euo pipefail
: "${H4D_DATA_DIR:?set H4D_DATA_DIR to the folder with the two parquet files}"
export H4D_OUT_DIR="${H4D_OUT_DIR:-./outputs}"
python -m h4d.cli --force preprocess
python -m h4d.cli --force stage1
python -m h4d.cli --force zeroshot
python -m h4d.cli sweep                       # per_n, official + extra (robustness check)
python -m h4d.cli sweep --rng-mode notebook   # organiser NB4 draw order = headline/deck numbers
for n in 5 25 50 100 200; do python -m h4d.cli adapt --n "$n"; done
python -m h4d.cli export
python -m h4d.cli dashboard
echo "DONE → $H4D_OUT_DIR"
