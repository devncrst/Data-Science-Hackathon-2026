#!/bin/bash
# scripts/ilifu_job.sbatch — adjust cpus/mem from your benchmark
#SBATCH --job-name=h4d-baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/h4d-%j.out
set -euo pipefail
module load python/3.12.3
source ~/venvs/h4d/bin/activate
export H4D_DATA_DIR=/path/given/at/venue        # MONDAY
export H4D_OUT_DIR=$HOME/h4d_outputs
cd ~/hack4dev2026 && bash scripts/run_all.sh
