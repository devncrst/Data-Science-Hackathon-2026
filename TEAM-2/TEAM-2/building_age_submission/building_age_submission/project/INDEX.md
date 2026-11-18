# Project map
- README.md          how to reproduce everything (one command per step)
- FEATURES.md        the 120 features, with a reason for each
- JUSTIFICATION.md   300-word written justification
- data/raw/          organisers' raw data (link, never modified)
- data/processed/    our feature files: spatial (120, main) and plain (60, backup)
- notebooks/         organisers' notebooks 1-4 (notebook 3 builds the features)
- src/               all code; run_pipeline.py = full pipeline (n as a parameter, seed 42)
- models/main/       FINAL: madrid_model.pkl (Stage 1) + amsterdam_n{5,25,50,100,200}.pkl (Stage 2)
- models/backup/     same, without neighbour features
- reports/figures/   F1 vs log2(n), Amsterdam map, EDA and correlation plots
- reports/tables/    F1 tables (main and backup), confusion matrix, block test, experiment log
Results: Amsterdam macro-F1 0.669 average (baseline 0.596), 0.660 at n = 25, 0.716 at n = 200.

## Results and evaluation (updated)

- Every output (tables, figures, logs) and what it answers: reports/OUTPUTS.md
- Every idea tested, its score and the decision: FEATURES.md
- Full evaluation script (reproduces all official numbers): src/full_evaluation.py
- Submitted version: Git tag v1.0-final (commit ed562ee)
