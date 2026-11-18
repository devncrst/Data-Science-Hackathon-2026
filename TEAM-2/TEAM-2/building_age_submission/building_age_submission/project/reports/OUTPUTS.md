# Outputs index (read these instead of re-running commands)

Final submitted system: Amsterdam average 0.669, n=25 0.660, n=200 0.716, Madrid 0.680, zero-shot 0.526.
Final models: /shared/collab/final/models (do not modify).

| Question | File |
|---|---|
| Everything in one page (baseline vs ours, model families, diagnostics, checks) | tables/MODEL_PERFORMANCE_SUMMARY.txt |
| Official scores per n | tables/results_table.csv |
| Baseline vs ours | tables/baseline_vs_ours.csv |
| Why LightGBM (5 model families incl. neural network) | tables/model_comparison.csv |
| Per-class precision/recall/F1, kappa, ROC-AUC | tables/evaluation_metrics.csv |
| Organisers' prototype method vs ours, all n | tables/prototype_vs_ours.csv |
| Are scores honest? (spatial CV + buffer) | tables/spatial_cv_report.csv, tables/spatial_block_cv.csv |
| Does the model know when it is wrong? | tables/spatial_uncertainty_summary.txt |
| Confusion matrices | figures/confusion_matrices.png |
| Overfitting check | figures/training_curve.png |
| Uncertainty and right/wrong maps | figures/spatial_uncertainty_maps_n25.png |
| F1 vs number of labels | figures/f1_vs_log2n_slide.png |
| Label-weighting test (rejected) | logs/weighted_*.log |
| Full printed run of the evaluation (8 PASS) | logs/full_evaluation_run.log |
| Every idea tested, its score and the decision | ../FEATURES.md |

Re-run everything: /shared/venv/bin/python src/full_evaluation.py (read-only; reproduces all official numbers).
