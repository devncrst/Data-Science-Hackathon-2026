# Building age classification: Madrid -> Amsterdam transfer

Predict the construction-age class (4 classes) of 30 m building pixels from 40 years of Landsat data.
Train on Madrid (Stage 1), adapt to Amsterdam with n labelled pixels per class (Stage 2). Metric: macro-F1.
Classes: Madrid before 1960 / Amsterdam before 1945, then up to 1984, 1984-2004, 2004-2024
(a year exactly on a boundary belongs to the older class, as in the organisers' notebooks).

## Method (short)
1. Features: notebook 3 gives 60 temporal features per pixel; save_features.py adds the mean of the
   8 neighbouring pixels (120 features). Each city is standardised on its own data, without labels.
   Full list with a reason per feature: FEATURES.md.
2. Madrid model: LightGBM (400 trees, lr 0.05, 63 leaves, class_weight balanced, seed 42).
3. Amsterdam adaptation (n labels per class, drawn at random per class, as in notebook 4):
   average of the class probabilities of two adapters:
   - fine-tuning: 50 extra trees grown on the Madrid model with the Amsterdam labels;
   - logistic regression on Amsterdam features + Madrid model probabilities.

## How to reproduce (one command per step)
    # 0. raw data -> preprocessed_data.pkl (organisers' notebook 3, run once)
    #    3-Preprocessing.ipynb, reading data/madrid_train.parquet and data/amsterdam_data.parquet
    # 1. features: spatial = with neighbours (120), plain = without (60)
    /shared/venv/bin/python save_features.py
    # 2. full pipeline: Madrid model, Amsterdam models for n = 5, 25, 50, 100, 200, F1 table, plot
    /shared/venv/bin/python run_pipeline.py --features spatial --out /shared/collab/final
    #    one budget only:        --n 25        other seed:  --seed 7       draws per n: --repeats 10
    #    backup, no neighbours:  --features plain --out /shared/collab/final_plain
    # 3. predict new pixels (picks main or backup models automatically from the number of features)
    /shared/venv/bin/python predict.py --city amsterdam --n 25 --features FILE.pkl --out predictions.csv
    # 4. checks behind our claims
    /shared/venv/bin/python spatial_block_cv.py      # unseen 1 km blocks vs random splits
    /shared/venv/bin/python make_maps.py --n 25      # Amsterdam map: predicted vs actual

Fixed seed 42; 10 random label draws per n (seeds 42-51); n_jobs = 4.
Re-running step 2 gives exactly the same F1 table.
On another machine, change the paths at the top of the scripts (/shared/collab/...).

## Files
- run_pipeline.py: final pipeline (Stage 1 + Stage 2, n as a parameter, fixed seed)
- predict.py: apply saved models to new pixels
- features.py, save_features.py, quick_test.py: feature building and checks
- model_lead.py, transfer_compare.py, finetune_study.py: experiments behind our choices
- spatial_block_cv.py, make_maps.py: spatial-leakage test and map
- eda_plot.py: class and year distributions
- FEATURES.md: feature table; JUSTIFICATION.md: 300-word written justification

## Outputs (/shared/collab/final)
- models/madrid_model.pkl (Stage 1), models/amsterdam_n{5,25,50,100,200}.pkl (Stage 2)
- results_table.csv (mean +- std macro-F1 per n), f1_vs_log2n.png, madrid_confusion_matrix.csv, summary.json

## Main results (macro-F1, mean +- std over 10 draws)
| Test | Baseline | Ours |
|---|---|---|
| Madrid, 5-fold CV | 0.618 | 0.680 +- 0.007 |
| Amsterdam, n = 5 | 0.544 | 0.589 +- 0.030 |
| Amsterdam, n = 25 | 0.599 | 0.660 +- 0.018 |
| Amsterdam, n = 50 | 0.608 | 0.678 +- 0.009 |
| Amsterdam, n = 100 | 0.613 | 0.701 +- 0.007 |
| Amsterdam, n = 200 | 0.615 | 0.716 +- 0.003 |
| Amsterdam average | 0.596 | 0.669 |

Backup without neighbours: 0.656 average. On unseen 1 km blocks: Amsterdam 0.664 -> 0.655 (no drop at n = 25).

## Full evaluation and outputs

- Run the full evaluation (read-only; reproduces all 8 official numbers, then adds model families, per-class metrics, kappa, ROC-AUC, confusion matrices and the training curve):
  /shared/venv/bin/python full_evaluation.py
- Every saved output (tables, figures, logs), with what each one answers: project/reports/OUTPUTS.md
- Every idea we tested, its score and the decision: FEATURES.md

## Run on another computer

1. Python 3.12, then: pip install -r requirements.txt  (exact versions; the .pkl models need them)
2. From the src/ folder, predict with a saved model, pointing --models at the model folder:
   python predict.py --city amsterdam --n 25 --features <features file> --models ../models/main
   (use ../models/backup for the 60-feature backup models)
3. Features are built from the raw data with save_features.py (see above).
