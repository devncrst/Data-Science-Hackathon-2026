# Building Age Classification: Madrid to Amsterdam transfer

## Getting it running from GitHub
1. `pip install -r requirements.txt`
2. `python reassemble_models.py`  (rebuilds `models/` from `models_split/`; checksum-verified)
3. Put the organisers' two parquet files in `data/`
4. Follow `START_HERE.md`: section B predicts with the saved models (no retraining); section C retrains everything.

## Which notebook is which
| Notebook | What it does |
|---|---|
| `00-MAIN-Pipeline-v2.ipynb` (top level) | The official final run, outputs saved. Running it retrains everything |
| `PREDICT_with_saved_models.ipynb` (top level) | Predict any new file with the saved models, no retraining |
| `experiments/00-MAIN-Pipeline-FINAL.ipynb` | The earlier v1 pipeline, kept for history |
| `experiments/01-Load-and-CloudMask.ipynb` | Loading the raw data and cloud/haze cleaning |
| `experiments/02-Feature-Table-v3.ipynb` | Building the 132 per-pixel features |
| `experiments/03-Stage1-Stage2-Results.ipynb` | Stage 1 (Madrid) and Stage 2 (Amsterdam transfer) results, v1 |
| `experiments/04-Robustness-Tests.ipynb` | Reverse direction (Amsterdam to Madrid) and stress tests |
| `experiments/05-Dress-Rehearsal.ipynb` | Whole neighbourhoods hidden, full pipeline retrained, hidden pixels scored |
| `experiments/06-Uncertainty-Maps.ipynb` | Uncertainty maps and confidence vs accuracy |
| `experiments/07-Self-Training-Experiment.ipynb` | Self-training with pseudo-labels (tested, not adopted) |
| `experiments/08-Improvement-Experiments.ipynb` | 7 ideas against a rule fixed in advance; 5x5 smoothing adopted |
| `experiments/09-Neighbourhood-Features.ipynb` | Neighbourhood features; adopted and confirmed on hidden neighbourhoods |

The experiments are evidence; they are not needed to use the models. `experiments/PROJECT_LOG.md` records every decision and number,
and `results/README.md` says which result file is final (v2) and which is history (v1).


NCRST Regional Data Science Hackathon 2026. Team: Silvio Ivanio, Antonio Palanca, Anselmo, Fenny Andimba, Jian Cloete.

**Task:** classify each 30 m Landsat pixel into building age class 1 to 4. Train on Madrid (Stage 1), then adapt to Amsterdam with only n labelled pixels per class (Stage 2), for n = 5, 25, 50, 100, 200. Metric: macro F1.

**Final method v2 (Wed 18 Nov, 01:00):**
temporal haze cleaning, then 132 time-series features, then **neighbourhood features** (average features of the surrounding pixels in a 3x3 and a 7x7 window; satellite features only, never labels; 398 inputs in total), then a Random Forest trained on Madrid, then Amsterdam scaled on its own unlabelled features, then blend3 transfer (fine-tuned forest + stacked logistic regression + class prototypes, averaged), then 5x5 spatial smoothing of the class probabilities (ON by default). The previous version (v1, 132 inputs) is kept in `submission_frozen_v1/`.

---

## 0. Start here

See `START_HERE.md`: (A) look at the saved results, nothing to run; (B) score new data with the saved models via `PREDICT_with_saved_models.ipynb`, no retraining; (C) retrain everything with `00-MAIN-Pipeline-v2.ipynb`. Every experiment notebook, with outputs, is in `experiments/`; every results file is in `results/`. The organisers confirmed there is no size limit on the models (about 1.7 GB).

## 1. Quick start (judges)

Requirements: Python 3.12 with numpy, pandas, scipy, scikit-learn, pyarrow, joblib, matplotlib (all on the hackathon VM). No GPU, no extra installs.

```text
Building_Age_Classification/
├── 00-MAIN-Pipeline-v2.ipynb      <- runs EVERYTHING end to end (Restart Kernel and Run All, about 1 h 20 min)
├── 00-MAIN-Pipeline-FINAL.ipynb   <- v1 pipeline (kept for reference)
├── data_prep.py                   <- loading + cloud/haze masking
├── features_v3.py                 <- the 132 features (same function for every city)
├── transfer_v3.py                 <- Stage 1, Stage 2 (blend3), smoothing, evaluation
├── predict.py                     <- apply the saved models to NEW raw parquet files
├── models/
│   ├── stage1_madrid_model.joblib             <- Stage 1 model (deliverable 1)
│   └── amsterdam_blend3_n{005,025,050,100,200}_seed42.joblib
├── results/                       <- tables and plots used in the slides
└── PROJECT_LOG.md                 <- every decision, test and number, with times
```

The raw files are expected in a `data/` folder next to the notebook: `data/madrid_train.parquet` and `data/amsterdam_data.parquet` (edit `RAW` in the third cell if they are elsewhere). Random seed: **42** everywhere.

## 2. Predict new pixels (the held-out test sets)

The test file can be a raw parquet in the same wide format as the files we received. Labels are not needed. If `weighted_mean_year` is present, macro F1 is printed for checking.

```python
import predict

# Madrid held-out pixels -> Stage 1 model
mad = predict.predict_city("madrid_test.parquet", city="Madrid",
                           stage1_path="models/stage1_madrid_model.joblib",
                           smooth=True, context_features="features_madrid_v3.parquet")

# Amsterdam held-out pixels -> the Amsterdam model for a chosen n (here n = 25)
ams = predict.predict_city("amsterdam_test.parquet", city="Amsterdam",
                           stage1_path="models/stage1_madrid_model.joblib",
                           amsterdam_model_path="models/amsterdam_blend3_n025_seed42.joblib",
                           smooth=True, context_features="features_amsterdam_v3.parquet")
ams.to_csv("predictions_amsterdam_n025.csv", index=False)
```

Command line, without smoothing: `python predict.py amsterdam_test.parquet Amsterdam models/amsterdam_blend3_n025_seed42.joblib`

What `predict_city` does:
- It applies the same cleaning and the same features as in training, then adds the neighbourhood features. Neighbours are the test pixels plus our own pixels of that city (`features_<city>_v3.parquet`, used by default), so scattered test pixels still have neighbours. The setting is read from the saved Stage 1 model.
- Every input pixel gets exactly one prediction. Pixels whose photos all fail our cleaning fall back to the organisers' `qa_valid` rule. A pixel with no usable photo at all gets a flagged default class.
- Smoothing is ON by default (final method: 5x5 window, strength 0.5). It mixes each pixel's class probabilities with its neighbours' (only predictions, never labels). `context_features` lets scattered test pixels use our own pixels as neighbours. Pass `smooth=False` for the plain model.

## 3. Train an Amsterdam model for any n (deliverable 2)

```python
import joblib, transfer_v3 as tv3

N, SEED = 25, 42                          # any n per class, any seed
data   = tv3.load_data(".")               # v3 features of both cities
stage1 = tv3.train_stage1(data)           # Random Forest on all Madrid pixels
f1, model = tv3.adapt_to_amsterdam(data, stage1, N, SEED, method="blend3")
print(f"macro F1 on the other Amsterdam pixels: {f1:.4f}")
joblib.dump(model, f"models/amsterdam_blend3_n{N:03d}_seed{SEED}.joblib")
```

`tv3.evaluate(data, stage1, "blend3", shots=(5, 25, 50, 100, 200), trials=10)` gives mean and std F1 over 10 random support samples (seeds 42 to 51).

## 4. Results (official run, 00-MAIN-Pipeline-v2)

| Setting | Macro F1 (mean +/- std) | Organisers' baseline |
|---|---|---|
| Madrid, random 5-fold x 5-repeat CV | 0.733 +/- 0.003 (optimistic: neighbouring pixels share neighbourhood features) | 0.618 |
| Madrid, spatial CV (1.5 km blocks held out) | **0.629 +/- 0.034** (realistic) | |
| Amsterdam n = 5 | 0.617 +/- 0.028 | 0.544 |
| Amsterdam n = 25 | 0.675 +/- 0.014 | 0.599 |
| Amsterdam n = 50 | 0.701 +/- 0.010 | 0.608 |
| Amsterdam n = 100 | 0.718 +/- 0.004 | 0.613 |
| Amsterdam n = 200 | 0.733 +/- 0.004 | 0.615 |
| **Amsterdam average** | **0.689** | 0.596 |

Amsterdam scores are for the final method (with 5x5 smoothing); `results_table_v2.csv` also lists them without smoothing (average 0.681). Plot: `f1_vs_log2n_v2.png`. Madrid confusion proportions: `results/madrid_confusion_proportions_v2_spatial.csv`. The v1 results (132 inputs, Amsterdam average 0.655) are in `submission_frozen_v1/`.

## 5. Honest checks

- **Spatial CV on Madrid** (whole 1.5 km blocks held out): v1 0.616 vs 0.657 random; v2 0.629 vs 0.733 random. Neighbouring pixels are near copies, so random CV is optimistic, more so with neighbourhood features.
- **Dress rehearsal** (`05-Dress-Rehearsal.ipynb`): 20% of each city is hidden as whole neighbourhoods from the raw files, the full pipeline is retrained on the rest, and the hidden pixels are scored with `predict.py`. On unseen neighbourhoods (with smoothing), Madrid scores 0.609 and the Amsterdam average 0.601 (n=25: 0.587, n=200: 0.644). Smoothing helps in 48 of 50 runs, so it is ON in the submission. Notebook 08 then tested 7 more ideas with a rule fixed in advance: only the wider 5x5 smoothing passed, and it was confirmed on the hidden neighbourhoods (Amsterdam 0.601 to 0.607, Madrid 0.609 to 0.617). Notebook 09 then added the neighbourhood features, confirmed the same way on the same hidden neighbourhoods: the hidden Amsterdam average rose from 0.607 to **0.617**, which is the hidden-neighbourhood figure on the deck. Trade-off, stated openly: Madrid on hidden areas ends at 0.606 (0.609 before the 5x5 smoothing, 0.617 after it). We adopted the upgrade because the rule fixed in advance judges Amsterdam, the scored city; Madrid is not a scored metric, and our realistic Madrid figure on the deck is the spatial CV score of 0.629. See PROJECT_LOG section 7h.
- **Uncertainty:** with the final models, the 20% of pixels the model is most sure about are 86% correct (Amsterdam, n=25), against 43% for the least sure 20% (`06-Uncertainty-Maps.ipynb`).
- **Tried and not adopted:** self-training with pseudo-labels (worse at n=25 in 9 of 10 seeds, `07-Self-Training-Experiment.ipynb`).
- **Reverse direction** (Amsterdam to Madrid) and **stress test** (darker, brighter, noisier, cloudier city): `04-Robustness-Tests.ipynb`, PROJECT_LOG section 7g.
- **Model comparison:** six model families on identical splits (PROJECT_LOG section 7e). Random Forest was chosen for speed; boosting was within noise.

## 5b. Reproducibility

A clean-room run (empty folder, only the code files + raw data, fresh kernel, `jupyter nbconvert --execute`) rebuilt every file and matched the official results to 3 decimals (multi-threaded forests can differ in the 4th decimal). Exact package versions: `requirements.txt`.

## 6. No data leakage

- These columns are **never** used as features: `weighted_mean_year`, `age_class`, `coverage`, pixel keys and coordinates, and observation counts (see `features_v3.FORBIDDEN`).
- Amsterdam labels are used **only** for the n support pixels per class. All other Amsterdam pixels are used for scoring only.
- Amsterdam scaling uses Amsterdam **features** without labels (label-free city calibration). Tested on the final method: own-city scaling 0.689 vs Madrid scaling 0.682 on average, better at every n (n=5: 0.617 vs 0.600; n=25: 0.675 vs 0.668; n=200: 0.733 vs 0.731; `results/scaling_comparison_v2.csv`). Backup if unlabelled data were not allowed: set `transfer_v3.SCALING = "madrid"`.
- Every learned step is fitted inside the training fold. Smoothing uses predictions only, never labels.

## 7. Notebook order (for reproducing every step)

| Notebook | What it does |
|---|---|
| 00-MAIN-Pipeline-v2 | the whole submission pipeline, end to end |
| 00-MAIN-Pipeline-FINAL | v1 pipeline (Tuesday) |
| 01-Load-and-CloudMask | cleaning + EDA (clean/clean_*.parquet) |
| 02-Feature-Table-v3 | features + fair comparison of feature sets |
| 03-Stage1-Stage2-Results | Madrid CV, transfer methods, model comparison, smoothing |
| 04-Robustness-Tests | reverse direction + stress test |
| 05-Dress-Rehearsal | hidden-neighbourhood test with predict.py |
| 06-Uncertainty-Maps | where the model is unsure (maps + confidence vs accuracy) |
| 07-Self-Training-Experiment | pseudo-label experiment with a pre-set accept rule |
| 08-Improvement-Experiments | 7 ideas screened with a fixed rule (5x5 smoothing adopted) |
| 09-Neighbourhood-Features | neighbourhood features screened and confirmed on hidden areas (adopted) |
