# Building Age Classification: Project Log

**Event:** NCRST Regional Data Science Hackathon 2026, National AI Institute, Windhoek (16 to 18 Nov 2026)
**Maintained by:** Jian (EDA and presentation) with Claude
**Last updated:** Tue 17 Nov, 11:50

This file is the single source of truth for what we did, why, and where the files are.
Read it before you start work, and tell Jian when you finish a step so it gets logged here.

---

## 1. The task in one paragraph

Each 30 m Landsat pixel has a yearly spectral time series (1984 to 2025, 6 bands). We predict the
pixel's building **age class 1 to 4**. We train on **Madrid**, then adapt to **Amsterdam** using only
**n = 5, 25, 50, 100, 200 labelled pixels per class**. Metric: **macro F1** (average F1 over the 4
classes). Prizes: best average F1, best peak F1, best low-data F1 (n = 25 only), best originality.
Deliverable freeze: **Wed 18 Nov, 11:30 sharp**.

| Class | Madrid | Amsterdam |
|---|---|---|
| 1 | up to 1960 | up to 1945 |
| 2 | 1961 to 1984 | 1946 to 1984 |
| 3 | 1985 to 2004 | 1985 to 2004 |
| 4 | 2005 onwards | 2005 onwards |

Bins are right-closed (`pd.cut` default): a building from exactly 1984 is class 2.

## 2. Team

| Role | Person | Owns |
|---|---|---|
| Model lead | Silvio Ivanio | The score: Madrid model and Amsterdam adaptation |
| Modelling support | Antonio Palanca | Baselines and comparisons, checks that gains hold in Amsterdam |
| Pipeline | Anselmo | One runnable notebook end to end, Git, shared folder |
| Features and satellite domain | Fenny Andimba | What the model is allowed to see |
| EDA and presentation | Jian Cloete | Data checks, this log, experiment log, slides, written justification |

## 3. The flow (pipeline so far)

```
 data/madrid_train.parquet          data/amsterdam_data.parquet
 (raw: 1 row per pixel-year, up to 3 photos per row, quality flags)
              |                                 |
              v                                 v
 [STEP 1] 01-Load-and-CloudMask.ipynb  +  data_prep.py           DONE (Jian), rerun Tue
          load, cloud mask per photo, TEMPORAL haze rule, median of clean photos
              |
              v
 clean/clean_madrid.parquet      clean/clean_amsterdam.parquet
 (1 row per pixel-year, 6 clean band values)
              |
              v
 [STEP 2] 02-Feature-Table-v3.ipynb + features_v3.py             DONE (Fenny), frozen v3_all
          1 row per pixel, 132 time-series features
              |
              v
 features_madrid_v3.parquet   features_amsterdam_v3.parquet   feature_cols_v3.json
              |
              v
 [STEP 3] Stage 1: Madrid RF + 5x5 CV (transfer_v3.py)             NEXT (Silvio)
              |
              v
 [STEP 4] Stage 2: blend stack+prototype, n = 5..200             NEXT (Antonio)
              |
              v
 [STEP 5] Improvements, one at a time, logged                     Day 2
              |
              v
 [STEP 6] Final run, results table, F1 vs log2(n) plot, slides    Day 2 evening / Day 3
```

## 4. Files (all in `Building_Age_Classification/`, checked Tue 09:45)

**Active pipeline (used for the final run)**

| File | What it is | Made by |
|---|---|---|
| `data/` | Raw parquet files (links to the real data) | Organisers |
| `data_prep.py` | Loading + cloud masking. Default haze rule = **temporal** (since Tue 09:35) | Jian + Claude |
| `01-Load-and-CloudMask.ipynb` | Step 1: load, mask, clean, EDA checks. Rerun Tue 09:39 with the temporal rule (removed 0.37% Madrid / 0.68% Amsterdam photos) | Jian + Claude |
| `clean/clean_*.parquet` | Step 1 output: clean pixel-year tables (Madrid 3,161,994 rows, 99.7%) | Step 1 |
| `clean/*.png`, `clean/class_counts.csv` | EDA plots and tables for the slides | Step 1 |
| `features_v3.py` | Feature code v3 (132 features, gap-proof change features) | Fenny + Jian + Claude |
| `02-Feature-Table-v3.ipynb` | Step 2: builds v3, label-free shift check, fair comparison of v1/v2/v3 | Fenny + Jian + Claude |
| `features_*_v3.parquet`, `feature_cols_v3.json`, `feature_sets_v3.json` | **Final features** (winner: v3_all) | Step 2 |
| `feature_comparison_v3.csv`, `feature_shift_v3.csv` | Evidence tables for the slides | Step 2 |
| `transfer_v3.py` | Stage 1 (Madrid RF) + stage 2 `adapt_to_amsterdam(n, seed, method)`: prototype, stack, stack_cv, blend, finetune, blend3 | Jian + Claude |
| `PROJECT_LOG.md` | This file. **Do not keep it open in a JupyterLab tab while Claude updates it** (the open tab saves an old copy over it) | Jian + Claude |
| `Original/`, `METRIC_LEARNING_APPROACH.md` | Organisers' reference notebooks and notes (untouched) | Organisers |

**Kept as evidence (read by the v3 comparison in section 3 of 02-Feature-Table-v3; do NOT delete)**

| File | Role |
|---|---|
| `features_madrid.parquet`, `features_amsterdam.parquet`, `feature_cols.json` | v1 (30 features), first row of the comparison |
| `features_madrid_v2.parquet`, `features_amsterdam_v2.parquet` | v2 (46 features), second row of the comparison |

**`archive/`: superseded files, kept for history only**

| File | Why archived |
|---|---|
| `02-Feature-Table.ipynb` | v1 notebook, replaced by v3 |
| `02-Feature-Table-v2.ipynb` | v2 notebook; cannot Run All (Amsterdam load cell was markdown); replaced by v3 |
| `feature_cols_final.json` | v2 keep/drop list. **NEVER use it**: it was chosen using all Amsterdam labels |

**Shared team folder:** `shared/collab/` is Anselmo's workspace (main notebook `00-MAIN-Pipeline.ipynb`, `models/`). Do not edit without asking him.

**Note:** after the step 1 rerun, step 2 (`02-Feature-Table-v3.ipynb`) must also be rerun so the v3 feature files use the temporal-cleaned data.

## 5. Rules everyone follows (no data leakage)

1. **Never use as features:** `weighted_mean_year` (it is the label), `age_class`, `coverage`
   (its link to age flips between cities), `px_key`, `py_key`, `pixel_x`, `pixel_y`, `pixel_id`
   (location lets the model memorise Madrid's map).
2. Anything that learns from data (scaler, feature selection, model) is fitted **only on training
   data**, inside each cross-validation fold (use a sklearn `Pipeline`).
3. Amsterdam: only the n support pixels' labels may be used. Query pixels' labels are used only for scoring.
4. Choose features and settings on **Madrid CV**, never by looking at Amsterdam test scores.
5. Fixed seed **42** everywhere.
6. Change **one thing at a time** and log the score (section 9).
7. Environment: scikit-learn, pandas, numpy, scipy only. No PyTorch, LightGBM or XGBoost on the VM.

## 6. Step 1: Data loading and cloud masking (DONE)

**What it does**
- Loads both cities (Madrid 3,172,765 rows / 76,263 pixels; Amsterdam 1,062,877 rows / 25,992 pixels;
  years 1984 to 2025). Matches the official notebooks exactly.
- Checks every photo (up to 3 per pixel per summer) against these rules: organisers' `qa_valid`;
  QA_PIXEL bits for fill, dilated cloud, cirrus, cloud, cloud shadow, snow, cloud confidence;
  QA_RADSAT saturation; valid value range 7273 to 43636; haze.
- Combines the clean photos of each year with the **median** (one bad photo is outvoted).

**Decisions and why**

| Decision | Why | Evidence |
|---|---|---|
| Decode QA_PIXEL bits ourselves | `qa_valid` ignores cirrus and missed haze (Notebook 2 found Blue up to 64,126) | USGS Collection 2 QA bit table |
| Median instead of first photo | One leftover hazy photo should not decide the value | Standard compositing practice |
| **Scene-level haze detection** instead of "Blue > 15,000" | The old pixel rule deleted bright new buildings as "haze" | Old rule removed 0.21% of class 1 vs **2.51%** of class 4 in Madrid (12x). New rule: 0.00 / 0.01 / 0.05 / 0.13% Madrid, 9.9 / 10.3 / 10.4 / 10.3% Amsterdam (equal across classes) |
| **Temporal haze rule** replaces scene-level (Tue) | Amsterdam haze is patchy; scene rule missed it and cost 0.016 F1 at n=25. Temporal = drop a photo whose Blue > 1.35x the pixel's own median Blue over all years (haze is temporary, roofs are permanent) | See section 7c, tests 2 and 3 |
| Keep the valid-range rule | Its "too low" values are all 6 bands at once = empty fill photos, same ones `not_fill` catches | Madrid 1.85%, Amsterdam 19.5% in every band equally |

**Results**
- Hazy scenes found: Madrid 1, Amsterdam 10.
- Photos kept: Madrid 94 / 95 / 89% (obs 1/2/3), Amsterdam 61 / 75 / 71% (cloudier city).
- Clean pixel-years: Madrid 99.9%, Amsterdam 98.8%.
- Clean photos per pixel-year: Madrid mostly 3; Amsterdam often 1 or 2 (noisier).
- `usable_label` is True for every row, so no pixels need dropping.

**EDA findings (for slides and modellers)**

| Class | Madrid | Amsterdam |
|---|---|---|
| 1 | 13,794 (18.1%) | 7,568 (29.1%) |
| 2 | 27,152 (35.6%) | 8,802 (33.9%) |
| 3 | 18,672 (24.5%) | 6,745 (26.0%) |
| 4 | 16,645 (21.8%) | 2,877 (11.1%) |

- Amsterdam has half the share of class 4 that Madrid has: the transfer step must correct for this.
- 6.5% of Madrid pixels (2.9% Amsterdam) sit exactly on a boundary year, many at 1960. Expect most
  confusion between classes 1 and 2.
- Mean coverage by class goes in opposite directions in the two cities (Madrid: oldest densest;
  Amsterdam: newest densest), so `coverage` is excluded as a feature.

## 7. Step 2: Feature table (DONE, reviewed)

**What it does:** one row per pixel, same function for both cities. Per band: fills missing years
by linear interpolation, then computes 5 features. 6 bands x 5 = **30 features**.

| Feature | Meaning | Why it may help |
|---|---|---|
| `{band}_mean` | Average over 1984 to 2025 | Overall material brightness |
| `{band}_std` | Variation over the years | Newer buildings vary more (Notebook 2) |
| `{band}_early_mean` | Average 1984 to 2003 | What stood there before 2004 |
| `{band}_late_mean` | Average 2004 onwards | The current building |
| `{band}_diff_std` | Spread of year-on-year changes | Separates pre-1984 (stable) from built-during-record (jumpy) |

**Review (Jian + Claude, 16:30)**
- Correct: 76,263 and 25,992 pixels, class counts match step 1, 0 missing values, seed fixed,
  one class per pixel checked with an assert, same function for both cities.
- No leakage: `feature_cols.json` holds only band statistics. `px_key`, `py_key` are IDs and
  `age_class` is the label, stored but not in the feature list.
- Notes for later, not blocking:
  1. This is 30 features, fewer than the official baseline's 60 (no spectral indices, no early/late
     std, no yoy mean). So a lower score than 0.618 at first is not a failure of the step.
  2. `diff_std` is computed after interpolation. Amsterdam has more gap years, and interpolated
     straight lines make changes look smoother, so this may be slightly lower in Amsterdam for
     reasons that are not age. Option: compute changes on observed years only.
  3. Indices (NDVI etc.) wait on the organisers' answer about reflectance scaling.
  4. `find()` takes the first matching file. If copies of the clean files appear in other folders
     (e.g. `shared/collab`), make sure the right one is picked.
  5. Delete the 3 empty cells at the end.

## 7b. Step 2 v3: Feature table version 3 (DONE, FROZEN: v3_all)

**Files:** features_v3.py, 02-Feature-Table-v3.ipynb -> features_madrid_v3.parquet, features_amsterdam_v3.parquet, feature_cols_v3.json, feature_sets_v3.json, feature_shift_v3.csv, feature_comparison_v3.csv

**Why v3:** v2's keep/drop list was chosen using ALL Amsterdam labels (inflates our Amsterdam scores and contradicts the few-label story); v2 notebook could not Run All (Amsterdam load cell was markdown); it overwrote v1 files; slope / biggest jump / jump year were missing.

**What v3 is:** 132 features = 11 channels (6 bands in reflectance + NDVI, NDBI, MNDWI, NDSWIR, BSI) x 12 statistics. Level stats (mean, std, early/late mean and std, late minus early) on gap-filled years. Change stats (slope, mean change per year, jumpiness, biggest jump, YEAR of biggest jump) on OBSERVED years only. Jumpiness uses the raw difference between consecutive photos, which tested gap-proof (identical old buildings: 0.0124 low-gap city vs 0.0122 high-gap city; dividing by the gap gave a biased 0.0121 vs 0.0102). Jump year recovered the true build year within 1 year for 99.5% of test pixels.

**Label-free shift check (no Amsterdam labels):** most stable = change features (NIR_late_std, Blue/SWIR1 max_jump, MNDWI change, jumpiness); most shifted = raw brightness levels (Red/Green/SWIR2 means, KS 0.74 to 0.84). Interpretation: WHEN a pixel changed transfers between cities; WHAT it looks like does not. 43 of 132 features have KS <= 0.30.

**Fair comparison (scene-haze cleaning; RF 300 trees, balanced, seed 42, same 5 Madrid folds, same 10 Amsterdam samples, prototype transfer with Madrid scaler):**

| Feature set | n | Madrid F1 | Ams n=5 | Ams n=25 |
|---|---|---|---|---|
| v1 | 30 | 0.610 +/- 0.006 | 0.460 +/- 0.055 | 0.508 +/- 0.021 |
| v2 all | 46 | 0.631 +/- 0.005 | 0.464 +/- 0.046 | 0.531 +/- 0.010 |
| **v3 all (winner)** | 132 | **0.660 +/- 0.006** | 0.465 +/- 0.047 | **0.534 +/- 0.026** |
| v3 stable | 43 | 0.617 +/- 0.007 | 0.473 +/- 0.044 | 0.520 +/- 0.024 |

Decision rule agreed before results: best Ams n=25 among sets within 0.01 of best Madrid; ties -> fewer features. Winner: **v3_all**. Features FROZEN. The stable subset did not help: honest failure for the slides.

## 7c. Diagnosis tests on the real data (Tue 09:10 to 10:20, Jian + Claude)

All tests: v3 features, the same 10 Amsterdam support samples per n (seeds 42 to 51), macro F1 on the query pixels only.

**Test 1: scaling.** Prototype with Amsterdam scaled by its own statistics (no labels) instead of Madrid's: n=5 0.465 -> 0.503, n=25 0.534 -> 0.571. Robust scaling: n=5 0.524, n=25 0.564.

**Test 2: what in our cleaning costs Amsterdam score?** (prototype, own scaler; paired against official cleaning)

| Cleaning variant | n=5 | n=25 | vs official n=25 |
|---|---|---|---|
| Official (first clean photo, Blue <= 15,000) | 0.504 | 0.586 | 0 |
| Ours (median, scene haze, all QA rules) | 0.503 | 0.571 | -0.016 |
| Ours but first photo | 0.497 | 0.574 | -0.013 |
| Ours, no dilated/conf rules | 0.506 | 0.571 | -0.015 |
| Ours, scene haze k=8 / k=16 / none | 0.514 | 0.576 | -0.011 |
| **Ours with pixel haze rule** | 0.520 | 0.586 | -0.001 |

Cause: the haze rule. Amsterdam haze is PATCHY; whole-scene detection misses it.

**Test 3: pixel vs TEMPORAL haze rule, scored on BOTH cities**

| Haze rule | Madrid CV | Ams proto n=5 | n=25 | Photos removed |
|---|---|---|---|---|
| Scene (old) | 0.660 +/- 0.006 | 0.503 | 0.571 | ~10% Amsterdam |
| Pixel | 0.657 +/- 0.005 | 0.520 | 0.586 | biased to class 4 |
| **Temporal (ADOPTED)** | 0.658 +/- 0.005 | **0.523** | 0.584 | **0.37% Madrid, 0.69% Amsterdam** |

Synthetic check: temporal rule removed 98% of injected temporary haze, kept 100% of permanently bright roofs. Now the default in data_prep.py.

**Test 4: transfer methods on temporal-cleaned v3** (Madrid RF 300 trees, min_samples_leaf=2, Amsterdam own scaler)

| Method | n=5 | n=25 | n=50 | n=100 | n=200 | avg of 5 |
|---|---|---|---|---|---|---|
| Official reference (prototype) | 0.544 | 0.599 | ~0.608 | 0.613 | 0.615 | ~0.596 |
| Zero-shot Madrid model (no Ams labels) | 0.453 | | | | | |
| Prototype, own scaler | 0.523 | 0.584 | 0.600 | 0.605 | 0.617 | 0.586 |
| Stack (LR on log Madrid probs + features) | 0.535 | 0.627 | 0.645 | 0.658 | 0.679 | 0.629 |
| **Blend stack + prototype** | 0.531 | **0.629** | **0.650** | **0.668** | **0.687** | **0.633** |

Earlier tests on old cleaning: recalibrate-only 0.484 / 0.529 (weak), prototype on Madrid probs 0.474 / 0.513 (weak), stack with C chosen by CV inside support 0.508 / 0.600 (no better than the fixed schedule), fine-tune RF (Madrid + weighted support) 0.555 / 0.621 (best at n=5, slow). Pending: 3-way blend (fine-tune + stack + prototype) on temporal data, results in /tmp/transfer_results_temporal.pkl on the VM.

**Open:** using Amsterdam's own scaling uses unlabelled Amsterdam data. Confirm with organisers; if not allowed, set SCALING = "madrid" in transfer_v3.py and re-measure.

## 7d. Final run + validation (Tue 10:30 to 11:45, Jian + Claude)

**Official run: 03-Stage1-Stage2-Results.ipynb** (temporal-cleaned v3 features, Amsterdam own scaling, 10 seeds per n)

| Setting | Ours (blend3) | Official baseline |
|---|---|---|
| Madrid, random 5x5 CV | **0.656 +/- 0.003** | 0.618 |
| Amsterdam n=5 | **0.580 +/- 0.038** | 0.544 |
| Amsterdam n=25 | **0.641 +/- 0.013** | 0.599 |
| Amsterdam n=50 | **0.666 +/- 0.012** | 0.608 |
| Amsterdam n=100 | **0.683 +/- 0.005** | 0.613 |
| Amsterdam n=200 | **0.703 +/- 0.004** | 0.615 |
| Amsterdam average of 5 | **0.655** | 0.596 |
| Zero-shot (no Amsterdam labels) | 0.453 | 0.343 |

Files: results/results_table.csv, results/amsterdam_all_methods.csv, results/madrid_confusion*.{csv,png}, results/f1_vs_log2n.png, models/stage1_madrid_model.joblib, models/amsterdam_blend3_n{005..200}_seed42.joblib (each verified to reproduce its score after reloading).

Madrid confusion (row = true class): class 1 -> 55% correct, 37% predicted as class 2 (many records exactly at 1960); errors almost always fall in a NEIGHBOURING age class.

**Validation (results/validation_log.txt)**
* Spatial CV on Madrid (whole 1.5 km blocks held out): **0.616 +/- 0.025** vs random CV 0.657 +/- 0.005. Random CV overstates by ~0.04 because neighbouring pixels are near copies (the official 0.618 is also random CV, so the comparison is fair). Report both.
* Spatial smoothing (each pixel's probabilities mixed 50/50 with its 3x3 neighbours' average; only predictions used, never labels; support pixels excluded as neighbours; strength fixed in advance):
  Madrid spatial CV 0.617 -> 0.632; Amsterdam n=5 0.580 -> 0.593, n=25 0.641 -> 0.658, n=50 0.666 -> 0.684, n=100 0.683 -> 0.702, each better in 10/10 seeds. n=200 and the Madrid-scaling backup are in the log.
* Decision pending: adopt blend3 + smoothing as final (recommended), reporting both columns.

**Robustness added:** predict.py applies the saved models to any new raw parquet (organisers' test set): same cleaning + features, works without labels, safety net so EVERY input pixel gets exactly one prediction, optional smoothing with our transfer pixels as context so scattered test pixels still have neighbours.

## 7e. Pipeline reproduced + model comparison (Tue 13:00 to 15:30, Jian + Claude)

**00-MAIN-Pipeline-FINAL.ipynb** (private folder) ran end to end from raw data with Restart and Run All, and every check passed. It reproduces section 7d: Madrid 0.656; Amsterdam 0.580 / 0.641 / 0.666 / 0.683 / 0.703; average 0.655; **with smoothing, average 0.671** (smoothing won 50 of 50 seed runs). Anselmo's older 00-MAIN used the old cleaning (average 0.641) and old save formats, so the FINAL version replaces it.

**Stage 1 model comparison** (03 notebook, results/stage1_model_comparison.csv; same 132 features, same folds, seed 42):

| Model | Madrid F1 | Time for 5 folds |
|---|---|---|
| HistGradientBoosting | 0.661 +/- 0.004 | 38 min |
| **Random Forest (chosen)** | **0.657 +/- 0.005** | **4 min** |
| Neural network (sklearn MLP) | 0.654 +/- 0.005 | 2 min |
| ExtraTrees | 0.648 +/- 0.005 | 1 min |
| kNN | 0.628 +/- 0.004 | under 1 min |
| Logistic regression | 0.604 +/- 0.005 | 7 min |

Decision: **keep Random Forest.** Boosting's +0.0045 is inside the error bar, it is about 10x slower, and blend3 retrains the model 50 times (about 10 hours to redo everything). XGBoost, LightGBM, CatBoost and PyTorch are not installed on the VM. Our gains came from cleaning, features, transfer and smoothing, not from the model type.

**METHOD FROZEN (Tue 15:30): Random Forest + v3 features + temporal haze rule + Amsterdam own scaling + blend3 + 3x3 smoothing.**

Still to run (optional, in this order): reverse direction test (train Amsterdam, adapt to Madrid); different conditions stress test (for example darkened or noisier Amsterdam features). Results go in section 9.

## 7g. Robustness tests (Tue 16:00 to 19:00, 04-Robustness-Tests.ipynb, Jian + Claude)

**In-city ceilings** (RF trained and tested inside one city, all labels, 5-fold): Madrid **0.657**, Amsterdam **0.775**.

**Test A, reverse direction (Amsterdam -> Madrid), 10 seeds** (results/reverse_direction.csv, results/reverse_vs_forward.png)

| n | Forward Mad->Ams | % of Ams ceiling | Reverse Ams->Mad | % of Mad ceiling |
|---|---|---|---|---|
| 5 | 0.580 | 75% | 0.424 +/- 0.027 | 65% |
| 25 | 0.641 | 83% | 0.512 +/- 0.014 | 78% |
| 50 | 0.666 | 86% | 0.518 +/- 0.012 | 79% |
| 100 | 0.683 | 88% | 0.540 +/- 0.007 | 82% |
| 200 | 0.703 | 91% | 0.555 +/- 0.005 | 85% |

Zero-shot: forward 0.453, reverse 0.336. Reverse average 0.510 (smoothed 0.523); smoothing better in 50/50 reverse runs.
Meaning: the method works both ways; reverse is weaker because Amsterdam is a smaller source (26k vs 76k pixels) and Madrid is the harder city.

**Test B, different conditions on Amsterdam, blend3, 5 seeds** (results/stress_test.csv)

| Condition | Zero-shot | Drop n=25 | Drop n=200 | Own minus Madrid scaling n=5 / n=25 |
|---|---|---|---|---|
| baseline | 0.453 | | | +0.019 / +0.009 |
| 15% darker | 0.466 | -0.003 | -0.001 | +0.023 / +0.012 |
| 15% brighter | 0.419 | 0.000 | -0.001 | +0.014 / +0.006 |
| noise (sd 0.01) | 0.372 | -0.010 | -0.018 | +0.020 / +0.014 |
| 30% fewer clear years | 0.460 | -0.012 | -0.029 | +0.016 / +0.014 |

Meaning: brightness changes are absorbed (ratio indices + own scaling); noise and cloudiness cost 1 to 3 points; the Madrid model alone is fragile, local labels rescue it; own scaling beats Madrid scaling in all 5 conditions at low n (ties at n=200).

**Brief check (Original notebooks, Tue 19:10):** the organisers test on **held-out Madrid and Amsterdam pixels only**. Class bins match their pd.cut exactly. Risks for the hidden test: spatial (random CV 0.657 vs spatial 0.616) and mild tuning optimism, both measured by the dress rehearsal (section 7h). Other-city limitations (southern hemisphere winter photos, dry cities, single city pair, small source city) are STATED on the limitations slide, not implemented.

**Plan (Tue 19:50):** 1 log (done), 2 dress rehearsal, 3 README, 4 uncertainty map slide, 5 self-training experiment only if time allows (adopt only if n=25 better in at least 8/10 seeds, no other n worse, holds in the dress rehearsal), 6 organiser questions Wed morning.

## 7h. Dress rehearsal, uncertainty maps, self-training (Tue 20:00 to 21:00, notebooks 05 to 07, all checks passed)

**05 Dress rehearsal** (rehearsal/): 20% of 1.5 km blocks hidden per city from the RAW files (Madrid 43 of 215 blocks = 14,532 pixels; Amsterdam 22 of 112 blocks = 3,477 pixels), full pipeline retrained on the rest, hidden pixels scored through predict.py. Shortcut scoring = predict.py exactly (0.5953 vs 0.5953). Every hidden pixel got a prediction.

| | Official CV | Inside training area | Hidden neighbourhoods | Hidden + smoothing |
|---|---|---|---|---|
| Madrid Stage 1 | 0.656 (spatial 0.616) | | 0.587 | **0.609** |
| Amsterdam n=5 | 0.580 | 0.576 | 0.511 +/- 0.038 | 0.516 |
| Amsterdam n=25 | 0.641 | 0.646 | 0.574 +/- 0.030 | **0.587** |
| Amsterdam n=50 | 0.666 | 0.677 | 0.602 +/- 0.017 | 0.620 |
| Amsterdam n=100 | 0.683 | 0.693 | 0.622 +/- 0.011 | 0.638 |
| Amsterdam n=200 | 0.703 | 0.711 | 0.628 +/- 0.006 | **0.644** |
| Amsterdam average | 0.655 | | 0.587 | **0.601** |

Meaning: (1) predict.py works end to end on unseen raw data; (2) no tuning inflation (training-area scores match official CV with 20% less data); (3) unseen neighbourhoods cost about 0.06 to 0.07: the model partly learns place-specific patterns; (4) smoothing wins 48/50 hidden runs, Madrid +0.022, so smoothing stays ON; Madrid hidden + smoothing (0.609) is close to spatial CV (0.616), so spatial CV is an honest predictor. Caveat: hidden Amsterdam is only 22 blocks.

**06 Uncertainty** (results/uncertainty_*.png, uncertainty_by_confidence.csv): accuracy of the most confident 20% vs least confident 20%: Amsterdam n=25 **0.84 vs 0.45**, Madrid out-of-fold **0.94 vs 0.44**, falling steadily across 5 groups. The model knows when it is unsure: a new city can trust confident pixels and check only uncertain areas.

**07 Self-training** (results/self_training.csv): DOES NOT PASS the pre-set rule. n=25 better in 1/10 seeds; mean change n=5 -0.013, n=25 -0.004, n=50 -0.002, n=100 +0.000, n=200 -0.002, although 82 to 96% of pseudo-labels were correct (confident mistakes get learned). Not adopted; slide: "tried, did not help".

**Status Tue 21:00: submission method confirmed (RF + blend3 + smoothing ON). Snapshot saved as submission_frozen_v1/ before any further experiment.** Deck moved to Wed early morning.

## 7i. Improvement experiments (Tue 21:30 onwards, notebooks 08 and 09, experiments_v4.py; results pending)

Safe copy first: submission_frozen_v1/. All screening on the dress-rehearsal 80% split with paired seeds; hidden 20% used ONCE for confirmation. Rule fixed in advance: pass if n=25 or 5-n average improves by >= +0.005 and no n worse by > 0.005 (Stage 1 ideas: Madrid random CV within -0.005; A also spatial +0.01). Built-in sanity checks: baseline must reproduce notebook 05 (training area and hidden); both matched on test data.

* **08 RESULT (Tue 23:00):** both sanity checks passed. Screening on the 80% split, smoothed final score, paired 10 seeds:

| Idea | n=5 | n=25 | n=50 | n=100 | n=200 | avg | Pass? |
|---|---|---|---|---|---|---|---|
| D align | +0.017 | -0.001 | -0.005 | -0.005 | -0.004 | +0.001 | no (hurts n>=50) |
| I year | 0.000 | 0.000 | -0.001 | 0.000 | 0.000 | 0.000 | no |
| H weights | +0.003 | -0.001 | -0.001 | 0.000 | 0.000 | 0.000 | no |
| E boost | +0.002 | +0.001 | +0.002 | 0.000 | 0.000 | +0.001 | no (too small) |
| A simpler forest | +0.004 | -0.001 | -0.001 | -0.001 | -0.003 | 0.000 | no (Madrid worse at every setting: leaf 5/10/20 spatial 0.614/0.610/0.605 vs 0.617) |
| C class-mix | -0.179 | -0.030 | -0.013 | -0.008 | -0.006 | -0.047 | no |
| **B smoothing 5x5, 0.5** | +0.004 | **+0.007** | +0.006 | +0.006 | +0.005 | **+0.006** | **YES (10/10 seeds at n=25)** |
| B smoothing 5x5, 0.7 | +0.004 | +0.007 | +0.006 | +0.006 | +0.004 | +0.006 | yes, but 0.5 chosen (higher avg) |
| no smoothing | -0.010 | -0.016 | -0.018 | -0.019 | -0.017 | -0.016 | no |

  Madrid spatial CV + smoothing: 3x3 0.632, 5x5 0.637. **Hidden confirmation:** Amsterdam average 0.601 -> **0.607** (n=25 +0.004, 7/10 seeds; n=200 about 0.644 -> 0.652), Madrid 0.609 -> **0.617**. **ADOPTED: smoothing radius 2 (5x5), strength 0.5** (transfer_v3.SMOOTH_RADIUS = 2; predict.py smooths by default). Models unchanged. Note: the confirmation table in 08 shows one merged "baseline" column (both setups had the same name); the averages and Madrid numbers were computed separately and are correct.
* **09 RESULT (Wed 00:20): neighbourhood features ADOPTED.** Tested on top of 5x5 smoothing, paired 10 seeds, sanity check passed.

| Amsterdam (smoothed) | n=5 | n=25 | n=50 | n=100 | n=200 | avg | wins n=25 |
|---|---|---|---|---|---|---|---|
| Screening: current | 0.589 | 0.669 | 0.701 | 0.718 | 0.734 | 0.682 | |
| G1 3x3 | +0.014 | +0.006 | 0.000 | -0.001 | +0.003 | +0.005 | 8/10 |
| **G2 3x3 + 7x7** | **+0.015** | **+0.014** | +0.007 | +0.007 | +0.010 | **+0.010** | **10/10** |
| **Hidden: current -> G2** | 0.518 -> 0.532 | **0.591 -> 0.603** | 0.628 -> 0.636 | 0.646 -> 0.651 | **0.652 -> 0.662** | **0.607 -> 0.617** | 9/10 |

  Madrid: random CV 0.662 -> 0.738 (**inflated, not a real gain**: in random CV a test pixel's neighbours are training pixels); spatial CV 0.617 -> **0.634** (honest); hidden Madrid + smoothing 0.609 -> 0.606 (within noise; Madrid is not a prize metric). Slides must show Madrid spatial CV next to random CV.

**v2 pipeline (Wed 01:00):** transfer_v3.py (NEIGHBOUR_RADII = (1, 3), SMOOTH_RADIUS = 2, load_data adds 266 neighbourhood columns -> 398 inputs), predict.py (neighbourhood features for test pixels over test + our own pixels; reads the setting from the saved model, so v1 models still work; smoothing ON by default), new notebook **00-MAIN-Pipeline-v2.ipynb** (adds Madrid spatial CV + spatial confusion, smoothed results table, v1 comparison, predict.py end-to-end check). Tested end to end on fake data. v1 stays in submission_frozen_v1/ and 00-MAIN-Pipeline-FINAL.ipynb. **Fallback:** if v2 is not finished and verified by Wed 08:30, submit v1 + 5x5 smoothing.

**v2 OFFICIAL RUN (00-MAIN-Pipeline-v2, finished Wed 01:20, verified 04:30, all checks OK):**

| Setting | v2 final (5x5 smoothing) | v2 without smoothing | v1 | Baseline |
|---|---|---|---|---|
| Madrid random 5x5 CV | 0.733 +/- 0.003 (optimistic) | | 0.656 | 0.618 |
| Madrid spatial CV | 0.629 +/- 0.034 (smoothed 0.630) | | 0.616 | |
| Amsterdam n=5 | **0.617 +/- 0.028** | 0.615 | 0.580 | 0.544 |
| Amsterdam n=25 | **0.675 +/- 0.014** | 0.670 | 0.641 | 0.599 |
| Amsterdam n=50 | **0.701 +/- 0.010** | 0.693 | 0.666 | 0.608 |
| Amsterdam n=100 | **0.718 +/- 0.004** | 0.703 | 0.683 | 0.613 |
| Amsterdam n=200 | **0.733 +/- 0.004** | 0.721 | 0.703 | 0.615 |
| **Amsterdam average** | **0.689** | 0.681 | 0.655 | 0.596 |

Smoothing better in 49/50 runs. All 5 v2 n beat v1 (check cell OK). Saved models reload with identical scores. predict.py end to end OK on raw files with v2 models (context neighbours used). Madrid spatial confusion (smoothed): class 1 53% correct (38% -> class 2), class 2 72%, class 3 57% (25% -> class 2), class 4 67%. Model files: stage 1 = 483 MB, each Amsterdam model about 235 MB (check the submission size limit). **v2 is the submission.** Abstract (147 words) and justification (271 words) in SUBMISSION_TEXT.md.

**Clean-room reproduction (Wed 04:45 to 06:10):** a new empty folder (`~/clean_run_test`) with only the 4 code files, the v2 notebook, README, requirements.txt and a link to the raw data; SKIP flags off; run headless with nbconvert in a fresh kernel. It rebuilt everything from raw files without errors. Compared with the official run: Madrid CV table, Madrid confusion, feature list IDENTICAL; feature files and all 6 model files identical in size; Amsterdam table identical to 3 decimals (two values differ in the 4th decimal: n=100 0.7180 vs 0.7181, n=200 without smoothing 0.7213 vs 0.7214; multi-threaded floating point in the forests). The reported 3-decimal results are fully reproducible. Package versions in requirements.txt (Python 3.12.3, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scipy 1.18.1, joblib 1.6.0, matplotlib 3.11.2).

**Scaling on the final method (Wed 08:40, results/scaling_comparison_v2.csv):** own-city scaling vs Madrid scaling, same seeds and smoothing: n=5 0.617 vs 0.600, n=25 0.675 vs 0.668, n=50 0.701 vs 0.698, n=100 0.718 vs 0.715, n=200 0.733 vs 0.731; average **0.689 vs 0.682**. Own-city scaling better at every n, most at low n. Backup ready if unlabelled data is ruled out (still +0.086 over the baseline). Slide 6 and the justification updated.

**Judge-style test run (Wed 08:00 to 08:40, submission_test/, Restart and Run All in the notebook):** rebuilt everything from raw data without errors; Madrid CV and confusion identical, Amsterdam table identical to 3 decimals (n=200 0.7331 vs 0.7332). **Organisers: no size limit on models.** SUBMISSION folder organised: START_HERE.md, PREDICT_with_saved_models.ipynb (no retraining), experiments/ (all notebooks with outputs), results/ (every results file), data/ (raw copies).

**Deck (Wed 06:00):** KasiData_deck_bold.pptx, 12 slides, bold black/red style, partner logos, "in a new city" thread, speaker notes with timing and Q&A answers on every slide. Abstract 147 words; justification 271 words (SUBMISSION_TEXT.md).


## 7f. Must go in the slides (checklist for the deck, Jian)

- [ ] Abstract (max 150 words) on slide 1
- [ ] Cleaning finding: patchy haze; temporal rule (photo brighter than 1.35x the pixel's own normal Blue) beats a fixed threshold that was biased against new buildings
- [ ] Features "when vs what": change and timing stats (132 features) lift Madrid 0.610 to 0.660
- [ ] City calibration: scaling Amsterdam on its own unlabelled data (Amsterdam is darker); show both columns, own 0.655 vs Madrid scaling 0.647
- [ ] Transfer method: blend3 (finetune + stack + prototype), results table with error bars incl. n=5 (+/- 0.038)
- [ ] Smoothing: average 0.655 to 0.671, 50 of 50 seed runs better
- [ ] F1 vs log2(n) plot (results/f1_vs_log2n.png)
- [ ] Honest checks: spatial CV 0.616 vs random 0.657; zero-shot 0.453; confusion shows errors fall in neighbouring classes
- [ ] Neighbourhood features (section 7i): "+0.010 average, +0.013 at n=25 on hidden neighbourhoods"; explain Madrid random CV inflation honestly
- [ ] "We tested 9 ideas with a rule fixed in advance; 2 passed and were confirmed on unseen neighbourhoods"
- [ ] **Model comparison table above** ("we tested 6 model families on identical splits")
- [ ] Reverse direction (section 7g): "works both ways", plot results/reverse_vs_forward.png, "91% of the Amsterdam all-labels ceiling at n=200"
- [ ] Stress test (section 7g): brightness absorbed, noise/cloud cost 1 to 3 points, own scaling wins in every condition at low n
- [ ] Dress rehearsal (section 7h): "expected score on unseen neighbourhoods", plot rehearsal/dress_rehearsal.png, smoothing 48/50
- [ ] Uncertainty maps (section 7h): results/uncertainty_maps_amsterdam_n025.png + "most sure 20% = 84% right, least sure = 45%"
- [ ] Tried and rejected: self-training (section 7h), HistGradientBoosting (7e)
- [ ] Limitations + future work: one city pair only; other climates (Windhoek); national registers (BAG, Catastro)

## 8. Next steps (Tue 09:50)

| Step | Who | Task | Done when |
|---|---|---|---|
| 1 | Jian | Step 1 rerun with temporal haze rule | DONE 09:39 |
| 2 | Jian + Fenny | Rerun 02-Feature-Table-v3 on the new clean files | Checks pass, comparison table saved |
| 3 | Silvio | Stage 1 on final features: RF, 5 folds x 5 repeats, confusion matrix proportions, save `models/stage1_madrid_model.joblib` | Madrid F1 mean +/- std logged |
| 4 | Antonio | Stage 2: all methods in transfer_v3.py at n = 5, 25, 50, 100, 200 x 10 seeds; check against section 7c; pick final method | Results table + chosen method logged |
| 5 | Anselmo | Fill main notebook steps 3 to 5 with transfer_v3.py; results table + F1 vs log2(n) plot; Git + README | Restart and Run All works end to end |
| 6 | Jian | Slides: cleaning story, "when transfers, what does not", method comparison; ask organisers about unlabelled Amsterdam data | Slides drafted |

## 9. Experiment log

| Time | Who | Change | Madrid F1 (mean +/- std) | Ams n=25 F1 | Keep? |
|---|---|---|---|---|---|
| ref | Organisers | Official baseline (RF, 60 features, prototype transfer) | 0.618 +/- 0.004 | 0.599 | reference |
| Tue 08:58 | Fenny/Jian | v1 features (30), RF 300, prototype | 0.610 +/- 0.006 | 0.508 | baseline |
| Tue 08:58 | Fenny/Jian | v2 all (46) | 0.631 +/- 0.005 | 0.531 | no |
| Tue 08:58 | Fenny/Jian | **v3 all (132)** | **0.660 +/- 0.006** | **0.534** | **YES, frozen** |
| Tue 08:58 | Fenny/Jian | v3 stable (43, label-free shift filter) | 0.617 +/- 0.007 | 0.520 | no |
| Tue 09:15 | Jian/Claude | Amsterdam own scaling (prototype) | | 0.571 | yes (confirm with organisers) |
| Tue 09:40 | Jian/Claude | Temporal haze rule (prototype, own scaler) | 0.658 +/- 0.005 | 0.584 | **YES, adopted** |
| Tue 10:00 | Jian/Claude | Stack transfer | | 0.627 | yes |
| Tue 10:00 | Jian/Claude | **Blend stack + prototype** | | **0.629** | **current best** |
| Tue 10:45 | Jian/Claude | **blend3 (finetune + stack + prototype), final cleaning** | 0.656 +/- 0.003 | **0.641** | **YES, final** |
| Tue 11:30 | Jian/Claude | 3x3 spatial smoothing (alpha 0.5) | spatial CV 0.617 to 0.632 | 0.658 | **YES** |
| Tue 13:30 | Jian/Claude | 00 FINAL reproduced from raw data | 0.656 | 0.641 | confirmed |
| Tue 15:00 | Jian/Claude | HistGradientBoosting instead of RF | 0.661 +/- 0.004 | | no (within noise, 10x slower) |
| Tue 15:00 | Jian/Claude | ExtraTrees / MLP / kNN / LogReg | 0.648 / 0.654 / 0.628 / 0.604 | | no |
| Tue 18:30 | Jian/Claude | Reverse direction (Amsterdam to Madrid), blend3 | Ams->Mad n=25 0.512 | (fwd 0.641) | evidence only |
| Tue 18:30 | Jian/Claude | Stress test (darker/brighter/noise/missing) | | 0.626 to 0.637 | evidence only |
| Tue 20:30 | Jian/Claude | Dress rehearsal, hidden neighbourhoods (+ smoothing) | 0.587 (0.609) | 0.574 (0.587) | evidence; smoothing ON |
| Tue 20:50 | Jian/Claude | Self-training (pseudo-labels) | | 0.637 vs 0.641 | **no** |
| Tue 23:00 | Jian/Claude | 08: D, I, H, E, A, C | | within noise or worse | **no** |
| Tue 23:00 | Jian/Claude | 08: smoothing 5x5 (hidden-confirmed) | hidden 0.617 | hidden 0.591 | **YES, adopted** |
| Wed 00:20 | Jian/Claude | 09: neighbourhood features 3x3 + 7x7 (hidden-confirmed) | spatial 0.634 (random 0.738 inflated) | hidden 0.603 | **YES, adopted (v2)** |

## 10. Open questions for the organisers

1. Are band values raw Collection 2 integers (reflectance = DN x 0.0000275 - 0.2) or already reflectance?
2. Are years exactly on a boundary (1960, 1984, 2004) meant for the older or the newer class?
3. Can we use unlabelled Amsterdam features (e.g. per-city standardisation)?
4. The brief mentions "four" Amsterdam F1 scores but lists five sample sizes. Which go on the slide?
5. How will our code be run on the hidden test set (function, notebook, or predictions file)?

## 11. Schedule

| When | Session | Our goal |
|---|---|---|
| Mon 14:00 to 17:15 | Baseline + refinement | Steps 1 to 4 done, baseline numbers logged |
| Tue 08:30 to 13:00 | Tuning + cross-city adaptation | Improvements one at a time, transfer methods compared |
| Tue 14:00 to 15:30 | Evaluation curves | Final F1 table and F1 vs log2(n) plot |
| Tue 16:00 to 17:30 | Code cleaning + slides | Method frozen, slides filled |
| Wed 08:30 to 11:00 | Final sprint | Restart and Run All, export models, finish slides |
| **Wed 11:30** | **Freeze** | Models, scripts, slides, 300-word justification submitted |
| Wed 13:30 | Presentations | 8 min pitch + 4 min Q&A |


### 7j. Final predict-only check (morning of 18 Nov)
- Ran PREDICT_with_saved_models.ipynb end to end. It caught an edge case: when the test file contains ALL of our own Amsterdam pixels, there are no extra context pixels left and predict.py crashed (empty array).
- Fix in predict.py: if no context pixels remain, it simply uses the test pixels' own neighbours. Copied to SUBMISSION, the main folder and submission_test.
- Rerun: 25,992 of 25,992 pixels predicted, no crash. The F1 it prints (0.665) is on ALL pixels, including the few labelled ones, so it is a smoke test only, not a result to quote. Our official numbers stay in results_table_v2.csv.
