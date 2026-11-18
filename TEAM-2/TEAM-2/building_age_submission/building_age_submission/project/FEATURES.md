# Feature table (final model: 120 features per pixel)

Each 30 m building pixel has 40 years of Landsat readings (1984-2025) in 6 bands:
Blue, Green, Red, NIR (near-infrared), SWIR1 and SWIR2 (short-wave infrared).
Notebook 3 turns this history into 60 features; save_features.py adds 60 neighbour features.
Early = 1984-2003, late = 2004 onwards (the class 3/4 boundary).
Not used as features: weighted_mean_year (the label, so no leakage) and coverage.

| # | Columns | Count | What it is | Why it helps |
|---|---|---|---|---|
| 1 | Blue_mean, Green_mean, Red_mean, NIR_mean, SWIR1_mean, SWIR2_mean | 6 | Average reflectance per band over 40 years | Brick, stone, concrete, glass and metal reflect light differently: the material signature of an era |
| 2 | Blue_std ... SWIR2_std | 6 | Variability of each band over 40 years | A pixel that changed (construction, renovation) varies more than a stable old building |
| 3 | NDVI_mean, NDBI_mean, UI_mean, MNDWI_mean, BSI_mean | 5 | Average spectral indices: vegetation (NDVI), built-up (NDBI), urban (UI), water (MNDWI), bare soil (BSI) | Ratios of bands cancel much of the light and climate difference, so they travel between cities |
| 4 | NDVI_std ... BSI_std | 5 | Variability of each index | Vegetation lost or built surface gained over time points to recent construction |
| 5 | Blue_early_mean ... SWIR2_early_mean, Blue_early_std ... SWIR2_early_std | 12 | Band mean and variability in 1984-2003 | What the site looked like before 2004: field or building already there |
| 6 | Blue_late_mean ... SWIR2_late_mean, Blue_late_std ... SWIR2_late_std | 12 | Band mean and variability from 2004 | What the site looks like now; compared with early, shows post-2004 building |
| 7 | d_Blue_mean ... d_SWIR2_mean, d_Blue_std ... d_SWIR2_std | 12 | Year-on-year change in each band (mean and variability) | A sudden jump in the light curve marks construction: field, then site, then building |
| 8 | has_early_data, has_late_data | 2 | Whether clear observations exist in each period | Tells the model when early or late features are missing (clouds, sensor gaps) |
| 9 | nb_ + each of the 60 columns above | 60 | Average of the same feature over the 8 surrounding pixels | Neighbourhoods are built in the same era (Tobler's first law); adds +0.013 on Amsterdam and +0.010 on unseen 1 km blocks |

Total: 60 own-pixel features + 60 neighbour features = 120.
Each city is standardised on its own data (mean 0, std 1 per feature, no labels).
Backup version (features_v2_plain.pkl): groups 1-8 only, 60 features.

Tested and dropped (they raised Madrid but not Amsterdam):
- 156 combined features: Madrid 0.696, Amsterdam 0.639
- Early vs late index change: 0.638
- Construction-year features (jump year and size, slope, coverage): Amsterdam 0.669, no gain; zero-shot 0.503

## Tested and rejected (evidence)

All tests use the official protocol (Madrid 5-fold stratified CV; Amsterdam 10 draws, seed 42+r). Final model: Amsterdam average 0.6687, n=25 0.6600, zero-shot 0.5263, Madrid 0.6797.

### Feature selection (6 versions)
| Version | Amsterdam avg | Note |
|---|---|---|
| A_noflip (drop MNDWI/BSI, which flip between cities) | 0.6700 | zero-shot 0.5517 |
| B_noredundant (drop pairs with abs r > 0.95) | 0.6692 | |
| C_lean | 0.6696 | |
| D_noshift | 0.6673 | |
| E_contrast | 0.6550 | |
| F_quantile | 0.6627 | zero-shot 0.4959 |
Decision: none gains more than +0.002 on average, so the 120 features are kept.

### Label-reliability weighting
| Weighting | Amsterdam avg | n=25 | zero-shot | Madrid |
|---|---|---|---|---|
| none (final) | 0.6687 | 0.6600 | 0.5263 | 0.6797 |
| coverage | 0.6694 | 0.6620 | 0.5312 | 0.6774 |
| boundary year +-1 | 0.6701 | 0.6618 | 0.5148 | 0.6775 |
| both | 0.6703 | 0.6622 | 0.5223 | 0.6779 |
Decision: gains are at most +0.0016, below draw-to-draw noise, and boundary weighting lowers zero-shot. Not adopted.

## Spatial validation
| Test | Macro-F1 |
|---|---|
| Madrid random StratifiedKFold (official, as baseline) | 0.6797 +- 0.0065 |
| Madrid StratifiedGroupKFold, 1 km blocks (34x34 px) | 0.6163 +- 0.0137 |
| Amsterdam n=25 official / buffer | 0.6600 / 0.6581 |
| Amsterdam n=200 official / buffer | 0.7160 / 0.7081 |
Buffer = test pixels touching (8-neighbourhood) any labelled pixel are removed. Amsterdam scores are not inflated by neighbouring pixels.

## Full evaluation (full_evaluation.py, reproduces all 8 official numbers)

### Model families (same 5 Madrid folds)
| Model | Madrid CV | Amsterdam zero-shot |
|---|---|---|
| Decision tree | 0.5600 | 0.4623 |
| Logistic regression | 0.5976 | 0.5275 |
| Random forest | 0.6606 | 0.5338 |
| Neural network (MLP) | 0.6736 | 0.4581 |
| LightGBM (ours) | 0.6797 | 0.5263 |
The MLP fits Madrid but does not transfer. Only LightGBM can continue training on Amsterdam labels (init_model), which gives the adaptation gain.

### Few-shot method: organisers' prototype method vs ours (same features, same 10 draws)
| n | 5 | 25 | 50 | 100 | 200 | Avg |
|---|---|---|---|---|---|---|
| Prototype | 0.5373 | 0.6114 | 0.6196 | 0.6278 | 0.6284 | 0.6049 |
| Ours | 0.5887 | 0.6600 | 0.6784 | 0.7005 | 0.7160 | 0.6687 |
Ours wins at every n; prototypes plateau, fine-tuned trees keep learning. No negative transfer.

### Diagnostics
- ROC-AUC (one-vs-rest) 0.87-0.91 for n >= 25; quadratic kappa 0.72 (Madrid), 0.74 (n=200).
- 77% of errors are in the adjacent era (about 50% expected by chance).
- Oldest class: Amsterdam recall 0.28 zero-shot, 0.71 at n=25, 0.76 at n=200.
- Uncertainty flags errors: AUC 0.71 (prototype 0.67) on the 25-shot split.
Outputs: reports/tables/MODEL_PERFORMANCE_SUMMARY.txt and reports/figures/.
