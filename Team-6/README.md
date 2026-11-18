# Building age classification: Madrid to Amsterdam transfer

Open `Building_Age_Pipeline.ipynb` and use Run All. It runs the whole pipeline with fixed seeds; `N_PER_CLASS` in its first code cell sets the number of labelled Amsterdam pixels per class. With the default settings it loads saved results and takes about a minute; three switches in the same cell rerun the long steps.

## The scripts, in the order they run

```bash
pip install lightgbm pyarrow scikit-learn scipy pandas matplotlib joblib

# 1. Stage 1: Madrid cross-validation (random folds and 1 km spatial blocks), then the final model
python train_stage1.py --madrid ../data/madrid_train.parquet --out outputs --trees 150 --lr 0.1

# 2a. Amsterdam, selection: try every method at n = 5, 25, 50, 100, 200; best per n -> best_per_n.json
python evaluate_fewshot.py --amsterdam ../data/amsterdam_data.parquet --out outputs --seed 42

# 2b. Amsterdam, report: score only those choices on fresh random draws (the numbers to present)
python evaluate_fewshot.py --amsterdam ../data/amsterdam_data.parquet --out outputs --seed 7 \
    --choose-from outputs/best_per_n.json --tag report_

# 3. Where the gain comes from: 60 vs 393 features, prototypes vs extra-trees, with and without Stage 1
python ablation.py --out outputs --trials 30 --seed 7

# 4. The organisers' random forest under the same Madrid folds, for a fair spatial comparison
python baseline_cv.py --madrid ../data/madrid_train.parquet --out outputs

# 5. The deliverable: adapt to Amsterdam with n labelled pixels per class (add --test file.parquet for held-out data)
python adapt_amsterdam.py --amsterdam ../data/amsterdam_data.parquet --out outputs --n 25 --seed 42
```

## Files

| File | What it does |
| --- | --- |
| `Building_Age_Pipeline.ipynb` | The deliverable notebook: every step above, with results and figures |
| `bac.py` | Shared library: labels, features, Stage 1, Stage 2, spatial blocks, confusion-matrix figure |
| `train_stage1.py` | Madrid cross-validation and the final Stage 1 model |
| `evaluate_fewshot.py` | Amsterdam few-shot evaluation (selection and report runs) |
| `adapt_amsterdam.py` | Adapts Stage 1 to Amsterdam for one n; can predict a held-out file |
| `baseline_features.py` | The organisers' 60 features, recomputed with Notebook 3's steps |
| `ablation.py` | Feature / classifier / Madrid-model comparison on identical draws |
| `baseline_cv.py` | The organisers' random forest under our random and spatial folds |
| `WRITTEN_JUSTIFICATION.md` | The 300-word justification |
| `tests/make_synthetic.py` | Stand-in data with the same columns, for checking the code runs |

## Results (macro F1)

| Result | Ours | Baseline |
| --- | --- | --- |
| Madrid, random 5-fold cross-validation | 0.750 +/- 0.005 | 0.618 (organisers) |
| Madrid, 1 km spatial block folds | 0.659 +/- 0.012 | 0.591 +/- 0.013 (their model, our folds) |
| Amsterdam, zero-shot | 0.450 | 0.343 |
| Amsterdam, n = 5 | 0.583 +/- 0.048 | 0.544 |
| Amsterdam, n = 25 | 0.688 +/- 0.011 | 0.599 |
| Amsterdam, n = 50 | 0.709 +/- 0.011 | 0.608 |
| Amsterdam, n = 100 | 0.731 +/- 0.009 | 0.613 |
| Amsterdam, n = 200 | 0.750 +/- 0.004 | 0.615 |
| Amsterdam, average of the five | 0.692 | 0.596 |

Amsterdam numbers come from the report run: 30 fresh random draws, with the method per n fixed beforehand by the selection run. On the selection draws themselves the average was 0.697; the difference is the optimism that choosing and reporting on the same draws would have added.

**Why two Madrid numbers.** Random folds put a test pixel's neighbours in the training set. Neighbours share construction history and, through the neighbourhood features, information, so random folds overstate accuracy on areas the model has not seen. Spatial block folds keep whole ~1 km blocks together.

**Where the gain comes from** (ablation, same 30 draws, n = 25): the organisers' method scores 0.596; switching prototypes for extra-trees adds 0.041; our 393 features add 0.048; the Stage 1 Madrid model's outputs add 0.003. With prototypes, the new features add only 0.009. A quick test of training one model on Madrid and Amsterdam pixels together gave no gain either.

## Rules this code respects

- Stage 1 uses Madrid data only.
- Stage 2 uses only the n labelled Amsterdam pixels per class for anything involving labels.
- Labels use the organisers' rule: `(-inf, event]`, `(event, 1984]`, `(1984, 2004]`, `(2004, inf)`.
- `--city-year-norm` (off by default) normalises against all pixels of a city, labelled or not. Turn it on only if the organisers confirm unlabelled Amsterdam pixels may be used.

## Practical notes

- **Runtime and memory.** With 6 threads, Stage 1 cross-validation takes about 1.5 minutes per fold; the evaluation runs take 10 to 25 minutes. Building the Madrid features peaks at about 1.7 GB of memory. `N_JOBS` in `bac.py` (default 6) sets the CPU threads; keep it modest on a shared server, because LightGBM slows sharply when other jobs compete for the same cores.
- **Held-out test file.** `adapt_amsterdam.py --test file.parquet` (or `TEST_FILE` in the notebook) predicts a file without labels. It assumes the same columns as the provided data; if `coverage` is missing, the coverage features fall back to Madrid medians.
- **Recomputed baseline.** `baseline_features.py` follows Notebook 3; on our random folds the organisers' random forest scores 0.629 (they report 0.618 over 5 x 5 folds), and the prototype baseline matches theirs to within 0.005.
