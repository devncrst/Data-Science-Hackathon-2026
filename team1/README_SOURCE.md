# Building Age Classification — Stage 2 Submission Source

**Presentation-day model:** frozen Madrid `RandomForest60` (20%) + Madrid `LightGBM354` continued on *only* target-city labeled support rows (35%) + support-trained shrinkage `LDA78` (45%).

**Verified development metric:** Experiment 017F at 200 labels/class: mean Amsterdam random-support Macro F1 **0.7232** ± population SD **0.0041** over 10 repeated support samples. These reuse query labels in development and are **not independent hidden-test results**. Buffered spatial evaluation around **0.6612**.

## Build on hosted JupyterLab

1. Upload this source ZIP into your hosted JupyterLab working directory and extract it.
2. Run `python build_bundle_on_jupyter.py` from the extracted source directory (or use the notebook cell from ChatGPT). This finds the trusted saved models and feature matrices under `~/hackathon_results`.
3. The builder refuses to ship unless the complete packaged Stage 2 function **exactly reproduces the previously saved 017F trial-1 predictions** (25,192 query rows), **without reading any query labels**. It also smoke-tests n=5.
4. The resulting ZIP is at `~/hackathon_results/presentation_day_submission_20261118/BuildingAge_Stage2_Submission.zip`.

## API

```python
from age_hackathon import load_bundle
model = load_bundle('model_bundle')
model.adapt(support102, support252, support_labels, n=200)
predictions = model.predict(query102, query252)
```

`n` must be one of `5, 25, 50, 100, 200`, and all classes must have exactly `n` support labels. At 5 shots the Madrid LightGBM remains frozen; at 25+ it continues for 120 rounds. The new `StandardScaler` and shrinkage `LDA78` fit only on the support examples. **No query labels enter prediction.**

## Command-line runnable Stage 2

Run with a processed features NPZ containing `support102`, `support252`, `support_labels`, `query102`, `query252` (and optionally `query_pixel_ids`) but **no query labels**:

```bash
python run_stage2.py --bundle model_bundle --input inputs.npz --n 200 --output predictions_200.npz
```

Change `--n` to 5, 25, 50, or 100 for the other competition budgets. This uses a fixed random state in the LightGBM continuation and outputs integer predictions aligned to the query rows. The caller must supply the correct original preprocessing and class-balanced support rows.

## Important remaining requirement

This is a **feature-level deployable source and model bundle**, not yet a proven raw-Parquet-to-model end-to-end implementation. Inputs are the exact saved **102 engineered original features plus 252 raw observed annual bands** for each pixel in the same columns and units used to train the model. The organizer's original 60-feature preprocessing and custom 42-feature construction must be verified on independent raw input before claiming arbitrary unseen rows work. Do not supply the old `preprocessed_data.pkl` scaler: its provenance differs from the frozen training matrices. Never put the private `verified_fixed_data_backup.npz` or Amsterdam labels into the submission ZIP.

## Reproducibility / provenance

Model bundle includes SHA256 manifest for each artifact and Python/scikit-learn/LightGBM versions. Only load joblib models from your trusted files. The resulting bundle is not a hosted API or interactive app: it is a runnable offline Stage 1/Stage 2 Python inference package. The real hidden-test execution environment and full raw-input contract still require organizer verification.
