# h4d2_model.pkl: frozen h4d2 in one file

Team Madrid2Dam, Hack4dev 2026. Method frozen and declared in commit 48a195f; this file is packaging only.

## Load and predict

    import joblib, pandas as pd
    model = joblib.load("h4d2_model.pkl")          # self-contained: does NOT need the h4d2/ package
    df = pd.read_parquet("amsterdam_test.parquet")  # raw organiser format
    out = model.predict(df, n=25)                   # n in 5, 25, 50, 100, 200 -> Amsterdam Stage 2 model
    out_mad = model.predict(pd.read_parquet("madrid_test.parquet"), n="madrid")   # or n=None -> Madrid Stage 1

predict returns a DataFrame with one row per pixel: px_key, py_key, pred_class (1-4).
model.available_n lists the Amsterdam label budgets. A call takes about 25 s (Amsterdam) or 50 s (Madrid) on 16 CPUs.

## Input columns (raw test parquet, one row per pixel-year)

px_key, py_key, year, and for each observation slot s = 1, 2, 3: Blue_s, Green_s, Red_s, NIR_s, SWIR1_s, SWIR2_s, qa_valid_s.
weighted_mean_year is optional (not used for prediction). Other columns are ignored.

## What is inside

Inference code (verbatim from h4d2/core.py plus the Stage 2 predict logic), the Stage 1 Madrid logistic regression,
the six CNNs (CPU weights), the five Stage 2 Amsterdam models (seed 42), and the provided Amsterdam and Madrid pixel
time series with labels removed. These serve as unlabelled spatial context, as in scripts/submit_predict.sh
(context=True, the default). Pass context=False to use only the test pixels.

The pickle rebuilds the model from embedded source on load (via eval/exec), so load it only from this trusted file.

## Verified

Fresh Python session in an empty folder (python -I, no h4d2 import): predictions for the 2,000-pixel dry-run samples
match scripts/submit_predict.sh byte for byte for n = 5, 25, 50, 100, 200 and Madrid.

## Environment used to build and test (hub, /shared/venv/bin/python)

| Package | Version |
|---|---|
| Python | 3.12.3 |
| scikit-learn | 1.9.1 |
| PyTorch | 2.15.0 (CPU inference; built on the +cu132 wheel, no GPU needed) |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| pyarrow | 26.0.1 (to read the parquet) |

Rebuild: H4D_DATA_DIR=<data> /shared/venv/bin/python scripts/build_pkl.py
