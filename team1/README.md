# Building Age Classification — hosted JupyterLab live demo

**Open `Start_Here.ipynb` and run its cells from top to bottom.**
This is a new, clean **demonstration workspace** built from the user's original Stage 2 source.
It keeps the predictor implementation, runnable entry script and tests intact.

## Quick start on the organizer's hosted JupyterLab server

1. Upload `BuildingAge_JupyterLab_Demo.zip` to the server's file browser, preferably into `~/hackathon_results/`.
2. Open a **JupyterLab Terminal** in the directory where the ZIP was uploaded, then run:

   ```bash
   unzip BuildingAge_JupyterLab_Demo.zip
   ```

   If `unzip` is unavailable, use: `python -m zipfile -e BuildingAge_JupyterLab_Demo.zip .`

   If the folder already exists, do not overwrite it. Rename/remove only your **demo copy** before extraction, not any saved experiment or final submission.
3. Open the new folder `BuildingAge_JupyterLab_Demo` in JupyterLab.
4. Double-click **`Start_Here.ipynb`**, choose the server's working Python 3 kernel, and select **Run > Run All Cells**. You may instead run one cell at a time to explain the workflow.
5. The notebook uses the existing trusted models under `~/hackathon_results/presentation_day_submission_20261118/model_bundle` and the verified Amsterdam archives, **without retraining the Madrid models**.
6. Inspect results in `demo_outputs/`. The input file in `demo_inputs/` has private labeled support rows; **keep that folder private, and do not include it in the official submission**.

## What is included

```text
BuildingAge_JupyterLab_Demo/
├── Start_Here.ipynb             # Main demonstration; BEGIN HERE
├── age_hackathon/
│   ├── __init__.py
│   └── predictor.py            # Actual original model code
├── run_stage2.py              # Actual Stage 2 CLI entry point
├── run_stage1.py              # Stage 1 CLI helper
├── example_predict.py
├── build_bundle_on_jupyter.py  # Original builder; DON'T RUN again
├── tests/test_predictor.py
├── demo_inputs/               # Created demo inputs (PRIVATE)
├── demo_outputs/              # Predictions and figures
├── requirements.txt
├── requirements-demo.txt
├── README_SOURCE.md            # Original source README
└── README.md                   # This guide
```

## Actual prediction entry point

`run_stage2.py` is the Stage 2 program. The notebook builds a **private demonstration NPZ** with
`support102`, `support252`, `support_labels`, `query102`, `query252`, optional `query_pixel_ids`,
then invokes the actual program by subprocess. It does not use held-out query labels during adaptation.

Example command once a correct private `demo_inputs/...npz` has been made:

```bash
python run_stage2.py --bundle /home/participant-a5/hackathon_results/presentation_day_submission_20261118/model_bundle \
  --input demo_inputs/demo_stage2_200shots.npz --n 200 \
  --output demo_outputs/my_new_predictions.npz
```

The notebook shows the exact output file name it creates. To run this manually, choose a filename that doesn't exist.

## Required trusted server files (NOT included in this small ZIP)

- `~/hackathon_results/presentation_day_submission_20261118/model_bundle/`
  with `manifest.json`, `schema.npz`, `madrid_rf60.joblib` and `madrid_lgbm354.joblib`
- `~/hackathon_results/exp013h_102feature_candidate.npz`
- `~/hackathon_results/exp017c_amsterdam_annual_history/amsterdam_annual_252features.npz`
- `~/hackathon_results/exp017f_random_support_354/trial_checkpoints/trial_01.npz`
- `~/hackathon_results/private_recovery/verified_fixed_data_backup.npz` (PRIVATE support and later scoring labels)
- optional ten-trial summary: `~/hackathon_results/exp017f_random_support_354/lgbm102_vs354_10trials.npz`

If a required archive is missing on your hosted server, the notebook identifies its exact path. Do not run the old bundle builder or retrain just to fix a missing file.

## Model and claims

- Madrid-trained Random Forest on 60 features contributes 20%.
- Madrid-trained LightGBM on 354 features contributes 35%; at 25+ shots/class it continues for 120 support-only rounds.
- Amsterdam-local shrinkage LDA on 78 features contributes 45%.
- Five supported Stage 2 budgets: 5, 25, 50, 100, 200 labeled pixels per class.
- Existing **200-shot development** 10-trial mean Macro F1: **0.7232**; not an official hidden-test score.
- The current inference scripts require **engineered feature arrays**. They do **not** accept standalone JPG/PNG/GeoTIFF or arbitrary raw Parquet files. Full raw preprocessing is not verified.

**Do not submit this demo ZIP in place of your already verified 251.98 MB `BuildingAge_Final_Submission.zip`, unless organizers expressly want the demo source.** The original final submission contains the trained models; this small demo uses the models already on your server.
