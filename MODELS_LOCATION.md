# The trained models are in `models_split/`

GitHub refuses files over 100 MB, so each of our 6 models was saved **compressed** (joblib compress=3: 1.6 GB down to about 374 MB)
and cut into pieces under 100 MB. They are the same models: on the full Amsterdam file the rebuilt models give identical
predictions for every pixel at every n, and every tree of the Stage 1 forest is identical.

## Rebuild `models/` (one command, a few seconds)
```bash
python reassemble_models.py
```
It joins the pieces and checks each file against the SHA-256 in `models_split/MANIFEST.txt`.
Then follow `START_HERE.md` section B and run `PREDICT_with_saved_models.ipynb` (no retraining).

## The data
`data/` (the organisers' raw files, 142 MB + 44 MB) is not included: it is the organisers' own release.
Put `amsterdam_data.parquet` and `madrid_data.parquet` in `data/` to rerun the pipeline.
The full original folder, with uncompressed models and data, is also on the hackathon server:
`/shared/collab/Building_Age_Classification/SUBMISSION_FINAL/`

## Or retrain from scratch
Run `00-MAIN-Pipeline-v2.ipynb` (about 45 to 80 minutes, seed 42). It recreates `models/` and every result file.

Team 4, KasiData
