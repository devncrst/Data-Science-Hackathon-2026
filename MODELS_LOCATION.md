# Where are the models and the data?

They are **not in this branch** because GitHub refuses files over 100 MB.

| Folder | Size | Where it is |
|---|---|---|
| `models/` (6 trained models) | about 1.6 GB (Stage 1 alone 472 MB) | on the hackathon server: `/shared/collab/Building_Age_Classification/SUBMISSION_FINAL/models/` |
| `data/` (the organisers' raw files) | 142 MB + 44 MB | the organisers' own released files; also in the same server folder under `data/` |

## To use the saved models (no retraining)
Copy this branch next to `models/` and `data/` (or work directly in the server folder above), then follow `START_HERE.md` section B and run `PREDICT_with_saved_models.ipynb`.

## To rebuild the models instead
Put the two raw parquet files in `data/` and run `00-MAIN-Pipeline-v2.ipynb` (about 45 to 80 minutes, seed 42).
It recreates `models/` and every result file; two clean runs matched our shipped numbers to 3 decimals.

Team 4, KasiData
