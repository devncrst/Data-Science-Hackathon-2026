# START HERE: KasiData submission (Building age, Madrid to Amsterdam)

Nothing needs to be run to see our results: every output is already saved in this folder.
There are three ways to use it.

## A. Just look (nothing to run)
- **Final results:** `results_table_v2.csv` (Amsterdam macro F1, mean and std for n = 5, 25, 50, 100, 200), `f1_vs_log2n_v2.png` (F1 vs log2 n with error bars), `results/madrid_cv_v2.csv` (Madrid CV), `results/madrid_confusion_proportions_v2_spatial.csv`.
- **The official run with all outputs saved:** `00-MAIN-Pipeline-v2.ipynb` (open it to read the outputs; running it retrains, see C).
- **Every experiment, with saved outputs:** `experiments/` (notebooks 01 to 09, the v1 pipeline, and `PROJECT_LOG.md` with every decision and number). These are evidence; they are not needed to use the models.
- **Write-ups:** `README.md` (method, how to run, checks), `SUBMISSION_TEXT.md` (abstract + 300-word justification), the slide deck.

## B. Score new data with our SAVED models (no retraining, a few minutes)
Open **`PREDICT_with_saved_models.ipynb`**, set `TEST_FILE` (and `CITY`, `N_PER_CLASS`), then Run All.
It loads `models/` as they are: the Stage 1 Madrid model and the 5 Amsterdam models (one per n).

## C. Retrain everything from scratch (about 45 to 80 minutes)
Open **`00-MAIN-Pipeline-v2.ipynb`** and use Kernel > Restart Kernel and Run All.
It reads the raw files in `data/`, rebuilds cleaning, features, all models and all results, and overwrites `models/` and the results files with identical ones (fixed seed 42; verified by two clean-room runs in empty folders, same numbers to 3 decimals).

Run everything from inside this folder (paths are relative). Package versions: `requirements.txt`.
