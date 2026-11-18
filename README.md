# Team-3 Madrid2Dam (`h4d2`)

This is Team 3's Hack4dev 2026 entry. The model, `h4d2`, predicts a building's age class (1 to 4) from a 42-year Landsat time series. It is trained on labelled Madrid pixels, then adapted to Amsterdam with only a few labelled examples per class (`n` = 5, 25, 50, 100 or 200). The frozen model file is `h4d2_model.pkl`. Method details are in `docs/h4d2_method_declaration.md` and `h4d2/README.md`. The organiser parquet files are not in this repository.

## Official scores

Protocol: organiser notebook draw order, seed 42, 10 draws, macro F1 on every Amsterdam pixel that was not given as a label. Stored (unrounded) values are in `h4d2/artifacts/results/official_and_seed1000.json`.

| Model | n=5 | n=25 | n=50 | n=100 | n=200 | Average |
|---|---|---|---|---|---|---|
| h4d2 | 0.642 | 0.698 | 0.715 | 0.736 | 0.748 | 0.708 |
| Organiser baseline | 0.544 | 0.599 | 0.608 | 0.613 | 0.615 | 0.596 |

## Requirements

The official `requirements.txt` (present in the Team-3 submission folder) was written for **Python 3.12**. The environment used to build and test the pickle on the hackathon hub was Python 3.12.3, with CPU inference. No operating system is declared. The pinned packages are:

```
numpy==2.5.4
pandas==2.3.3
scipy==1.18.1
torch==2.15.0
joblib==1.6.0
pyarrow==26.0.1
scikit-learn==1.9.1
```

On the hub, PyTorch 2.15.0 came from the `+cu132` wheel; GPU is not required for inference. `requirements.txt` does not pin a PyTorch index URL, so `pip` will install whatever `torch==2.15.0` wheel your platform provides.

Do not use `pip install -e .`; run every command from the `Team-3` folder so `python -m h4d2` can import the local `h4d2/` package.

Jupyter, `ipywidgets` and `matplotlib` are used by `h4d2/Dashboard.ipynb` and `python -m h4d2.report`, but they are not listed in `requirements.txt`.

## Install

Work in the project folder: the directory that contains `h4d2/`, `scripts/`, `h4d2_model.pkl` and `requirements.txt`.

### 1. Clone

The project lives in the `Team-3` folder of the `Team-3` branch:

```bash
git clone https://github.com/devncrst/Data-Science-Hackathon-2026.git
cd Data-Science-Hackathon-2026
git checkout Team-3
cd Team-3
```

### 2. Create a virtual environment and install packages

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

The organiser parquet files are **not** in the repo (the `data/` folder is gitignored). Set `H4D_DATA_DIR` to the folder that contains both of these exact filenames, which is what `h4d2/cli.py` and the dashboard loaders expect:

- `madrid_train.parquet`
- `amsterdam_data.parquet`

If `H4D_DATA_DIR` is unset, `python -m h4d2` and `python -m h4d2.dashboard_data` default to `../Building_Age_Classification/data`. The live dashboard helper defaults to `/home/participant-c1/Building_Age_Classification/data`. Those are hub paths, not something this repo provides. Always set the variable yourself:

```bash
export H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files
```

Windows (PowerShell):

```powershell
$env:H4D_DATA_DIR = "C:\path\to\the\folder\with\the\two\parquet\files"
```

Committed Stage 1 and Stage 2 models live in `h4d2/artifacts/`. Point `H4D2_OUT` at that folder when you want to use them (`scripts/submit_predict.sh` already defaults to it). If you leave `H4D2_OUT` unset, `python -m h4d2` writes to `./outputs_h4d2` instead.

## Run

All commands below assume you are in the project folder, with the virtual environment active.

### 1. Verify the pickle

The file `h4d2_model.pkl` should have SHA-256:

```
6690c9222bff83f3bcd1be417817a6fed01262f40f80327a3d89bc53629bc370
```

macOS or Linux:

```bash
sha256sum h4d2_model.pkl
```

(on macOS, `shasum -a 256 h4d2_model.pkl` is an equivalent check)

Windows (PowerShell):

```powershell
Get-FileHash h4d2_model.pkl -Algorithm SHA256
```

There is no verification script in the repo. The hash is the one recorded for the Team-3 submission commit.

### 2. Load the pickle and predict

The pickle is self-contained. It does not need the `h4d2/` package. Load it only from this trusted file (it rebuilds the model from embedded source on load).

```python
import joblib
import pandas as pd

model = joblib.load("h4d2_model.pkl")
df = pd.read_parquet("amsterdam_test.parquet")   # raw organiser format
out = model.predict(df, n=25)                    # n in 5, 25, 50, 100, 200
# Madrid: model.predict(pd.read_parquet("madrid_test.parquet"), n="madrid")
#         or n=None
out.to_csv("amsterdam_n25_predictions.csv", index=False)
```

`predict` returns a DataFrame with one row per pixel and columns `px_key`, `py_key`, `pred_class` (1 to 4). `model.available_n` lists the Amsterdam label budgets. A call takes about 25 s (Amsterdam) or 50 s (Madrid) on 16 CPUs. Pass `context=False` to use only the test pixels; the default uses the provided city file as unlabelled spatial context.

Input columns (raw test parquet, one row per pixel-year): `px_key`, `py_key`, `year`, and for each observation slot `s` = 1, 2, 3: `Blue_s`, `Green_s`, `Red_s`, `NIR_s`, `SWIR1_s`, `SWIR2_s`, `qa_valid_s`. `weighted_mean_year` is optional and is not used for prediction. Other columns are ignored.

Official test parquet files are not in this repo. They can be used when the organisers release them.

### 3. Predict with the package and write CSVs

This path uses the committed models in `h4d2/artifacts` (not the pickle):

```bash
export H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files
export H4D2_OUT=h4d2/artifacts

python -m h4d2 predict --city Amsterdam --n 25 --seed 42 \
  --input amsterdam_test.parquet --out amsterdam_n25_predictions.csv

python -m h4d2 predict --city Madrid \
  --input madrid_test.parquet --out madrid_predictions.csv
```

To write all five Amsterdam budgets plus Madrid in one go (bash):

```bash
H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files \
  scripts/submit_predict.sh amsterdam_test.parquet madrid_test.parquet out_dir
```

That writes `amsterdam_n5_predictions.csv`, `amsterdam_n25_predictions.csv`, `amsterdam_n50_predictions.csv`, `amsterdam_n100_predictions.csv`, `amsterdam_n200_predictions.csv` and `madrid_predictions.csv` into `out_dir`.

### 4. Reproduce the official h4d2 scores

This reruns the official harness on Amsterdam (notebook draw order, seed 42, 10 trials). It needs the organiser parquet files and the committed Stage 1 bundle.

```bash
export H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files
export H4D2_OUT=h4d2/artifacts
python -m h4d2 sweep --seeds 42 --trials 10 --rng-mode notebook
```

Those flags are also the defaults of `python -m h4d2 sweep`. Results go to `h4d2/artifacts/results/amsterdam_trials_notebook.csv` and `amsterdam_summary_notebook.csv`. This command scores **h4d2 only**. There is no repo command that also recomputes the organiser baseline and rewrites `official_and_seed1000.json` in one step. The published baseline numbers are already in that JSON file and in `docs/h4d2_method_declaration.md`.

## View the dashboard and model bench

A prebuilt static results page is committed at `h4d2/dashboard/index.html`. You can open that file in a browser with no server and no data.

To serve the live results dashboard and the model bench (this is the serve entry point; there is no `scripts/` serve script):

```bash
export H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files
python -m h4d2.app
```

Windows (PowerShell):

```powershell
$env:H4D_DATA_DIR = "C:\path\to\the\folder\with\the\two\parquet\files"
python -m h4d2.app
```

This binds to `127.0.0.1` only and, unless you pass `--no-browser`, opens the results page:

- Results dashboard: `http://127.0.0.1:8765`
- Model bench: `http://127.0.0.1:8765/bench`

Useful flags: `python -m h4d2.app --port 9000 --no-browser`. Uploads and bench runs are kept in `~/h4d2_bench` (set `H4D2_BENCH_HOME` to change that). The page is built from `h4d2/dashboard/template.html` plus `h4d2/artifacts/dashboard_payload.json`. Only upload `.pkl` files you trust: unpickling runs code.

The live helper defaults `H4D2_OUT` to `h4d2/artifacts`, so the committed payload is used if you do not set that variable. To rebuild the payload from the parquet files and committed models:

```bash
export H4D_DATA_DIR=/path/to/the/folder/with/the/two/parquet/files
export H4D2_OUT=h4d2/artifacts
python -m h4d2.dashboard_data
```

There is also `h4d2/Dashboard.ipynb`, which can show the same dashboard or run the model on an uploaded parquet. It is written for the hackathon hub (its data path points at `~/Building_Age_Classification/data`) and needs Jupyter plus `ipywidgets`, which are not in `requirements.txt`.

`README_bench.md` mentions `scripts/env_local.ps1`. That file is **not** in the repository. Set `H4D_DATA_DIR` yourself, as shown above.

## Repo layout

```
.                          Team-3/ (project folder)
├── README_pkl.md          pickle load and predict API
├── requirements.txt       pinned packages
├── h4d2_model.pkl         frozen model (SHA-256 as above)
├── docs/                  method declaration and written justification
├── h4d2/                  model package, CLI, dashboard and bench
│   ├── app.py             python -m h4d2.app  (dashboard + bench server)
│   ├── cli.py             python -m h4d2      (stage1, sweep, adapt, predict, madrid-cv)
│   ├── core.py            parquet loading, features, graph, CNN
│   ├── dashboard/         template.html and prebuilt index.html
│   ├── dashboard_data.py  rebuild dashboard_payload.json
│   ├── dashboard_live.py  render the dashboard page
│   ├── Dashboard.ipynb    hub notebook for the live dashboard
│   ├── artifacts/         committed Stage 1/2 models, official results, payload
│   └── research/          CNN checkpoints used when assembling Stage 1
└── scripts/
    ├── submit_predict.sh  write the official prediction CSVs
    └── build_pkl.py       rebuild h4d2_model.pkl (needs H4D_DATA_DIR)
```

## Troubleshooting

- `FileNotFoundError` for a parquet file: set `H4D_DATA_DIR` to the folder that contains `madrid_train.parquet` and `amsterdam_data.parquet`, and run from the project folder.
- `python -m h4d2` looks for models in `H4D2_OUT` (default `./outputs_h4d2`), not in `h4d2/artifacts`. Set `export H4D2_OUT=h4d2/artifacts` (or the PowerShell equivalent) to use the committed models.
- `ModuleNotFoundError: h4d2` or `torch`: activate the venv, install from `requirements.txt`, and run from the project folder (do not rely on `pip install -e .`).
- The dashboard opens but the bench cannot load Amsterdam or Madrid: `H4D_DATA_DIR` is unset or points at the wrong folder.
- Nothing listens on the URL: `python -m h4d2.app` binds only to `127.0.0.1`. Open it on the same machine, or use `--no-browser` and browse to `http://127.0.0.1:8765` yourself.
- Official test prediction files: the test parquets are not in this repo. Use them with the pickle `predict` API or `scripts/submit_predict.sh` when they are released.

## Team and licence

Team 3 (Madrid2Dam), Hack4dev 2026. No licence file is included.
