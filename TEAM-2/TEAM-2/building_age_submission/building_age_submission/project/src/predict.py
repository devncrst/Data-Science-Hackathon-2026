"""predict.py - apply our saved models to new pixels (e.g. the organisers' hidden test set).
Automatic switch: with no --models, a 120-feature file uses the main models (/shared/collab/final/models)
and a 60-feature file uses the backup models (/shared/collab/final_plain/models).
Usage: /shared/venv/bin/python predict.py --city amsterdam --n 25 --features <file.pkl or .csv> [--out predictions.csv]
Output: CSV with px_key, py_key (if available) and predicted_class (1-4).
"""
import argparse, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb  # noqa: F401  (needed to unpickle the models)


class AmsterdamModel:
    """Same class as in run_pipeline.py (needed to load the saved Amsterdam models)."""
    def __init__(self, madrid, mean, std):
        self.madrid, self.mean, self.std = madrid, mean, std

    def predict_proba_scaled(self, X):
        return (self.ft.predict_proba(X) + self.lr.predict_proba(np.hstack([X, self.madrid.predict_proba(X)]))) / 2

    def predict_scaled(self, X):
        return self.classes_[self.predict_proba_scaled(X).argmax(1)]


ap = argparse.ArgumentParser()
ap.add_argument('--city', choices=['madrid', 'amsterdam'], required=True)
ap.add_argument('--n', type=int, default=25, help='Amsterdam model to use: 5, 25, 50, 100 or 200')
ap.add_argument('--features', required=True)
ap.add_argument('--models', default=None,
                help='model folder; default: chosen automatically from the number of features '
                     '(120 = main version with neighbours, 60 = backup without)')
ap.add_argument('--out', default='predictions.csv')
args = ap.parse_args()

# ── Load the new pixels ─────────────────────────────────────────────────────────
MODEL_DIRS = {120: '/shared/collab/final/models', 60: '/shared/collab/final_plain/models'}


def names_in(folder):
    return list(pickle.load(open(f'{folder}/madrid_model.pkl', 'rb'))['feature_names'])


ids = None
if args.features.endswith('.csv'):
    df = pd.read_csv(args.features)
    if args.models is None:                       # main version if all its columns are there, else backup
        args.models = next((m for m in MODEL_DIRS.values() if set(names_in(m)) <= set(df.columns)), None)
        if args.models is None:
            raise SystemExit('The CSV has neither the 120 main nor the 60 backup feature columns.')
    names = names_in(args.models)
    X = df[names].to_numpy(float)
    if {'px_key', 'py_key'} <= set(df.columns):
        ids = df[['px_key', 'py_key']].to_numpy()
else:
    d = pickle.load(open(args.features, 'rb'))
    X = np.asarray(d[f'X_{args.city}'], float)
    if d.get('scaler') is not None:
        X = d['scaler'].inverse_transform(X)
    ids = d.get(f'pixel_ids_{args.city}')

if args.models is None:
    if X.shape[1] not in MODEL_DIRS:
        raise SystemExit(f'{X.shape[1]} features: expected 120 (main) or 60 (backup). Use --models.')
    args.models = MODEL_DIRS[X.shape[1]]
expected = len(names_in(args.models))
if X.shape[1] != expected:
    raise SystemExit(f'{X.shape[1]} features but the models in {args.models} expect {expected}.')
print(f'Models: {args.models} ({"main, with neighbours" if expected == 120 else "backup, no neighbours"})')
X = (X - X.mean(0)) / (X.std(0) + 1e-9)          # each city scaled on its own data, as in training
print(f'{len(X):,} pixels, {X.shape[1]} features ({args.city})')

# ── Predict ─────────────────────────────────────────────────────────────────────
if args.city == 'madrid':
    model = pickle.load(open(f'{args.models}/madrid_model.pkl', 'rb'))['model']
    pred = model.predict(X)
else:
    model = pickle.load(open(f'{args.models}/amsterdam_n{args.n}.pkl', 'rb'))['model']
    pred = model.predict_scaled(X)

out = pd.DataFrame({'predicted_class': np.asarray(pred).astype(int)})
if ids is not None:
    ids = np.asarray(ids)
    out.insert(0, 'py_key', ids[:, 1]); out.insert(0, 'px_key', ids[:, 0])
out.to_csv(args.out, index=False)
print(f'Saved {args.out}. Predicted class shares: '
      + ', '.join(f'C{c}: {s:.0%}' for c, s in out.predicted_class.value_counts(normalize=True).sort_index().items()))
