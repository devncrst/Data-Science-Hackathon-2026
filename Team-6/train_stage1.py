"""Stage 1: cross-validate and train the transferable model on Madrid only.

Two cross-validation schemes, both 5-fold and stratified by class:
  random   folds of randomly chosen pixels; the same protocol as the organisers'
           baseline, so the two numbers are directly comparable
  spatial  folds made of whole ~1 km blocks, so a test pixel's neighbours are
           never in the training set; a fairer estimate for areas the model
           has not seen

For each scheme it saves macro F1 (mean and standard deviation over folds)
and the confusion matrix proportions to madrid_cv.json, plus a confusion
matrix figure. It then trains the final model on all Madrid pixels.

    python train_stage1.py --madrid ../data/madrid_train.parquet --out outputs --trees 150 --lr 0.1
    python train_stage1.py --madrid ../data/madrid_train.parquet --out outputs --cv spatial --skip-final
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

import bac

matplotlib.use('Agg')

LABELS = ['1: pre-1960', '2: 1960-84', '3: 1984-2004', '4: 2004-24']   # Madrid's classes
SCHEME_NAMES = {'random': 'random pixel folds', 'spatial': '1 km spatial block folds'}

ap = argparse.ArgumentParser()
ap.add_argument('--madrid', required=True)
ap.add_argument('--out', default='outputs')
ap.add_argument('--cv', choices=['both', 'random', 'spatial'], default='both')
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--trees', type=int, default=400, help='boosting rounds')
ap.add_argument('--lr', type=float, default=0.05, help='learning rate')
ap.add_argument('--skip-final', action='store_true', help='keep the existing final model, only cross-validate')
ap.add_argument('--city-year-norm', action='store_true',
                help='normalise against all pixels of the city per year (needs organiser approval for Amsterdam)')
args = ap.parse_args()
bac.N_TREES, bac.LEARNING_RATE = args.trees, args.lr
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- features
cache = out / f'features_madrid{"_norm" if args.city_year_norm else ""}.parquet'
if cache.exists():
    df = pd.read_parquet(cache)
else:
    print('Building Madrid features...')
    df = bac.build_features([args.madrid], city_year_norm=args.city_year_norm)
    df.to_parquet(cache)
df = df[df['age_class'].notna()].reset_index(drop=True)
cols = bac.feature_columns(df)
city = df['city'].iloc[0]
X = df[cols].fillna(df[cols].median()).to_numpy(np.float32)
y = df['age_class'].to_numpy(int)
year = df['weighted_mean_year'].to_numpy()
print(f'{city}: {len(df):,} pixels, {len(cols)} features')


# ---------------------------------------------------------- cross-validation
def folds_for(scheme):
    if scheme == 'random':
        return RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=1, random_state=args.seed).split(X, y)
    blocks = bac.spatial_blocks(df)
    print(f'  {blocks.max() + 1:,} blocks of about 1 km')
    return StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(X, y, groups=blocks)


cv_path = out / 'madrid_cv.json'
results = json.load(open(cv_path)) if cv_path.exists() else {}
if 'macro_f1_mean' in results:          # file from the earlier single-scheme version: that was random folds
    results = {'random': results}

schemes = ['random', 'spatial'] if args.cv == 'both' else [args.cv]
for scheme in schemes:
    print(f'Cross-validation with {SCHEME_NAMES[scheme]}:')
    scores, counts = [], np.zeros((4, 4))
    for i, (train, test) in enumerate(folds_for(scheme), 1):
        model = bac.fit_bins(X[train], year[train], args.seed)
        pred = bac.bins_to_class_proba(bac.predict_bins(model, X[test]), city).argmax(axis=1) + 1
        scores.append(f1_score(y[test], pred, average='macro'))
        counts += confusion_matrix(y[test], pred, labels=[1, 2, 3, 4])
        print(f'  fold {i}/{args.folds}: macro F1 = {scores[-1]:.4f}')
    proportions = counts / counts.sum(axis=1, keepdims=True)
    print(f'  macro F1: {np.mean(scores):.4f} +/- {np.std(scores):.4f}')
    results[scheme] = {'macro_f1_mean': float(np.mean(scores)), 'macro_f1_std': float(np.std(scores)),
                       'scores': [float(s) for s in scores], 'folds': args.folds,
                       'confusion_matrix_proportions': proportions.tolist(), 'n_features': len(cols)}
    bac.plot_confusion(proportions, LABELS,
                       f'Madrid, {SCHEME_NAMES[scheme]}\n'
                       f'macro F1 {np.mean(scores):.3f} ± {np.std(scores):.3f}  (each row sums to 1)',
                       path=out / f'madrid_confusion_{scheme}.png')

results['block_size_m'] = bac.BLOCK_PIXELS * 30
json.dump(results, open(cv_path, 'w'), indent=2)
print(f'Saved {cv_path}')

# ---------------------------------------------------------------- final model
if not args.skip_final:
    print('Training final Stage 1 model on all Madrid pixels...')
    bundle = bac.make_bundle(df, args.seed)
    bundle['city_year_norm'] = args.city_year_norm
    joblib.dump(bundle, out / 'stage1_model.joblib', compress=3)
    print(f'Saved {out / "stage1_model.joblib"}')
