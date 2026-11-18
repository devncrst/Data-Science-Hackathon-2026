"""Stage 2 deliverable: adapt the Madrid model to Amsterdam with n labelled pixels per class.

Runs end to end with a fixed random seed. The sample size per class is the
--n parameter.

    python adapt_amsterdam.py --amsterdam ../data/amsterdam_data.parquet --n 25 --seed 42
    python adapt_amsterdam.py --amsterdam ../data/amsterdam_data.parquet --n 25 --test held_out.parquet

Without --test it scores on every Amsterdam pixel not drawn as a labelled
sample. With --test it predicts the pixels of that file (the provided
Amsterdam file is still needed for the labelled samples and for neighbourhoods).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

import bac

# Method per sample size, used when best_per_n.json is missing; these are the
# choices that won the selection run (evaluate_fewshot.py, seed 42).
DEFAULT = {5: ('blend', True), 25: ('stack_et', False), 50: ('stack_et', True),
           100: ('stack_et', False), 200: ('stack_et', False)}

ap = argparse.ArgumentParser()
ap.add_argument('--amsterdam', required=True)
ap.add_argument('--n', type=int, required=True, help='labelled Amsterdam pixels per class')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--out', default='outputs')
ap.add_argument('--method', default='auto', choices=['auto'] + bac.METHODS)
ap.add_argument('--align', default='auto', choices=['auto', 'on', 'off'])
ap.add_argument('--test', default=None, help='optional held-out parquet to predict')
args = ap.parse_args()
out = Path(args.out)
bundle = joblib.load(out / 'stage1_model.joblib')
norm = bundle['city_year_norm']

# ---------------------------------------------------------------- features
if args.test:
    # build test and provided pixels together, so test pixels get complete neighbourhoods
    print('Building features for the provided and held-out pixels...')
    df = bac.build_features([args.test, args.amsterdam], city_year_norm=norm)
    test_keys = pd.read_parquet(args.test, columns=['px_key', 'py_key']).drop_duplicates()
    df = df.merge(test_keys.assign(is_test=True), on=['px_key', 'py_key'], how='left')
    df['is_test'] = df['is_test'].fillna(False).astype(bool)
else:
    cache = out / f'features_amsterdam{"_norm" if norm else ""}.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        print('Building Amsterdam features...')
        df = bac.build_features([args.amsterdam], city_year_norm=norm)
        df.to_parquet(cache)
    df['is_test'] = False
city = df['city'].iloc[0]
X = bac.matrix(bundle, df)
y = df['age_class'].to_numpy()

# ---------------------------------------------------------------- method
method, align = DEFAULT.get(args.n, ('stack_et', False))
best = out / 'best_per_n.json'
if best.exists() and str(args.n) in (choice := json.load(open(best))):
    method, align = choice[str(args.n)]['method'], choice[str(args.n)]['align']
if args.method != 'auto':
    method = args.method
if args.align != 'auto':
    align = args.align == 'on'

# ---------------------------------------------- draw labelled pixels and adapt
pool = np.where(np.isfinite(y) & ~df['is_test'].to_numpy())[0]        # labelled, provided pixels only
rng = np.random.default_rng(args.seed)
sup = pool[bac.sample_support(y[pool].astype(int), args.n, rng)]
model = bac.Stage2(bundle, city, method, align, args.seed).fit(X[sup], y[sup].astype(int))
print(f'{city}: adapted with n = {args.n} per class ({len(sup)} labelled pixels), method = {method}, align = {align}')

# ---------------------------------------------------------------- predict
# with --test: the held-out pixels; otherwise every pixel not used for adaptation
query = df['is_test'].to_numpy() if args.test else ~np.isin(np.arange(len(df)), sup)
pred = model.predict(X[query])
res = df.loc[query, ['px_key', 'py_key']].assign(predicted_class=pred)
res.to_csv(out / f'predictions_n{args.n}.csv', index=False)
joblib.dump({'stage2': model, 'support_pixels': df.loc[sup, ['px_key', 'py_key']], 'n': args.n, 'seed': args.seed},
            out / f'stage2_model_n{args.n}.joblib', compress=3)

# score when true classes are known (always, unless the held-out file has no labels)
truth = y[query]
if np.isfinite(truth).any():
    ok = np.isfinite(truth)
    print(f'Macro F1 on {ok.sum():,} pixels not used for adaptation: '
          f'{f1_score(truth[ok].astype(int), pred[ok], average="macro"):.4f}')
    print(classification_report(truth[ok].astype(int), pred[ok], digits=4))
print(f'Saved {out}/predictions_n{args.n}.csv and {out}/stage2_model_n{args.n}.joblib')
