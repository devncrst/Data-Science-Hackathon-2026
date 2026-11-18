"""model_lead.py - Model lead experiments. Same data + scoring as the team (results.csv).
Setups: lgb_base (team, Madrid ~0.640), lgb_long, lgb_small, lgb_big, year_reg (predict the year, then cut per city).
Run from ~/Building_Age_Classification:
  /shared/venv/bin/python model_lead.py --only lgb_base
  nohup /shared/venv/bin/python model_lead.py > model_lead.log 2>&1 &
"""
import argparse, csv, os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

DATA    = '/shared/collab/preprocessed_data.pkl'
RESULTS = '/shared/collab/results.csv'
PARQUET = {'madrid': 'data/madrid_train.parquet', 'amsterdam': 'data/amsterdam_data.parquet'}
EVENTS  = {'madrid': 1960, 'amsterdam': 1945}
SEED, NS, REPEATS, JOBS = 42, [5, 25, 50, 100, 200], 5, 4

BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=JOBS)
SETUPS = {
    'lgb_base':  dict(BASE),
    'lgb_long':  dict(BASE, n_estimators=1000, learning_rate=0.03),
    'lgb_small': dict(BASE, num_leaves=31),
    'lgb_big':   dict(BASE, num_leaves=127),
    'year_reg':  dict(BASE),
}

ap = argparse.ArgumentParser()
ap.add_argument('--only', nargs='*', default=list(SETUPS))
ap.add_argument('--data', default=DATA)
args = ap.parse_args()

d = pickle.load(open(args.data, 'rb'))
sc = d['scaler']
Xm = sc.inverse_transform(d['X_madrid']); Xa = sc.inverse_transform(d['X_amsterdam'])
Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9); Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
ym, ya = d['y_madrid'].astype(int), d['y_amsterdam'].astype(int)
classes = np.unique(ym)
print(f'Data: Madrid {Xm.shape}, Amsterdam {Xa.shape}', flush=True)


def years_for(city, ids):
    px = pd.read_parquet(PARQUET[city], columns=['px_key', 'py_key', 'weighted_mean_year'])
    px = px.drop_duplicates(['px_key', 'py_key']).set_index(['px_key', 'py_key'])['weighted_mean_year']
    yr = px.reindex(pd.MultiIndex.from_arrays([ids[:, 0], ids[:, 1]])).values.astype(float)
    assert not np.isnan(yr).any(), f'{city}: some pixels have no year'
    return yr


def to_class(year, city):
    return pd.cut(year, bins=[-np.inf, EVENTS[city], 1984, 2004, np.inf], labels=[1, 2, 3, 4]).astype(int)


def few_shot(extra_a):
    Za = np.hstack([Xa, extra_a])
    res = {}
    for n in NS:
        s = []
        for r in range(REPEATS):
            rng = np.random.default_rng(SEED + r)
            tr = np.concatenate([rng.choice(np.where(ya == k)[0], n, replace=False) for k in classes])
            te = np.setdiff1d(np.arange(len(ya)), tr)
            clf = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Za[tr], ya[tr])
            s.append(f1_score(ya[te], clf.predict(Za[te]), average='macro'))
        res[n] = (float(np.mean(s)), float(np.std(s)))
        print(f'Amsterdam n={n:3d}        : {res[n][0]:.3f} ± {res[n][1]:.3f}', flush=True)
    return res


yr_m = yr_a = None
for name in args.only:
    t0 = time.time()
    params = SETUPS[name]
    print(f'[{name}] {params if name != "year_reg" else "LGBMRegressor on construction year"}', flush=True)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    f1m = []
    if name == 'year_reg':
        if yr_m is None:
            yr_m = years_for('madrid', d['pixel_ids_madrid'])
            yr_a = years_for('amsterdam', d['pixel_ids_amsterdam'])
            assert (to_class(yr_m, 'madrid') == ym).all(), 'Madrid years do not match labels'
        for tr, te in cv.split(Xm, ym):
            m = lgb.LGBMRegressor(**params).fit(Xm[tr], yr_m[tr])
            f1m.append(f1_score(ym[te], to_class(m.predict(Xm[te]), 'madrid'), average='macro'))
        print(f'Madrid CV macro-F1     : {np.mean(f1m):.3f} ± {np.std(f1m):.3f}', flush=True)
        final = lgb.LGBMRegressor(**params).fit(Xm, yr_m)
        pred_a = final.predict(Xa)
        zs = f1_score(ya, to_class(pred_a, 'amsterdam'), average='macro')
        print(f'Amsterdam zero-shot    : {zs:.3f}', flush=True)
        extra = ((pred_a - pred_a.mean()) / (pred_a.std() + 1e-9))[:, None]
    else:
        params = dict(params, class_weight='balanced')
        for tr, te in cv.split(Xm, ym):
            m = lgb.LGBMClassifier(**params).fit(Xm[tr], ym[tr])
            f1m.append(f1_score(ym[te], m.predict(Xm[te]), average='macro'))
        print(f'Madrid CV macro-F1     : {np.mean(f1m):.3f} ± {np.std(f1m):.3f}', flush=True)
        final = lgb.LGBMClassifier(**params).fit(Xm, ym)
        zs = f1_score(ya, final.predict(Xa), average='macro')
        print(f'Amsterdam zero-shot    : {zs:.3f}', flush=True)
        extra = final.predict_proba(Xa)
    res = few_shot(extra)
    ams_mean = float(np.mean([res[n][0] for n in NS]))
    print(f'Amsterdam mean (5 n\'s) : {ams_mean:.3f}    [baseline 0.596]   ({time.time() - t0:.0f}s)', flush=True)
    row = {'name': f'model_{name}', 'what_changed': f'model lead: {name}', 'n_features': Xm.shape[1],
           'madrid_f1': round(np.mean(f1m), 4), 'madrid_std': round(np.std(f1m), 4),
           'ams_zeroshot': round(zs, 4)}
    for n in NS:
        row[f'ams_n{n}'] = round(res[n][0], 4); row[f'ams_n{n}_std'] = round(res[n][1], 4)
    row['ams_mean'] = round(ams_mean, 4)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    print(f'Logged to {RESULTS}\n', flush=True)
