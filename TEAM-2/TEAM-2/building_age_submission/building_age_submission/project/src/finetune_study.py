"""finetune_study.py - does Madrid really transfer, and which Madrid model fine-tunes best?"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/shared/collab/features/features_v2_spatial.pkl')
ap.add_argument('--repeats', type=int, default=5)
ap.add_argument('--jobs', type=int, default=4)
args = ap.parse_args()

SEED, NS = 42, [5, 25, 50, 100, 200]
BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=args.jobs)
MADRID = {'base': dict(BASE), 'small': dict(BASE, num_leaves=31), 'big': dict(BASE, num_leaves=127),
          'long': dict(BASE, n_estimators=1000, learning_rate=0.03)}
FT = dict(BASE, n_estimators=50, min_child_samples=3)
t0 = time.time()

d = pickle.load(open(args.data, 'rb'))
Xm, Xa = np.asarray(d['X_madrid'], float), np.asarray(d['X_amsterdam'], float)
if d.get('scaler') is not None:
    Xm, Xa = d['scaler'].inverse_transform(Xm), d['scaler'].inverse_transform(Xa)
Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
ym, ya = d['y_madrid'].astype(int), d['y_amsterdam'].astype(int)
classes = np.unique(ym)
print(f'Data {args.data}: Madrid {Xm.shape}, Amsterdam {Xa.shape}', flush=True)

models = {}
for k, p in MADRID.items():
    models[k] = lgb.LGBMClassifier(**p).fit(Xm, ym)
    print(f'Madrid model {k} trained ({time.time() - t0:.0f}s)', flush=True)
Za = np.hstack([Xa, models['base'].predict_proba(Xa)])


def ft(k, tr):
    return lgb.LGBMClassifier(**FT).fit(Xa[tr], ya[tr], init_model=models[k].booster_)


def scratch(trees, tr):
    return lgb.LGBMClassifier(**dict(FT, n_estimators=trees)).fit(Xa[tr], ya[tr])


rows = []
for n in NS:
    res = {}
    for r in range(args.repeats):
        rng = np.random.default_rng(SEED + r)
        tr = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in classes])
        te = np.setdiff1d(np.arange(len(ya)), tr)
        preds = {'scratch_50': scratch(50, tr).predict(Xa[te]), 'scratch_200': scratch(200, tr).predict(Xa[te])}
        p_ft = None
        for k in MADRID:
            m = ft(k, tr)
            preds[f'ft_{k}'] = m.predict(Xa[te])
            if k == 'base':
                p_ft = m.predict_proba(Xa[te])
        lr = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Za[tr], ya[tr])
        preds['ens_ft_lr'] = classes[(p_ft + lr.predict_proba(Za[te])).argmax(1)]
        for name, pr in preds.items():
            res.setdefault(name, []).append((f1_score(ya[te], pr, average='macro'),
                                             recall_score(ya[te], pr, average=None, labels=classes).min()))
    for name, s in res.items():
        s = np.array(s)
        rows.append({'method': name, 'n': n, 'f1_mean': s[:, 0].mean(), 'f1_std': s[:, 0].std(),
                     'min_recall': s[:, 1].mean()})
        print(f'{name:12s} n={n:3d}  F1 {s[:, 0].mean():.3f} ± {s[:, 0].std():.3f}  min recall {s[:, 1].mean():.2f}',
              flush=True)

df = pd.DataFrame(rows)
table = df.pivot(index='method', columns='n', values='f1_mean')
table['average'] = table.mean(axis=1)
table = table.sort_values('average', ascending=False).round(3)
print('\n===== AMSTERDAM MACRO-F1 =====')
print(table.to_string())
best_scratch = table.loc[['scratch_50', 'scratch_200']].max()
print(f"\nTRANSFER PROOF: fine-tuning from Madrid vs best Amsterdam-only model = "
      f"{table.loc['ft_base', 'average'] - best_scratch['average']:+.3f} on average "
      f"(n=5: {table.loc['ft_base', 5] - best_scratch[5]:+.3f}, n=25: {table.loc['ft_base', 25] - best_scratch[25]:+.3f})")
print(f'BEST: {table.index[0]} (average {table.iloc[0]["average"]:.3f}, n=25 {table.iloc[0][25]:.3f})'
      f'   [team best so far: ft_base 0.661 / n=25 0.663; baseline 0.596 / 0.599]   ({time.time() - t0:.0f}s)')
os.makedirs('/shared/collab/results', exist_ok=True)
tag = os.path.basename(args.data).replace('.pkl', '')
df.round(4).to_csv(f'/shared/collab/results/finetune_study_{tag}.csv', index=False)
print(f'Saved /shared/collab/results/finetune_study_{tag}.csv')
