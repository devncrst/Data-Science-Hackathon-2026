"""transfer_compare.py - Modelling support: which Amsterdam transfer method works best?"""
import argparse, os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument('--data', default='/shared/collab/preprocessed_data.pkl')
ap.add_argument('--repeats', type=int, default=5)
ap.add_argument('--jobs', type=int, default=4)
args = ap.parse_args()

SEED, NS = 42, [5, 25, 50, 100, 200]
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1,
              n_jobs=args.jobs)
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

mad = lgb.LGBMClassifier(**PARAMS).fit(Xm, ym)
P = mad.predict_proba(Xa)
print(f'Madrid model trained ({time.time() - t0:.0f}s)', flush=True)


def em_prior(P, iters=100):
    old = np.full(P.shape[1], 1 / P.shape[1]); new = old.copy()
    for _ in range(iters):
        Q = P * (new / old); Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
    return Q, new

P_em, mix = em_prior(P)
true_mix = np.bincount(ya, minlength=classes.max() + 1)[classes] / len(ya)
print('Amsterdam class mix  EM estimate: ' + ' '.join(f'C{c}={m:.2f}' for c, m in zip(classes, mix))
      + '   true: ' + ' '.join(f'C{c}={m:.2f}' for c, m in zip(classes, true_mix)), flush=True)


def score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average=None,
                                                                   labels=classes).min()


def proto(Xs, ys, Xq):
    pr = np.stack([Xs[ys == c].mean(0) for c in classes])
    dist = (Xq ** 2).sum(1)[:, None] - 2 * Xq @ pr.T + (pr ** 2).sum(1)[None, :]
    return classes[dist.argmin(1)]


def lr(Z, tr, te):
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Z[tr], ya[tr])
    return m.predict(Z[te])


Z_p, Z_em = np.hstack([Xa, P]), np.hstack([Xa, P_em])
Z_knn = np.hstack([Xa, 5 * P])
METHODS = {
    'proto':       lambda tr, te: proto(Xa[tr], ya[tr], Xa[te]),
    'lr_proba':    lambda tr, te: lr(Z_p, tr, te),
    'lr_proba_em': lambda tr, te: lr(Z_em, tr, te),
    'knn_proba':   lambda tr, te: KNeighborsClassifier(5).fit(Z_knn[tr], ya[tr]).predict(Z_knn[te]),
    'finetune':    lambda tr, te: lgb.LGBMClassifier(**dict(PARAMS, n_estimators=50, min_child_samples=3))
                                     .fit(Xa[tr], ya[tr], init_model=mad.booster_).predict(Xa[te]),
}

rows = []
f1, rec = score(ya, mad.predict(Xa))
rows.append({'method': 'zero_shot', 'n': 0, 'f1_mean': f1, 'f1_std': 0.0, 'min_recall': rec})
f1, rec = score(ya, classes[P_em.argmax(1)])
rows.append({'method': 'em_prior', 'n': 0, 'f1_mean': f1, 'f1_std': 0.0, 'min_recall': rec})
for r in rows:
    print(f"{r['method']:12s} n=  0  F1 {r['f1_mean']:.3f}  min class recall {r['min_recall']:.2f}", flush=True)

for n in NS:
    res = {k: [] for k in METHODS}
    for r in range(args.repeats):
        rng = np.random.default_rng(SEED + r)
        tr = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in classes])
        te = np.setdiff1d(np.arange(len(ya)), tr)
        for name, fn in METHODS.items():
            res[name].append(score(ya[te], fn(tr, te)))
    for name, s in res.items():
        s = np.array(s)
        rows.append({'method': name, 'n': n, 'f1_mean': s[:, 0].mean(), 'f1_std': s[:, 0].std(),
                     'min_recall': s[:, 1].mean()})
        print(f'{name:12s} n={n:3d}  F1 {s[:, 0].mean():.3f} ± {s[:, 0].std():.3f}  '
              f'min class recall {s[:, 1].mean():.2f}', flush=True)

df = pd.DataFrame(rows)
table = df[df.n > 0].pivot(index='method', columns='n', values='f1_mean')
table['mean_5'] = table.mean(axis=1)
table = table.sort_values('mean_5', ascending=False).round(3)
print('\n===== MACRO-F1 BY METHOD (rows) AND n PER CLASS (columns) =====')
print(table.to_string())
print('\nBaseline to beat (organisers): n=5 0.544 | n=25 0.599 | n=200 0.615 | mean 0.596')
print('Team so far (lr_proba): mean 0.644 plain features, 0.646 with spatial | n=25 0.631 / 0.635')
best = table.index[0]
print(f'\nRECOMMENDATION: best average = {best} (mean {table.loc[best, "mean_5"]:.3f}); '
      f'best at n=25 = {table[25].idxmax()} ({table[25].max():.3f}).  ({time.time() - t0:.0f}s)')
print('Check min class recall above: below ~0.2 means the method nearly ignores a class (collapse).')

os.makedirs('/shared/collab/results', exist_ok=True)
tag = os.path.basename(args.data).replace('.pkl', '')
df.round(4).to_csv(f'/shared/collab/results/transfer_comparison_{tag}.csv', index=False)
print(f'Saved /shared/collab/results/transfer_comparison_{tag}.csv')
