"""spatial_block_cv.py - is our score inflated by spatial leakage? (organisers' pitfall 02: data leakage)

Our pipeline tests on randomly chosen pixels, so a test pixel often sits right next to a training pixel.
Here the map is cut into square blocks (default 34 x 34 pixels = about 1 km). Training and test pixels
come from DIFFERENT blocks, so the model is tested on whole areas it has never seen.

For each feature set (spatial = with neighbours, plain = without):
  1. Madrid: 5-fold CV with random folds vs block folds (same LightGBM as the final pipeline)
  2. Amsterdam: the final transfer method (ft+lr), labelled pool / test set split 50/50
     either by random pixels or by blocks; n labelled pixels per class drawn from the pool
Gap = random - block. A small gap means the scores are honest.

Usage:  /shared/venv/bin/python spatial_block_cv.py            (about 15-25 min, n_jobs 4)
        options: --block 34  --repeats 10  --n 5 25 50 100 200  --features spatial plain
Output: /shared/collab/results/spatial_block_cv.csv
"""
import argparse, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

FEATURES = {'spatial': '/shared/collab/features/features_v2_spatial.pkl',
            'plain': '/shared/collab/features/features_v2_plain.pkl'}
ap = argparse.ArgumentParser()
ap.add_argument('--features', nargs='+', default=['spatial', 'plain'])
ap.add_argument('--n', type=int, nargs='+', default=[5, 25, 50, 100, 200])
ap.add_argument('--repeats', type=int, default=10)
ap.add_argument('--block', type=int, default=34, help='block side in pixels (34 x 30 m = about 1 km)')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--jobs', type=int, default=4)
ap.add_argument('--out', default='/shared/collab/results/spatial_block_cv.csv')
args = ap.parse_args()

SEED = args.seed
MADRID_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1,
                     n_jobs=args.jobs)
FT_PARAMS = dict(MADRID_PARAMS, n_estimators=50, min_child_samples=3)
LR_PARAMS = dict(C=0.5, max_iter=3000, class_weight='balanced')
t0 = time.time()


def log(msg):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


def blocks_of(ids):
    ids = np.asarray(ids, dtype=np.int64)
    bx, by = ids[:, 0] // args.block, ids[:, 1] // args.block
    return bx * 1_000_000 + by


def adapt_and_score(madrid, X, y, pool, test, n, rng, classes):
    tr = np.concatenate([rng.choice(pool[y[pool] == c], min(n, (y[pool] == c).sum()), replace=False)
                         for c in classes])
    ft = lgb.LGBMClassifier(**FT_PARAMS).fit(X[tr], y[tr], init_model=madrid.booster_)
    lr = LogisticRegression(**LR_PARAMS).fit(np.hstack([X[tr], madrid.predict_proba(X[tr])]), y[tr])
    p = (ft.predict_proba(X[test]) + lr.predict_proba(np.hstack([X[test], madrid.predict_proba(X[test])]))) / 2
    return f1_score(y[test], ft.classes_[p.argmax(1)], average='macro')


rows = []
for feat in args.features:
    path = FEATURES.get(feat, feat)
    d = __import__('pickle').load(open(path, 'rb'))
    Xm, Xa = np.asarray(d['X_madrid'], float), np.asarray(d['X_amsterdam'], float)
    if d.get('scaler') is not None:
        Xm, Xa = d['scaler'].inverse_transform(Xm), d['scaler'].inverse_transform(Xa)
    Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
    Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
    ym, ya = d['y_madrid'].astype(int), d['y_amsterdam'].astype(int)
    if d.get('pixel_ids_madrid') is None or d.get('pixel_ids_amsterdam') is None:
        raise SystemExit(f'{path} has no pixel_ids_madrid / pixel_ids_amsterdam: cannot build blocks')
    gm, ga = blocks_of(d['pixel_ids_madrid']), blocks_of(d['pixel_ids_amsterdam'])
    classes = np.unique(ym)
    log(f'{feat}: Madrid {Xm.shape} in {len(np.unique(gm))} blocks, '
        f'Amsterdam {Xa.shape} in {len(np.unique(ga))} blocks')

    # 1. Madrid CV: random folds vs block folds
    order = np.random.default_rng(SEED).permutation(np.unique(gm))         # shuffle blocks before GroupKFold
    gm_shuffled = pd.Series(np.arange(len(order)), index=order)[gm].to_numpy()
    for split, folds in [('random', StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xm, ym)),
                         ('block', GroupKFold(5).split(Xm, ym, gm_shuffled))]:
        f1 = [f1_score(ym[te], lgb.LGBMClassifier(**MADRID_PARAMS).fit(Xm[tr], ym[tr]).predict(Xm[te]),
                       average='macro') for tr, te in folds]
        rows.append({'features': feat, 'test': 'Madrid CV', 'n_per_class': 'all', 'split': split,
                     'f1_mean': np.mean(f1), 'f1_std': np.std(f1)})
        log(f'{feat} | Madrid CV, {split:6s} folds: {np.mean(f1):.4f} ± {np.std(f1):.4f}')

    # 2. Amsterdam transfer: random vs block 50/50 pool/test split
    madrid = lgb.LGBMClassifier(**MADRID_PARAMS).fit(Xm, ym)
    ublocks = np.unique(ga)
    for n in args.n:
        res = {'random': [], 'block': []}
        for r in range(args.repeats):
            rng = np.random.default_rng(SEED + r)
            pool_r = rng.permutation(len(ya))[:len(ya) // 2]
            pool_b = np.where(np.isin(ga, rng.permutation(ublocks)[:len(ublocks) // 2]))[0]
            for split, pool in [('random', pool_r), ('block', pool_b)]:
                test = np.setdiff1d(np.arange(len(ya)), pool)
                res[split].append(adapt_and_score(madrid, Xa, ya, pool, test, n, rng, classes))
        for split, s in res.items():
            rows.append({'features': feat, 'test': 'Amsterdam', 'n_per_class': n, 'split': split,
                         'f1_mean': np.mean(s), 'f1_std': np.std(s)})
        log(f'{feat} | Amsterdam n={n:3d}: random {np.mean(res["random"]):.4f} | '
            f'block {np.mean(res["block"]):.4f} | gap {np.mean(res["random"]) - np.mean(res["block"]):+.4f}')

table = pd.DataFrame(rows)
table[['f1_mean', 'f1_std']] = table[['f1_mean', 'f1_std']].round(4)
table.to_csv(args.out, index=False)
wide = table.pivot_table(index=['features', 'test', 'n_per_class'], columns='split', values='f1_mean',
                         sort=False)
wide['gap'] = (wide['random'] - wide['block']).round(4)
print('\n' + wide.to_string())
for feat in args.features:
    a = table[(table.features == feat) & (table.test == 'Amsterdam')]
    print(f'{feat}: Amsterdam average  random {a[a.split == "random"].f1_mean.mean():.4f} | '
          f'block {a[a.split == "block"].f1_mean.mean():.4f}')
log(f'DONE. Saved {args.out}')
