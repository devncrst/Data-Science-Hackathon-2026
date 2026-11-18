"""run_pipeline.py - FINAL PIPELINE (Madrid -> Amsterdam). One command -> all deliverables.
Usage: /shared/venv/bin/python run_pipeline.py [--features spatial|plain] [--n 5 25 50 100 200] [--repeats 10]
Outputs in /shared/collab/final/: models/ (madrid + amsterdam_n*), results_table.csv,
madrid_confusion_matrix.csv, f1_vs_log2n.png, summary.json
Transfer: average of (a) fine-tuning the Madrid LightGBM with 50 trees on the Amsterdam examples and
(b) logistic regression on [Amsterdam features + Madrid probabilities]. Each city scaled on its own data."""
import argparse, json, os, pickle, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

FEATURES = {'spatial': '/shared/collab/features/features_v2_spatial.pkl',
            'plain':   '/shared/collab/features/features_v2_plain.pkl'}
ap = argparse.ArgumentParser(description='Final Madrid -> Amsterdam pipeline')
ap.add_argument('--features', default='spatial')
ap.add_argument('--n', type=int, nargs='+', default=[5, 25, 50, 100, 200])
ap.add_argument('--repeats', type=int, default=10)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--jobs', type=int, default=4)
ap.add_argument('--out', default='/shared/collab/final')
args = ap.parse_args()

SEED = args.seed
MADRID_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                     colsample_bytree=0.8, class_weight='balanced', random_state=SEED, verbose=-1,
                     n_jobs=args.jobs)
FT_PARAMS = dict(MADRID_PARAMS, n_estimators=50, min_child_samples=3)
LR_PARAMS = dict(C=0.5, max_iter=3000, class_weight='balanced')
t0 = time.time()
os.makedirs(f'{args.out}/models', exist_ok=True)


def log(msg):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


path = FEATURES.get(args.features, args.features)
d = pickle.load(open(path, 'rb'))
Xm, Xa = np.asarray(d['X_madrid'], float), np.asarray(d['X_amsterdam'], float)
if d.get('scaler') is not None:
    Xm, Xa = d['scaler'].inverse_transform(Xm), d['scaler'].inverse_transform(Xa)
scale = {c: (X.mean(0), X.std(0) + 1e-9) for c, X in [('madrid', Xm), ('amsterdam', Xa)]}
Xm = (Xm - scale['madrid'][0]) / scale['madrid'][1]
Xa = (Xa - scale['amsterdam'][0]) / scale['amsterdam'][1]
ym, ya = d['y_madrid'].astype(int), d['y_amsterdam'].astype(int)
classes = np.unique(ym)
log(f'Data {path}: Madrid {Xm.shape}, Amsterdam {Xa.shape}, {len(d["feature_names"])} features')


class AmsterdamModel:
    """Madrid model adapted to Amsterdam. predict() takes raw features in feature_names order."""
    def __init__(self, madrid, mean, std):
        self.madrid, self.mean, self.std = madrid, mean, std

    def fit_scaled(self, X, y):
        self.ft = lgb.LGBMClassifier(**FT_PARAMS).fit(X, y, init_model=self.madrid.booster_)
        self.lr = LogisticRegression(**LR_PARAMS).fit(np.hstack([X, self.madrid.predict_proba(X)]), y)
        self.classes_ = self.ft.classes_
        return self

    def predict_proba_scaled(self, X):
        return (self.ft.predict_proba(X) + self.lr.predict_proba(np.hstack([X, self.madrid.predict_proba(X)]))) / 2

    def predict_scaled(self, X):
        return self.classes_[self.predict_proba_scaled(X).argmax(1)]

    def predict(self, X_raw):
        return self.predict_scaled((np.asarray(X_raw, float) - self.mean) / self.std)


def sample(n, rng):
    return np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in classes])


cv = StratifiedKFold(args.folds, shuffle=True, random_state=SEED)
mad_f1, cm = [], np.zeros((len(classes), len(classes)))
for k, (tr, te) in enumerate(cv.split(Xm, ym), 1):
    m = lgb.LGBMClassifier(**MADRID_PARAMS).fit(Xm[tr], ym[tr])
    p = m.predict(Xm[te])
    mad_f1.append(f1_score(ym[te], p, average='macro'))
    cm += confusion_matrix(ym[te], p, labels=classes)
    log(f'Madrid fold {k}/{args.folds}: macro-F1 {mad_f1[-1]:.4f}')
cm_prop = pd.DataFrame(cm / cm.sum(1, keepdims=True), index=[f'true C{c}' for c in classes],
                       columns=[f'pred C{c}' for c in classes]).round(3)
log(f'Madrid CV macro-F1 = {np.mean(mad_f1):.4f} ± {np.std(mad_f1):.4f}')
print(cm_prop.to_string(), flush=True)
cm_prop.to_csv(f'{args.out}/madrid_confusion_matrix.csv')

madrid = lgb.LGBMClassifier(**MADRID_PARAMS).fit(Xm, ym)
pickle.dump({'model': madrid, 'feature_names': d['feature_names'], 'scale_mean': scale['madrid'][0],
             'scale_std': scale['madrid'][1], 'params': MADRID_PARAMS, 'features_file': path},
            open(f'{args.out}/models/madrid_model.pkl', 'wb'))
zero = f1_score(ya, madrid.predict(Xa), average='macro')
log(f'Saved models/madrid_model.pkl | Amsterdam zero-shot macro-F1 = {zero:.4f}')

rows = [{'dataset': 'Madrid (CV)', 'n_per_class': 'all', 'f1_mean': np.mean(mad_f1), 'f1_std': np.std(mad_f1)},
        {'dataset': 'Amsterdam zero-shot', 'n_per_class': 0, 'f1_mean': zero, 'f1_std': 0.0}]
for n in args.n:
    scores = []
    for r in range(args.repeats):
        rng = np.random.default_rng(SEED + r)
        tr = sample(n, rng)
        te = np.setdiff1d(np.arange(len(ya)), tr)
        am = AmsterdamModel(madrid, *scale['amsterdam']).fit_scaled(Xa[tr], ya[tr])
        scores.append(f1_score(ya[te], am.predict_scaled(Xa[te]), average='macro'))
    rows.append({'dataset': 'Amsterdam', 'n_per_class': n, 'f1_mean': np.mean(scores), 'f1_std': np.std(scores)})
    idx = sample(n, np.random.default_rng(SEED))
    final = AmsterdamModel(madrid, *scale['amsterdam']).fit_scaled(Xa[idx], ya[idx])
    pickle.dump({'model': final, 'n_per_class': n, 'seed': SEED, 'feature_names': d['feature_names']},
                open(f'{args.out}/models/amsterdam_n{n}.pkl', 'wb'))
    log(f'Amsterdam n={n:3d}: macro-F1 {np.mean(scores):.4f} ± {np.std(scores):.4f} '
        f'({args.repeats} draws) | saved models/amsterdam_n{n}.pkl')

table = pd.DataFrame(rows)
table[['f1_mean', 'f1_std']] = table[['f1_mean', 'f1_std']].round(4)
table.to_csv(f'{args.out}/results_table.csv', index=False)
ams = table[table.dataset == 'Amsterdam']
summary = {'features': path, 'n_features': len(d['feature_names']), 'seed': SEED, 'repeats': args.repeats,
           'madrid_f1': round(float(np.mean(mad_f1)), 4), 'madrid_std': round(float(np.std(mad_f1)), 4),
           'amsterdam_zero_shot': round(float(zero), 4),
           'amsterdam_average': round(float(ams.f1_mean.mean()), 4),
           'amsterdam_by_n': {int(r.n_per_class): [r.f1_mean, r.f1_std] for r in ams.itertuples()}}
json.dump(summary, open(f'{args.out}/summary.json', 'w'), indent=2)

ns = ams.n_per_class.astype(int).values
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(np.log2(ns), ams.f1_mean, yerr=ams.f1_std, fmt='o-', capsize=5, linewidth=2,
            color='#2b6cb0', label=f'Amsterdam, transfer from Madrid ({args.repeats} draws)')
ax.axhline(np.mean(mad_f1), linestyle='--', color='grey', label=f'Madrid CV = {np.mean(mad_f1):.3f}')
ax.axhline(zero, linestyle=':', color='grey', label=f'Amsterdam zero-shot = {zero:.3f}')
for x, yv in zip(np.log2(ns), ams.f1_mean):
    ax.annotate(f'{yv:.3f}', (x, yv), textcoords='offset points', xytext=(0, 9), ha='center')
ax.set_xticks(np.log2(ns)); ax.set_xticklabels([f'{n}\n(log2={np.log2(n):.1f})' for n in ns])
ax.set_xlabel('Labelled Amsterdam pixels per class (log2 scale)'); ax.set_ylabel('Macro F1')
ax.set_title('Macro F1 vs log2(sample size)', fontweight='bold'); ax.grid(alpha=0.3); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(f'{args.out}/f1_vs_log2n.png', dpi=200)

print('\n' + table.to_string(index=False))
log(f"DONE. Amsterdam average = {summary['amsterdam_average']:.4f} | outputs in {args.out}/")
