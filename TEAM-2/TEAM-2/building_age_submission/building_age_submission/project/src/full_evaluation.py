"""Full evaluation of the final system: official scores, model families, diagnostics. Read-only for final/."""
import os, glob, hashlib, pickle, warnings, numpy as np, pandas as pd, lightgbm as lgb, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
OUT = '/shared/collab/results/evaluation'; os.makedirs(OUT, exist_ok=True)
md5 = lambda: {os.path.basename(f): hashlib.md5(open(f, 'rb').read()).hexdigest() for f in sorted(glob.glob('/shared/collab/final/models/*.pkl'))}
before = md5()
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, class_weight='balanced', random_state=42, verbose=-1, n_jobs=4)
FT = dict(P, n_estimators=50, min_child_samples=3); LR = dict(C=0.5, max_iter=3000, class_weight='balanced')
d = pickle.load(open('/shared/collab/features/features_v2_spatial.pkl', 'rb'))
sc = lambda X: (lambda Z: (Z - Z.mean(0)) / (Z.std(0) + 1e-9))(d['scaler'].inverse_transform(np.asarray(X, float)))
Xm, Xa = sc(d['X_madrid']), sc(d['X_amsterdam']); ym, ya = np.asarray(d['y_madrid']).astype(int), np.asarray(d['y_amsterdam']).astype(int)
C, NS = [1, 2, 3, 4], [5, 25, 50, 100, 200]; rows, cms = [], {}
SKF = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xm, ym))

def metrics(y, pred, proba):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=C, zero_division=0); wrong = pred != y
    m = dict(macro_F1=f1_score(y, pred, average='macro'), balanced_accuracy=balanced_accuracy_score(y, pred), accuracy=accuracy_score(y, pred),
             kappa_quadratic=cohen_kappa_score(y, pred, weights='quadratic'), roc_auc_ovr=roc_auc_score(y, proba, multi_class='ovr', average='macro', labels=C),
             errors_in_adjacent_era=(np.abs(pred - y) == 1)[wrong].mean())
    for i, c in enumerate(C): m['precision_c%d' % c], m['recall_c%d' % c], m['F1_c%d' % c] = p[i], r[i], f[i]
    return m

def add(name, ms, cm):
    rows.append({'set': name, **pd.DataFrame(ms).mean().to_dict(), 'macro_F1_std': np.std([m['macro_F1'] for m in ms])})
    cms[name] = cm / cm.sum(1, keepdims=True); print('  %-26s macro-F1 %.4f +- %.4f' % (name, rows[-1]['macro_F1'], rows[-1]['macro_F1_std']), flush=True)

print('=== Diagnostics on the final system (official protocol) ===', flush=True)
ms, cm, its = [], np.zeros((4, 4)), list(range(25, 401, 25))
for k, (tr, te) in enumerate(SKF):
    m = lgb.LGBMClassifier(**P).fit(Xm[tr], ym[tr]); pred = m.predict(Xm[te])
    ms.append(metrics(ym[te], pred, m.predict_proba(Xm[te]))); cm += confusion_matrix(ym[te], pred, labels=C)
    if k == 0:
        sub = np.random.default_rng(42).choice(tr, 20000, replace=False)
        curve_tr = [f1_score(ym[sub], m.predict(Xm[sub], num_iteration=i), average='macro') for i in its]
        curve_va = [f1_score(ym[te], m.predict(Xm[te], num_iteration=i), average='macro') for i in its]
add('Madrid 5-fold CV', ms, cm)
madrid = lgb.LGBMClassifier(**P).fit(Xm, ym); z = madrid.predict(Xa)
add('Amsterdam zero-shot', [metrics(ya, z, madrid.predict_proba(Xa))], confusion_matrix(ya, z, labels=C).astype(float))
for n in NS:
    ms, cm = [], np.zeros((4, 4))
    for r in range(10):
        rng = np.random.default_rng(42 + r); tr = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in C]); te = np.setdiff1d(np.arange(len(ya)), tr)
        ft = lgb.LGBMClassifier(**FT).fit(Xa[tr], ya[tr], init_model=madrid.booster_); lr = LogisticRegression(**LR).fit(np.hstack([Xa[tr], madrid.predict_proba(Xa[tr])]), ya[tr])
        pr = (ft.predict_proba(Xa[te]) + lr.predict_proba(np.hstack([Xa[te], madrid.predict_proba(Xa[te])]))) / 2; pred = ft.classes_[pr.argmax(1)]
        ms.append(metrics(ya[te], pred, pr)); cm += confusion_matrix(ya[te], pred, labels=C)
    add('Amsterdam n=%d' % n, ms, cm)
ev = pd.DataFrame(rows).set_index('set'); ams_avg = ev.loc[['Amsterdam n=%d' % n for n in NS], 'macro_F1'].mean()

print('=== Part B: model families (same 5 Madrid folds) ===', flush=True)
fam = {'Logistic regression': lambda: LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced'),
       'Decision tree': lambda: DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42),
       'Random forest': lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', n_jobs=4, random_state=42),
       'Neural network (MLP)': lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True, max_iter=200, random_state=42)}
comp = []
for name, make in fam.items():
    f = [f1_score(ym[te], make().fit(Xm[tr], ym[tr]).predict(Xm[te]), average='macro') for tr, te in SKF]
    zz = f1_score(ya, make().fit(Xm, ym).predict(Xa), average='macro'); comp.append([name, np.mean(f), np.std(f), zz]); print('  %-22s done' % name, flush=True)
comp.append(['LightGBM (ours)', ev.loc['Madrid 5-fold CV', 'macro_F1'], ev.loc['Madrid 5-fold CV', 'macro_F1_std'], ev.loc['Amsterdam zero-shot', 'macro_F1']])
comp = pd.DataFrame(comp, columns=['model', 'madrid_cv_macro_F1', 'std', 'amsterdam_zero_shot_macro_F1']).round(4)
best = comp.loc[(comp.madrid_cv_macro_F1 + comp.amsterdam_zero_shot_macro_F1).idxmax()]

A = pd.DataFrame({'Madrid CV': [0.618, ev.loc['Madrid 5-fold CV', 'macro_F1']], 'Zero-shot': [0.343, ev.loc['Amsterdam zero-shot', 'macro_F1']],
                  'n=25': [0.599, ev.loc['Amsterdam n=25', 'macro_F1']], 'n=200': [0.615, ev.loc['Amsterdam n=200', 'macro_F1']],
                  'Amsterdam avg': [0.596, ams_avg]}, index=['Organisers baseline', 'Ours (final)']).round(4)
OFFICIAL = {'Madrid 5-fold CV': 0.6797, 'Amsterdam zero-shot': 0.5263, 'Amsterdam n=5': 0.5887, 'Amsterdam n=25': 0.6600, 'Amsterdam n=50': 0.6784, 'Amsterdam n=100': 0.7005, 'Amsterdam n=200': 0.7160}
checks = ['  %-24s %.4f vs official %.4f  %s' % (k, ev.loc[k, 'macro_F1'], v, 'PASS' if abs(ev.loc[k, 'macro_F1'] - v) < 0.0006 else 'FAIL') for k, v in OFFICIAL.items()]
checks.append('  %-24s %.4f vs official %.4f  %s' % ('Amsterdam average', ams_avg, 0.6687, 'PASS' if abs(ams_avg - 0.6687) < 0.0006 else 'FAIL'))
after = md5(); checks.append('  FINAL MODELS UNCHANGED (%d files)' % len(after) if after == before and len(after) == 6 else '  WARNING: final models changed or missing')
key = ['macro_F1', 'macro_F1_std', 'balanced_accuracy', 'accuracy', 'kappa_quadratic', 'roc_auc_ovr', 'errors_in_adjacent_era']
pcl = [p + '%d' % c for c in C for p in ['precision_c', 'recall_c', 'F1_c']]
sp = '/shared/collab/results/spatial_cv_report.csv'; spatial = pd.read_csv(sp).to_string(index=False) if os.path.exists(sp) else '(spatial_cv_report.csv not found)'
L = ['MODEL PERFORMANCE SUMMARY - Building age classification, Madrid -> Amsterdam (macro-F1)', '',
     'A. BASELINE VS OURS', A.to_string(), '',
     'B. MODEL FAMILIES (same 5 Madrid folds; zero-shot = trained on all Madrid, tested on all Amsterdam)', comp.to_string(index=False),
     'BEST MODEL: %s (Madrid %.4f | zero-shot %.4f)' % (best.model, best.madrid_cv_macro_F1, best.amsterdam_zero_shot_macro_F1), '',
     'C. DIAGNOSTICS (official protocol; Amsterdam = mean of 10 draws)', ev[key].round(4).to_string(), '',
     'D. PER-CLASS PRECISION / RECALL / F1', ev[pcl].T.round(3).to_string(), '',
     'E. SPATIAL CHECKS', spatial, '', 'F. CHECKS'] + checks
open(OUT + '/MODEL_PERFORMANCE_SUMMARY.txt', 'w').write('\n'.join(L) + '\n'); print('\n' + '\n'.join(L))
ev.round(4).to_csv(OUT + '/evaluation_metrics.csv'); comp.to_csv(OUT + '/model_comparison.csv', index=False); A.to_csv(OUT + '/baseline_vs_ours.csv')
fig, ax = plt.subplots(2, 2, figsize=(10, 9))
for a, name in zip(ax.ravel(), ['Madrid 5-fold CV', 'Amsterdam zero-shot', 'Amsterdam n=25', 'Amsterdam n=200']):
    cmn = cms[name]; a.imshow(cmn, cmap='Blues', vmin=0, vmax=1); a.set_title(name, fontsize=11); a.set_xticks(range(4)); a.set_yticks(range(4))
    a.set_xticklabels(C); a.set_yticklabels(C); a.set_xlabel('predicted class'); a.set_ylabel('true class')
    for i in range(4):
        for j in range(4): a.text(j, i, '%.2f' % cmn[i, j], ha='center', va='center', color='white' if cmn[i, j] > 0.5 else 'black')
plt.suptitle('Confusion matrices (row-normalised = recall per class)'); plt.tight_layout(); plt.savefig(OUT + '/confusion_matrices.png', dpi=150)
plt.figure(figsize=(7, 4.5)); plt.plot(its, curve_tr, 'o-', label='train (20k sample)'); plt.plot(its, curve_va, 'o-', label='validation (unseen fold)')
plt.xlabel('number of trees'); plt.ylabel('macro-F1'); plt.title('Madrid training curve (fold 1)'); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(OUT + '/training_curve.png', dpi=150)
print('Saved in', OUT, ':', sorted(os.listdir(OUT)))
# END OF SCRIPT
