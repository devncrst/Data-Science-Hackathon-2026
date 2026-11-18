"""make_maps.py - Amsterdam maps of OUR model (n = 25 labels per class) vs the real age classes.

Same style as the organisers' notebook maps: C1 black, C2 red, C3 green, C4 blue, grey = no building.
Three panels: predicted classes | actual classes | where the model is right or wrong.
The 100 labelled pixels used to adapt the model are left out, so the map shows only unseen pixels.
Usage: /shared/venv/bin/python make_maps.py [--n 25]
Output: /shared/collab/results/map_amsterdam_n{n}.png
"""
import argparse, pickle
import numpy as np
import lightgbm as lgb  # noqa: F401  (needed to unpickle the models)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import f1_score


class AmsterdamModel:
    """Same class as in run_pipeline.py (needed to load the saved Amsterdam models)."""
    def __init__(self, madrid, mean, std):
        self.madrid, self.mean, self.std = madrid, mean, std

    def predict_proba_scaled(self, X):
        return (self.ft.predict_proba(X) + self.lr.predict_proba(np.hstack([X, self.madrid.predict_proba(X)]))) / 2

    def predict_scaled(self, X):
        return self.classes_[self.predict_proba_scaled(X).argmax(1)]


ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=25)
ap.add_argument('--features', default='/shared/collab/features/features_v2_spatial.pkl')
ap.add_argument('--models', default='/shared/collab/final/models')
ap.add_argument('--out', default=None)
args = ap.parse_args()
out = args.out or f'/shared/collab/results/map_amsterdam_n{args.n}.png'

d = pickle.load(open(args.features, 'rb'))
X = np.asarray(d['X_amsterdam'], float)
if d.get('scaler') is not None:
    X = d['scaler'].inverse_transform(X)
X = (X - X.mean(0)) / (X.std(0) + 1e-9)
y = np.asarray(d['y_amsterdam']).astype(int)
ids = np.asarray(d['pixel_ids_amsterdam'], dtype=np.int64)
model = pickle.load(open(f'{args.models}/amsterdam_n{args.n}.pkl', 'rb'))
seed = model.get('seed', 42)
rng = np.random.default_rng(seed)                       # the same labelled pixels as the saved model
labelled = np.concatenate([rng.choice(np.where(y == c)[0], args.n, replace=False) for c in np.unique(y)])
show = np.setdiff1d(np.arange(len(y)), labelled)
pred = model['model'].predict_scaled(X[show])
f1 = f1_score(y[show], pred, average='macro')
print(f'{len(show):,} unseen Amsterdam pixels, macro-F1 {f1:.3f}')

x0, y0 = ids[:, 0].min(), ids[:, 1].min()
W, H = ids[:, 0].max() - x0 + 1, ids[:, 1].max() - y0 + 1


def grid(values):
    g = np.full((H, W), np.nan)
    g[ids[show, 1] - y0, ids[show, 0] - x0] = values
    return np.flipud(g)                                 # north up (py_key grows northwards)


cls_cmap = ListedColormap(['#000000', '#E3120B', '#1FA34A', '#1F5BD8'])
cls_cmap.set_bad('#B0B0B0')
ok_cmap = ListedColormap(['#E3120B', '#2B6CB0'])
ok_cmap.set_bad('#B0B0B0')
fig, axes = plt.subplots(1, 3, figsize=(24, 9))
panels = [(grid(pred), cls_cmap, 1, 4, f'Our model, {args.n} labels per class: predicted class'),
          (grid(y[show]), cls_cmap, 1, 4, 'Actual class'),
          (grid((pred == y[show]).astype(float)), ok_cmap, 0, 1,
           f'Right (blue) or wrong (red): {np.mean(pred == y[show]):.0%} right')]
for ax, (g, cmap, lo, hi, title) in zip(axes, panels):
    ax.imshow(g, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title, fontsize=15); ax.set_xticks([]); ax.set_yticks([])
axes[1].legend(handles=[Patch(color=c, label=l) for c, l in
                        zip(['#000000', '#E3120B', '#1FA34A', '#1F5BD8', '#B0B0B0'],
                            ['C1 before 1945', 'C2 1945-1984', 'C3 1984-2004', 'C4 after 2004', 'No building'])],
               loc='lower right', fontsize=12)
fig.suptitle(f'Amsterdam, unseen pixels: macro-F1 {f1:.3f} with {args.n} labelled pixels per class',
             fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Saved {out}')
