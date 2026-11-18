"""Building age classification: shared library.

Everything the scripts and the notebook need lives here, in five parts:

  1. Labels            age classes and the finer construction-year bins
  2. Features          one row of numbers per pixel, built from 42 years of Landsat data
  3. Stage 1           the year-bin model trained on Madrid only
  4. Stage 2           few-shot adaptation to a new city with n labelled pixels per class
  5. Evaluation tools  spatial blocks for cross-validation, confusion-matrix figure
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ======================================================================= settings
BANDS = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
INDICES = ['NDVI', 'NDBI', 'UI', 'MNDWI', 'BSI']
SERIES = BANDS + INDICES                 # the 11 yearly series every pixel gets
Y0, Y1 = 1984, 2025                      # fixed year axis, so every file gives the same feature columns
YEARS = np.arange(Y0, Y1 + 1)
BLUE_MAX = 15_000                        # Blue above this is undetected haze (Notebook 2)
CLASS_EVENTS = {'Amsterdam': 1945, 'Madrid': 1960}   # city-specific end of class 1

# Stage 1 predicts these finer construction-year bins instead of the four classes.
# Every class boundary of both cities (1945, 1960, 1984, 2004) is a bin edge,
# so any city's four classes are exact sums of bins.
BIN_EDGES = np.array([1900, 1920, 1945, 1960, 1970, 1984, 1994, 2004, 2014])
N_BINS = len(BIN_EDGES) + 1

DECADES = [(1984, 1993), (1994, 2003), (2004, 2013), (2014, 2025)]
RADII = (1, 3)                           # neighbourhood windows: 3x3 and 7x7 pixels
KEY = 10_000_000                         # packs (px_key, py_key) into one integer

N_TREES, LEARNING_RATE = 400, 0.05       # Stage 1 boosting size; scripts and notebook can override
N_JOBS = min(6, os.cpu_count() or 1)     # CPU threads; kept modest so a shared server stays responsive


# ============================================================== 1. labels
def assign_age_class(year, city):
    """Age class 1-4 from construction year, same rule as the organisers' pd.cut.

    Classes are (-inf, event], (event, 1984], (1984, 2004], (2004, inf), where
    the event year is 1960 for Madrid and 1945 for Amsterdam.
    """
    edges = [CLASS_EVENTS[city], 1984, 2004]
    out = np.digitize(year, edges, right=True) + 1.0
    out[~np.isfinite(year)] = np.nan
    return out


def year_bins(year):
    """Year bin 0..9 for each construction year."""
    return np.digitize(year, BIN_EDGES, right=True)


def bin_to_class(city):
    """For each year bin, the age class (0..3) it falls into in this city."""
    bin_upper_edges = np.append(BIN_EDGES, np.inf)
    return np.digitize(bin_upper_edges, [CLASS_EVENTS[city], 1984, 2004], right=True)


def bins_to_class_proba(p_bins, city):
    """Turn year-bin probabilities into this city's four class probabilities by adding bins up."""
    membership = np.zeros((N_BINS, 4))
    membership[np.arange(N_BINS), bin_to_class(city)] = 1.0
    return p_bins @ membership


# ============================================================ 2. features
def _read(path):
    """Read only the columns we use from one parquet file."""
    import pyarrow.parquet as pq
    need = (['city', 'year', 'px_key', 'py_key', 'coverage', 'weighted_mean_year']
            + [f'{b}_{s}' for b in BANDS for s in (1, 2, 3)] + [f'qa_valid_{s}' for s in (1, 2, 3)])
    have = pq.ParquetFile(path).schema.names
    return pd.read_parquet(path, columns=[c for c in need if c in have])


def load_cube(paths):
    """Read parquet files of ONE city into a (pixel, year, band) array.

    Each year's value is the median of that year's valid observations (the
    baseline takes the first valid one). An observation counts as valid if its
    qa_valid flag is set, every band is inside the sensor range and Blue is
    below the haze threshold. Missing years are filled by linear interpolation;
    years before the first or after the last observation take the nearest value.
    """
    df = pd.concat([_read(p) for p in paths], ignore_index=True)
    df = df.drop_duplicates(['px_key', 'py_key', 'year'])
    df = df[(df['year'] >= Y0) & (df['year'] <= Y1)].reset_index(drop=True)

    # integer pixel index (0..n_pix-1) and year index (0..41) for every row
    key = df['px_key'].to_numpy(np.int64) * KEY + df['py_key'].to_numpy(np.int64)
    pixel_idx, pixel_keys = pd.factorize(key)
    year_idx = df['year'].to_numpy(int) - Y0
    n_pix = len(pixel_keys)

    # which of the up-to-three observations in each row are usable
    valid = []
    for s in (1, 2, 3):
        if f'qa_valid_{s}' not in df:
            valid.append(np.zeros(len(df), bool))
            continue
        ok = df[f'qa_valid_{s}'].fillna(False).to_numpy(bool).copy()
        for b in BANDS:
            x = df[f'{b}_{s}'].to_numpy(float)
            ok &= np.isfinite(x) & (x > 0) & (x < 65535)
        ok &= df[f'Blue_{s}'].to_numpy(float) <= BLUE_MAX
        valid.append(ok)

    # median of the valid observations, per band
    cube = np.full((n_pix, len(YEARS), len(BANDS)), np.nan, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN rows give NaN, which is what we want
        for bi, b in enumerate(BANDS):
            obs = np.stack([np.where(valid[s - 1], df[f'{b}_{s}'].to_numpy(float), np.nan)
                            if f'{b}_{s}' in df else np.full(len(df), np.nan) for s in (1, 2, 3)], axis=1)
            cube[pixel_idx, year_idx, bi] = np.nanmedian(obs, axis=1)

    frac_valid = np.isfinite(cube[:, :, 0]).mean(axis=1)       # share of years actually observed
    for bi in range(len(BANDS)):
        cube[:, :, bi] = pd.DataFrame(cube[:, :, bi]).interpolate(axis=1, limit_direction='both').to_numpy()

    # one row of metadata per pixel
    meta = pd.DataFrame({'px_key': pixel_keys // KEY, 'py_key': pixel_keys % KEY,
                         'city': df['city'].iloc[0], 'frac_valid': frac_valid})
    for col in ('weighted_mean_year', 'coverage'):
        arr = np.full(n_pix, np.nan)
        if col in df:
            ok = df[col].notna().to_numpy()
            arr[pixel_idx[ok]] = df[col].to_numpy(float)[ok]
        meta[col] = arr
    return cube, meta


def spectral_series(cube):
    """Add the five indices to the six bands: returns a (pixel, year, 11) array."""
    b, g, r, n, s1, s2 = [cube[:, :, i] for i in range(6)]
    eps = 1e-6
    ndvi = (n - r) / (n + r + eps)                               # vegetation
    ndbi = (s1 - n) / (s1 + n + eps)                             # built-up
    ui = (s2 - n) / (s2 + n + eps)                               # urban
    mndwi = (g - s1) / (g + s1 + eps)                            # water
    bsi = ((s1 + r) - (n + b)) / ((s1 + r) + (n + b) + eps)      # bare soil
    return np.concatenate([cube, np.stack([ndvi, ndbi, ui, mndwi, bsi], axis=2)], axis=2)


def temporal_features(S):
    """Summarise each pixel's 42-year series, for each of the 11 series.

    Per series:
      mean, std                    overall level and variability
      d0..d3 mean and std          level and variability per decade
      late_minus_early             last decade minus first decade
      yoy_mean, yoy_std            year-to-year change
      jump_year, jump_size         the largest single-year change and when it happened
      slope                        straight-line trend
      brk_year, brk_step, brk_score   the best single step change: the year that best splits
                                   the series into a 'before' and an 'after' level, the size
                                   of that step, and how clear it is
    """
    T = S.shape[1]
    feats = {}
    centred_t = (np.arange(T) - (T - 1) / 2).astype(np.float32)
    splits = np.arange(3, T - 2)                       # candidate break positions, at least 3 years each side
    split_weight = np.sqrt(splits * (T - splits) / T).astype(np.float32)   # favours balanced splits
    rows = np.arange(S.shape[0])
    for k, name in enumerate(SERIES):
        s = S[:, :, k]
        sd = s.std(axis=1) + 1e-6
        feats[f'{name}_mean'] = s.mean(axis=1)
        feats[f'{name}_std'] = sd
        for d, (a, b) in enumerate(DECADES):
            part = s[:, a - Y0:b - Y0 + 1]
            feats[f'{name}_d{d}_mean'] = part.mean(axis=1)
            feats[f'{name}_d{d}_std'] = part.std(axis=1)
        feats[f'{name}_late_minus_early'] = feats[f'{name}_d3_mean'] - feats[f'{name}_d0_mean']

        diff = np.diff(s, axis=1)
        feats[f'{name}_yoy_mean'] = diff.mean(axis=1)
        feats[f'{name}_yoy_std'] = diff.std(axis=1)
        j = np.abs(diff).argmax(axis=1)
        feats[f'{name}_jump_year'] = YEARS[j + 1].astype(np.float32)
        feats[f'{name}_jump_size'] = diff[rows, j] / sd
        feats[f'{name}_slope'] = (s * centred_t).sum(axis=1) / (centred_t ** 2).sum()

        # mean before and after every candidate split, via cumulative sums
        cs = np.cumsum(s, axis=1)
        before = cs[:, splits - 1] / splits
        after = (cs[:, -1:] - cs[:, splits - 1]) / (T - splits)
        score = np.abs(after - before) * split_weight
        j = score.argmax(axis=1)
        feats[f'{name}_brk_year'] = YEARS[splits[j]].astype(np.float32)
        feats[f'{name}_brk_step'] = (after - before)[rows, j]
        feats[f'{name}_brk_score'] = score[rows, j] / sd
    return pd.DataFrame(feats)


def neighbourhood_features(F, meta):
    """Context from surrounding pixels: buildings in a neighbourhood tend to be built together.

    For level and step-change features: the average over the building pixels in
    a 3x3 and a 7x7 window. Plus building density in three window sizes,
    coverage and its local average, and distance to the densest part of the
    city relative to the city's median distance. Uses only neighbours'
    spectra and positions, never their labels.
    """
    # place pixels on a grid
    px = (meta['px_key'] - meta['px_key'].min()).to_numpy(int)
    py = (meta['py_key'] - meta['py_key'].min()).to_numpy(int)
    shape = (py.max() + 1, px.max() + 1)
    is_pixel = np.zeros(shape, np.float32)
    is_pixel[py, px] = 1.0

    out = {}
    share_in_window = {}                 # share of the window covered by building pixels
    for r in RADII + (7,):
        share_in_window[r] = uniform_filter(is_pixel, 2 * r + 1, mode='constant')
        out[f'nb{r}_density'] = share_in_window[r][py, px]

    def window_mean(values, r):
        """Average of `values` over building pixels in a (2r+1) x (2r+1) window."""
        grid = np.zeros(shape, np.float32)
        grid[py, px] = values
        return (uniform_filter(grid, 2 * r + 1, mode='constant') / np.maximum(share_in_window[r], 1e-9))[py, px]

    cols = [c for c in F.columns if c.endswith(('_mean', '_brk_year', '_brk_step', '_brk_score'))
            and '_yoy_' not in c]
    for c in cols:
        for r in RADII:
            out[f'{c}_nb{r}'] = window_mean(F[c].to_numpy(np.float32), r)

    if meta['coverage'].notna().any():
        out['coverage'] = meta['coverage'].to_numpy()
        for r in (3, 7):
            out[f'coverage_nb{r}'] = window_mean(meta['coverage'].fillna(0).to_numpy(np.float32), r)

    density_3km = uniform_filter(is_pixel, 101, mode='constant')    # 101 pixels is about 3 km
    cy, cx = np.unravel_index(density_3km.argmax(), shape)
    dist = np.hypot(py - cy, px - cx)
    out['dist_core'] = dist / (np.median(dist) + 1e-9)              # relative, so cities of different size compare
    return pd.DataFrame(out)


def build_features(paths, city_year_norm=False):
    """Full feature table for one city: one row per pixel.

    paths: one or more parquet files of the same city. To predict a held-out
        file, pass [test_file, provided_city_file] so neighbourhoods are complete.
    city_year_norm: express every series relative to the city's median in that
        year. It uses unlabelled pixels of the file, so keep it False unless
        the organisers confirm that is allowed.
    """
    cube, meta = load_cube(paths)
    S = spectral_series(cube)
    if city_year_norm:
        med = np.nanmedian(S, axis=0, keepdims=True)
        q75, q25 = np.nanpercentile(S, [75, 25], axis=0, keepdims=True)
        S = (S - med) / (q75 - q25 + 1e-6)
    F = temporal_features(S)
    F['frac_valid'] = meta['frac_valid'].to_numpy()
    F = pd.concat([F, neighbourhood_features(F, meta)], axis=1)
    city = meta['city'].iloc[0]
    head = meta[['px_key', 'py_key', 'city', 'weighted_mean_year']].copy()
    head['age_class'] = assign_age_class(meta['weighted_mean_year'].to_numpy(), city)
    return pd.concat([head, F.astype(np.float32)], axis=1)


META_COLS = ['px_key', 'py_key', 'city', 'weighted_mean_year', 'age_class']


def feature_columns(df):
    """All columns of a feature table except identifiers and labels."""
    return [c for c in df.columns if c not in META_COLS]


# ============================================================= 3. Stage 1
def make_model(seed=42):
    """The Stage 1 classifier: LightGBM, or scikit-learn's boosting if LightGBM is missing."""
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=N_TREES, learning_rate=LEARNING_RATE, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, min_child_samples=20,
                              reg_lambda=1.0, random_state=seed, n_jobs=N_JOBS, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=N_TREES, learning_rate=LEARNING_RATE, max_leaf_nodes=63,
                                              l2_regularization=1.0, random_state=seed)


def bin_weights(bins):
    """Sample weights that balance the year bins, so common bins are not over-predicted (capped at 10x)."""
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    w = counts.sum() / (np.maximum(counts, 1) * (counts > 0).sum())
    return np.minimum(w, 10.0)[bins]


def fit_bins(X, year, seed=42):
    """Train a Stage 1 model to predict the year bin from features."""
    bins = year_bins(year)
    return make_model(seed).fit(X, bins, sample_weight=bin_weights(bins))


def predict_bins(model, X):
    """Probabilities over all N_BINS year bins (zero for bins never seen in training)."""
    p = np.zeros((len(X), N_BINS))
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return p


def make_bundle(df, seed=42):
    """Train the final Stage 1 model on all Madrid pixels and keep what Stage 2 needs.

    The bundle holds the model, the feature order, Madrid medians (to fill
    missing values), a scaler fitted on Madrid, and Madrid's average feature
    vector per year bin (used by the alignment step in Stage 2).
    """
    cols = feature_columns(df)
    med = df[cols].median()
    X = df[cols].fillna(med).to_numpy(np.float32)
    year = df['weighted_mean_year'].to_numpy()
    bins = year_bins(year)
    bin_mean = np.vstack([X[bins == b].mean(axis=0) if (bins == b).any() else np.full(X.shape[1], np.nan)
                          for b in range(N_BINS)])
    return {'model': fit_bins(X, year, seed), 'features': cols, 'median': med.to_numpy(np.float32),
            'scaler': StandardScaler().fit(X), 'bin_mean': bin_mean,
            'bin_count': np.bincount(bins, minlength=N_BINS), 'source_city': df['city'].iloc[0]}


def matrix(bundle, df):
    """Feature matrix in the model's column order, missing values filled with Madrid medians."""
    X = np.empty((len(df), len(bundle['features'])), np.float32)
    for i, c in enumerate(bundle['features']):
        col = df[c].to_numpy(np.float32) if c in df.columns else np.full(len(df), np.nan, np.float32)
        X[:, i] = np.where(np.isfinite(col), col, bundle['median'][i])
    return X


def zero_shot(bundle, X, city):
    """The Madrid model applied directly, with its year bins regrouped at the target city's boundaries."""
    return bins_to_class_proba(predict_bins(bundle['model'], X), city).argmax(axis=1) + 1


# ============================================================= 4. Stage 2
def support_shift(bundle, X_sup, y_sup, city, n):
    """How far the target city's features sit from Madrid's, from the labelled pixels only.

    For each class: the mean of the labelled target pixels minus Madrid's mean
    for the same class (Madrid's year bins regrouped at the target city's
    boundaries). Comparing class by class means an unusual class mix in the
    sample does not distort the estimate. The result is averaged over classes
    and shrunk towards zero when n is small, because few pixels give a noisy
    estimate.
    """
    cls = bin_to_class(city)
    shifts = []
    for c in range(4):
        bins_of_class = (cls == c) & (bundle['bin_count'] > 0)
        if not bins_of_class.any() or not (y_sup == c + 1).any():
            continue
        madrid_mean = np.average(bundle['bin_mean'][bins_of_class], axis=0, weights=bundle['bin_count'][bins_of_class])
        shifts.append(X_sup[y_sup == c + 1].mean(axis=0) - madrid_mean)
    if not shifts:
        return np.zeros(X_sup.shape[1], np.float32)
    return (n / (n + 3.0)) * np.mean(shifts, axis=0)


METHODS = ['prototype', 'bias', 'stack_lr', 'stack_et', 'blend']


class Stage2:
    """Few-shot adaptation of the Madrid model to a new city.

    method:
      prototype  nearest class mean in feature space (the organisers' baseline; ignores Stage 1)
      bias       Stage 1 class probabilities, recalibrated per class
      stack_lr   regularised logistic regression on Stage 1 year-bin log-probabilities
      stack_et   extra-trees on the features plus Stage 1 log-probabilities
      blend      average of stack_lr and stack_et probabilities
    align: first subtract support_shift(), i.e. move the target city's features
      towards Madrid's, using the labelled pixels only.

    fit() and predict() are the normal interface. fit_embedded() and
    proba_embedded() take precomputed Stage 1 outputs, so an evaluation can
    reuse them across methods instead of recomputing them.
    """

    def __init__(self, bundle, city, method='stack_lr', align=True, seed=42):
        self.bundle, self.city, self.method, self.align, self.seed = bundle, city, method, align, seed

    def _embed(self, X):
        """Aligned features, Stage 1 year-bin log-probabilities, and class log-probabilities."""
        Xa = X - self.shift
        p = predict_bins(self.bundle['model'], Xa)
        return Xa, np.log(p + 1e-6), np.log(bins_to_class_proba(p, self.city) + 1e-6)

    def fit(self, X_sup, y_sup):
        n = int(np.bincount(y_sup.astype(int)).max())
        self.shift = support_shift(self.bundle, X_sup, y_sup, self.city, n) if self.align else 0.0
        return self.fit_embedded(X_sup, *self._embed(X_sup), y_sup)

    def fit_embedded(self, X_raw, Xa, L, A, y_sup):
        n = int(np.bincount(y_sup.astype(int)).max())
        self.classes_ = np.unique(y_sup)
        self.parts = {}
        if self.method == 'prototype':
            Z = self.bundle['scaler'].transform(X_raw)
            self.parts['proto'] = np.stack([Z[y_sup == c].mean(axis=0) for c in self.classes_])
        if self.method == 'bias':
            self.parts['bias'] = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(A, y_sup)
        if self.method in ('stack_lr', 'blend'):
            sc = StandardScaler().fit(L)
            # less regularisation as n grows: the model can trust more labels
            lr = LogisticRegression(C=float(np.clip(n / 50.0, 0.1, 3.0)), class_weight='balanced', max_iter=2000)
            self.parts['lr'] = (sc, lr.fit(sc.transform(L), y_sup))
        if self.method in ('stack_et', 'blend'):
            Z = np.hstack([self.bundle['scaler'].transform(Xa), L])
            self.parts['et'] = ExtraTreesClassifier(500, min_samples_leaf=2 if n >= 25 else 1, class_weight='balanced',
                                                    random_state=self.seed, n_jobs=N_JOBS).fit(Z, y_sup)
        return self

    def predict_proba(self, X):
        return self.proba_embedded(X, *self._embed(X))

    def proba_embedded(self, X_raw, Xa, L, A):
        if self.method == 'prototype':
            Z = self.bundle['scaler'].transform(X_raw)
            d = np.stack([np.linalg.norm(Z - p, axis=1) for p in self.parts['proto']], axis=1)
            e = np.exp(-(d - d.min(axis=1, keepdims=True)))
            return e / e.sum(axis=1, keepdims=True)
        if self.method == 'bias':
            return self.parts['bias'].predict_proba(A)
        out = []
        if 'lr' in self.parts:
            sc, lr = self.parts['lr']
            out.append(lr.predict_proba(sc.transform(L)))
        if 'et' in self.parts:
            out.append(self.parts['et'].predict_proba(np.hstack([self.bundle['scaler'].transform(Xa), L])))
        return np.mean(out, axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def sample_support(y, n, rng):
    """Indices of n labelled pixels per class, drawn at random (same protocol as Notebook 4)."""
    idx = [rng.choice(np.where(y == c)[0], min(n, int((y == c).sum())), replace=False) for c in np.unique(y)]
    return np.concatenate(idx)


# ===================================================== 5. evaluation tools
BLOCK_PIXELS = 33                        # 33 pixels x 30 m = about 1 km


def spatial_blocks(df, block=BLOCK_PIXELS):
    """Id of the ~1 km square block each pixel falls in.

    Cross-validation folds built from whole blocks keep neighbouring pixels on
    the same side of the split. Random folds put a pixel's neighbours in the
    training set, and since neighbours share construction history (and the
    neighbourhood features), random folds overstate accuracy on new areas.
    """
    bx = df['px_key'].to_numpy(np.int64) // block
    by = df['py_key'].to_numpy(np.int64) // block
    return pd.factorize(bx * KEY + by)[0]


GREEN, RED = '#1B5E20', '#8B0000'       # dark green = correct, dark red = errors


def colour_strength(value, correct):
    """How strongly to colour a confusion-matrix cell, from 0 (white) to 1 (full colour).

    Correct cells reach full dark green at 0.75; error cells reach full dark red
    at 0.35, because errors are rarely larger and would otherwise look pale.
    """
    return float(min(value / (0.75 if correct else 0.35), 1.0))


def plot_confusion(cm_prop, labels, title, path=None, ax=None):
    """Confusion matrix as a heatmap: correct predictions in dark green, errors in dark red.

    cm_prop: 4x4 array, rows = actual class, columns = predicted class, each
    row summing to 1. Colour strength follows the proportion in each cell.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    cm_prop = np.asarray(cm_prop, float)
    k = len(labels)
    white = np.ones(3)
    rgb = np.zeros((k, k, 3))
    strength = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            base = np.array(to_rgb(GREEN if i == j else RED))
            strength[i, j] = colour_strength(cm_prop[i, j], i == j)
            rgb[i, j] = white + (base - white) * strength[i, j]

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=200)
    ax.imshow(rgb)
    for i in range(k):
        for j in range(k):
            dark_cell = strength[i, j] > 0.55
            ax.text(j, i, f'{cm_prop[i, j]:.2f}', ha='center', va='center', fontsize=11,
                    fontweight='bold' if i == j else 'normal', color='white' if dark_cell else '#1a1a1a')
    ax.set_xticks(range(k), labels, fontsize=8.5)
    ax.set_yticks(range(k), labels, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, k), minor=True)
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='both', length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel('Predicted class', fontsize=10)
    ax.set_ylabel('Actual class', fontsize=10)
    ax.set_title(title, fontsize=10.5, loc='left')
    ax.legend(handles=[Patch(color=GREEN, label='Correct (diagonal)'), Patch(color=RED, label='Error')],
              loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    if own_figure:
        fig.tight_layout()
        if path:
            fig.savefig(path)
        return fig
