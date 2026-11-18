"""
features.py - GIS feature functions for preprocessed_data.pkl (notebook 3 output)
  load()             raw (un-scaled) X per city, y, names, grid keys
  add_change()       spectral indices for early (1984-2003) and late (2004+) periods + late-early change
  add_spatial()      mean of the 8 neighbouring pixels on the 30 m grid
  standardise_city() scale one city with its own mean/std (no labels used)
"""
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse

EPS = 1e-9
BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]


def load(path):
    d = pickle.load(open(path, "rb"))
    out = {"names": list(d["feature_names"])}
    sc = d.get("scaler")
    Xm0 = np.asarray(d["X_madrid"], dtype=np.float64)
    invert = sc is not None and abs(np.nanmean(Xm0)) < 0.5 and 0.5 < np.nanstd(Xm0) < 2
    for city in ("madrid", "amsterdam"):
        X = np.asarray(d[f"X_{city}"], dtype=np.float64)
        if invert:
            X = sc.inverse_transform(X)          # undo notebook 3's Madrid-fitted scaler
        out[f"X_{city}"] = X
        out[f"y_{city}"] = np.asarray(d[f"y_{city}"]).astype(int)
        out[f"xy_{city}"] = np.asarray(d[f"pixel_ids_{city}"], dtype=np.float64)
    return out


def _refl(v):
    # Landsat Collection 2 Level-2 digital numbers -> surface reflectance
    return v * 2.75e-5 - 0.2 if np.nanmedian(v) > 100 else v


def _nd(a, b):
    return (a - b) / (a + b + EPS)


def add_change(X, names):
    """NDVI/NDBI/MNDWI/BSI + band ratios for the early and late period, and late - early."""
    idx = {n: i for i, n in enumerate(names)}
    need = [f"{b}_{p}_mean" for b in BANDS for p in ("early", "late")]
    if not all(n in idx for n in need):
        print("add_change: early/late band means not found, skipped")
        return X, names, []
    per = {}
    for p in ("early", "late"):
        B = {b: _refl(X[:, idx[f"{b}_{p}_mean"]]) for b in BANDS}
        per[p] = {
            "NDVI": _nd(B["NIR"], B["Red"]),
            "NDBI": _nd(B["SWIR1"], B["NIR"]),
            "MNDWI": _nd(B["Green"], B["SWIR1"]),
            "BSI": _nd(B["SWIR1"] + B["Red"], B["NIR"] + B["Blue"]),
            "ratio_SWIR1_SWIR2": B["SWIR1"] / (B["SWIR2"] + EPS),
            "ratio_Red_Blue": B["Red"] / (B["Blue"] + EPS),
        }
    cols, new = [], []
    for k in per["early"]:
        for p in ("early", "late"):
            cols.append(per[p][k]); new.append(f"{k}_{p}")
        cols.append(per["late"][k] - per["early"][k]); new.append(f"chg_{k}")
    Xn = np.column_stack(cols)
    Xn[~np.isfinite(Xn)] = np.nan
    return np.hstack([X, Xn]), names + new, new


def add_spatial(X, xy, names, radius=1.5):
    """Mean of the 8 neighbours (grid keys differ by 1 per 30 m pixel)."""
    tree = cKDTree(xy)
    A = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    A = sparse.csr_matrix((np.ones_like(A.data), (A.row, A.col)), shape=(len(xy), len(xy)))
    A.setdiag(0)
    A.eliminate_zeros()
    deg = np.asarray(A.sum(1)).ravel()
    nb = A @ np.nan_to_num(X)
    nb = np.where(deg[:, None] > 0, nb / np.maximum(deg, 1)[:, None], X)
    new = [f"nb_{n}" for n in names]
    return np.hstack([X, nb]), names + new, new


def standardise_city(X):
    return (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + EPS)


def fill_nan(X):
    med = np.nanmedian(X, 0)
    med = np.where(np.isnan(med), 0, med)
    r, c = np.where(np.isnan(X))
    X[r, c] = med[c]
    return X


REASONS = {
    "NDVI": "vegetation; old blocks have mature trees, new sites lose green",
    "NDBI": "built-up surface; concrete and new materials read high",
    "MNDWI": "wetness; separates Amsterdam's damp surfaces from the material itself",
    "BSI": "bare soil; high at construction sites",
    "ratio_SWIR1_SWIR2": "material signature (brick/stone/metal) without overall brightness",
    "ratio_Red_Blue": "roof colour (red brick/tile vs grey concrete)",
}


def feature_table(added):
    rows = []
    for n in added:
        if n.startswith("nb_"):
            rows.append((n, "mean of 8 neighbouring 30 m pixels: a block shares one era; reduces mixed-pixel noise"))
            continue
        key = next((k for k in REASONS if k in n), "")
        why = REASONS.get(key, "")
        if n.startswith("chg_"):
            why = "late minus early: construction after 1984 (classes 3/4); city brightness cancels. " + why
        elif n.endswith("_early"):
            why = "1984-2003 period: " + why
        elif n.endswith("_late"):
            why = "2004+ period (current building): " + why
        rows.append((n, why))
    return rows
