"""
transfer_v3.py  :  Stage 1 Madrid model + Stage 2 Amsterdam adaptation (team, with Claude)

    import transfer_v3 as tv3
    data  = tv3.load_data()                       # frozen v3 features
    stage1 = tv3.train_stage1(data)               # Random Forest on ALL Madrid pixels
    f1, model = tv3.adapt_to_amsterdam(data, stage1, n=25, seed=42, method="stack")

THE IDEA IN PLAIN WORDS
Stage 1 learns age from Madrid (lots of labels). Stage 2 gets only n labelled Amsterdam pixels per
class (the "support set"). It does NOT retrain from scratch. It learns a small correction on top of
what Madrid taught us:
  * The Madrid forest gives every Amsterdam pixel 4 probabilities (one per age class).
  * "stack" fits a strongly regularised logistic regression on [log Madrid probabilities + features]
    using only the support set. Regularisation (small C) stops it memorising 20 to 800 pixels.
  * "prototype" (official baseline): average support pixels per class, nearest average wins.
  * "blend": average of stack and prototype probabilities (two views, more stable at small n).
  * "finetune": retrain the forest on Madrid + the support pixels, support pixels weighted up so
    20 to 800 Amsterdam pixels are not drowned by 76,263 Madrid pixels.
  * "blend3": average of finetune, stack and prototype probabilities.

NO-LEAKAGE RULES ENFORCED HERE
  * Only support-set labels are used to adapt. Query labels are used ONLY in f1_score.
  * Amsterdam features are scaled with Amsterdam's own mean/std, computed WITHOUT labels
    (SCALING="own"). Set SCALING="madrid" if the organisers say unlabelled Amsterdam data is not allowed.
  * Regularisation C is chosen by cross-validation INSIDE the support set (method="stack_cv").
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

SEED = 42
CLASSES = np.array([1, 2, 3, 4])
SCALING = "own"            # "own" = Amsterdam scaled by its own statistics (no labels); "madrid" = Madrid scaler
RF_PARAMS = dict(n_estimators=300, class_weight="balanced", min_samples_leaf=2,
                 random_state=SEED, n_jobs=-1)

# NEIGHBOURHOOD FEATURES (adopted Wed 00:30 after notebook 09 + hidden-neighbourhood confirmation).
# Next to its own 132 features, every pixel gets the AVERAGE features of the other pixels in a 3x3 window
# (radius 1, 90 m) and a 7x7 window (radius 3, 210 m), plus the share of that window that has pixels.
# Satellite features only, never labels. Each city uses its own pixels as neighbours.
# Set NEIGHBOUR_RADII = () to get the previous 132-feature method (v1).
NEIGHBOUR_RADII = (1, 3)


def neighbour_means(F, ids, r):
    """Mean of each feature over the OTHER pixels in the (2r+1) x (2r+1) window around each pixel.
    Pixels with no neighbour keep their own values; the neighbour count is returned too."""
    key = ids[:, 0].astype(np.int64) * 10_000_000 + ids[:, 1].astype(np.int64)
    index = pd.Index(key)
    acc = np.zeros_like(F, dtype=float)
    cnt = np.zeros(len(F))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            j = index.get_indexer((ids[:, 0].astype(np.int64) + dx) * 10_000_000 + ids[:, 1].astype(np.int64) + dy)
            ok = j >= 0
            acc[ok] += F[j[ok]]
            cnt[ok] += 1
    out = F.astype(float).copy()
    has = cnt > 0
    out[has] = acc[has] / cnt[has][:, None]
    return out, cnt


def add_neighbourhood(F, ids, radii=NEIGHBOUR_RADII, n_own=None):
    """[own features | neighbour means r1 | share r1 | neighbour means r2 | share r2 ...].
    The rows passed in decide who counts as a neighbour. n_own: return only the first n_own rows
    (used at test time: test pixels first, then our own pixels of that city as extra neighbours)."""
    blocks = [np.asarray(F, dtype=float)]
    for r in radii:
        m, c = neighbour_means(np.asarray(F, dtype=float), ids, r)
        blocks += [m, c[:, None] / ((2 * r + 1) ** 2 - 1)]
    out = np.hstack(blocks)
    return out if n_own is None else out[:n_own]


def neighbour_names(own_cols, radii=NEIGHBOUR_RADII):
    """Column names that match add_neighbourhood's output order."""
    names = list(own_cols)
    for r in radii:
        w = 2 * r + 1
        names += [f"nb{w}x{w}_{c}" for c in own_cols] + [f"nb{w}x{w}_share"]
    return names


def load_data(feat_dir=".", radii=None):
    """v3 features (one row per pixel) + neighbourhood features (NEIGHBOUR_RADII) and the feature list.
    data["own_cols"] = the 132 own features (what features_v3 builds); data["cols"] = all model inputs."""
    radii = NEIGHBOUR_RADII if radii is None else radii
    own = json.load(open(f"{feat_dir}/feature_cols_v3.json"))
    m = pd.read_parquet(f"{feat_dir}/features_madrid_v3.parquet")
    a = pd.read_parquet(f"{feat_dir}/features_amsterdam_v3.parquet")
    ids_m = m[["px_key", "py_key"]].to_numpy(np.int64)
    ids_a = a[["px_key", "py_key"]].to_numpy(np.int64)
    return {"cols": neighbour_names(own, radii), "own_cols": own, "radii": tuple(radii),
            "Xm": add_neighbourhood(m[own].to_numpy(float), ids_m, radii), "ym": m["age_class"].to_numpy(int),
            "Xa": add_neighbourhood(a[own].to_numpy(float), ids_a, radii), "ya": a["age_class"].to_numpy(int),
            "ids_m": ids_m, "ids_a": ids_a}


def train_stage1(data):
    """Stage 1: Random Forest on ALL Madrid pixels. Returns the model plus Amsterdam inputs for stage 2."""
    rf = RandomForestClassifier(**RF_PARAMS).fit(data["Xm"], data["ym"])
    if SCALING == "own":
        scaler_a = StandardScaler().fit(data["Xa"])                # Amsterdam statistics, no labels
    else:
        scaler_a = StandardScaler().fit(data["Xm"])
    Za = scaler_a.transform(data["Xa"])
    LPa = np.log(rf.predict_proba(data["Xa"]) + 1e-4)            # Madrid model's view of each Amsterdam pixel
    return {"rf": rf, "scaler_a": scaler_a, "Za": Za, "LPa": LPa, "Zst": np.hstack([LPa, Za])}


def sample_support(ya, n, seed):
    """n labelled pixels per class (without replacement) = support; everything else = query."""
    smallest = int(min((ya == c).sum() for c in CLASSES))
    assert n <= smallest, f"n={n} per class is more than the smallest class has ({smallest} pixels)"
    rng = np.random.default_rng(seed)
    sup = np.concatenate([rng.choice(np.where(ya == c)[0], n, replace=False) for c in CLASSES])
    return sup, np.setdiff1d(np.arange(len(ya)), sup)


def _soft(d):
    d = d / np.median(d)
    e = np.exp(-4 * (d - d.min(1, keepdims=True)))
    return e / e.sum(1, keepdims=True)


def proto_proba(Z, ys, Zs_idx, Zq_idx):
    P = np.stack([Z[Zs_idx][ys == c].mean(0) for c in CLASSES])
    return _soft(np.sqrt(((Z[Zq_idx][:, None, :] - P[None]) ** 2).sum(2)))


def _c_for(n):
    """Fixed regularisation schedule: smaller n -> stronger regularisation (smaller C)."""
    return 0.02 if n <= 5 else 0.05 if n <= 25 else 0.1 if n <= 50 else 0.3


def _c_by_cv(X, y, grid=(0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0)):
    """Choose C by cross-validation inside the support set only."""
    k = min(5, int(np.bincount(y)[1:].min()))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    best, best_c = -1, grid[0]
    for c in grid:
        sc = [f1_score(y[va], LogisticRegression(C=c, max_iter=5000).fit(X[tr], y[tr]).predict(X[va]),
                       average="macro") for tr, va in skf.split(X, y)]
        if np.mean(sc) > best:
            best, best_c = np.mean(sc), c
    return best_c


def finetune_proba(data, sup, qry):
    """Retrain the forest on Madrid + Amsterdam support. Weight per support pixel = Madrid size / n x 0.0625."""
    n = len(sup)
    w = max(1.0, len(data["ym"]) / n * 0.0625)
    X = np.vstack([data["Xm"], data["Xa"][sup]])
    y = np.concatenate([data["ym"], data["ya"][sup]])
    sw = np.concatenate([np.ones(len(data["ym"])), np.full(n, w)])
    params = dict(RF_PARAMS, n_estimators=150)
    f = RandomForestClassifier(**params).fit(X, y, sample_weight=sw)
    return f.predict_proba(data["Xa"][qry]), f


def adapt_to_amsterdam(data, stage1, n, seed, method="stack"):
    """Stage 2 for one support sample. Returns (macro F1 on the query pixels, fitted adapter)."""
    ya = data["ya"]
    sup, qry = sample_support(ya, n, seed)
    ys = ya[sup]
    Zst = stage1["Zst"]
    model = None
    if method == "prototype":
        P = proto_proba(stage1["Za"], ys, sup, qry)
    elif method == "finetune":
        P, model = finetune_proba(data, sup, qry)
    elif method == "blend3":
        Pf, rf_ft = finetune_proba(data, sup, qry)
        lr = LogisticRegression(C=_c_for(n), max_iter=5000).fit(Zst[sup], ys)
        P = (Pf + lr.predict_proba(Zst[qry]) + proto_proba(stage1["Za"], ys, sup, qry)) / 3
        # everything needed to re-apply this Amsterdam model later (saved with joblib)
        model = {"method": "blend3", "n_per_class": n, "seed": seed,
                 "finetuned_rf": rf_ft, "stack_lr": lr,
                 "prototypes": np.stack([stage1["Za"][sup][ys == c].mean(0) for c in CLASSES]),
                 "support_index": sup}
    elif method in ("stack", "stack_cv", "blend"):
        C = _c_by_cv(Zst[sup], ys) if method == "stack_cv" else _c_for(n)
        model = LogisticRegression(C=C, max_iter=5000).fit(Zst[sup], ys)
        P = model.predict_proba(Zst[qry])
        if method == "blend":
            P = 0.5 * P + 0.5 * proto_proba(stage1["Za"], ys, sup, qry)
    else:
        raise ValueError(method)
    pred = CLASSES[P.argmax(1)]
    return f1_score(ya[qry], pred, average="macro"), model


def predict_proba_blend3(model, stage1, Xa_new):
    """Class probabilities of a saved blend3 Amsterdam model for new pixels (same 132 columns, same order)."""
    Za = stage1["scaler_a"].transform(Xa_new)
    LP = np.log(stage1["rf"].predict_proba(Xa_new) + 1e-4)
    Pf = model["finetuned_rf"].predict_proba(Xa_new)
    Ps = model["stack_lr"].predict_proba(np.hstack([LP, Za]))
    d = np.sqrt(((Za[:, None, :] - model["prototypes"][None]) ** 2).sum(2))
    return (Pf + Ps + _soft(d)) / 3


def predict_blend3(model, stage1, Xa_new):
    """Apply a saved blend3 Amsterdam model to new Amsterdam pixels (e.g. the organisers' test set)."""
    return CLASSES[predict_proba_blend3(model, stage1, Xa_new).argmax(1)]


# Spatial smoothing setting of the final method. Notebook 08 (Tue 17 Nov night) tested 6 variants with a rule
# fixed in advance: 5x5 (r=2) at strength 0.5 beat 3x3 in 10/10 seeds at n=25 and was confirmed on hidden
# neighbourhoods (Amsterdam average 0.601 -> 0.607, Madrid 0.609 -> 0.617). See PROJECT_LOG section 7i.
SMOOTH_ALPHA = 0.5
SMOOTH_RADIUS = 2          # 2 = 5x5 window (60 m each side); was 1 (3x3) before Tue 23:10


def smooth_proba(P, ids, alpha=SMOOTH_ALPHA, r=SMOOTH_RADIUS):
    """Spatial smoothing (part of the final method; validated in notebooks 03, 05 and 08).
    Buildings of the same era cluster in neighbourhoods, so each pixel's class probabilities are mixed
    with the average of its neighbours' probabilities in a (2r+1) x (2r+1) window:
        smoothed = (1 - alpha) * own + alpha * mean(neighbours)
    ids = array of (px_key, py_key) for the SAME pixels as P. Only predicted probabilities are used,
    never labels. Pixels with no neighbour in the set keep their own probabilities."""
    key = ids[:, 0].astype(np.int64) * 10_000_000 + ids[:, 1].astype(np.int64)
    index = pd.Index(key)
    acc, cnt = np.zeros_like(P), np.zeros(len(P))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            j = index.get_indexer((ids[:, 0].astype(np.int64) + dx) * 10_000_000 + ids[:, 1].astype(np.int64) + dy)
            ok = j >= 0
            acc[ok] += P[j[ok]]
            cnt[ok] += 1
    out = P.copy()
    has = cnt > 0
    out[has] = (1 - alpha) * P[has] + alpha * acc[has] / cnt[has][:, None]
    return out


def evaluate(data, stage1, method, shots=(5, 25, 50, 100, 200), trials=10):
    """Mean and std macro F1 per n over `trials` support samples (seeds 42, 43, ...)."""
    rows = []
    for n in shots:
        f = [adapt_to_amsterdam(data, stage1, n, SEED + t, method)[0] for t in range(trials)]
        rows.append({"method": method, "n_per_class": n, "f1_mean": np.mean(f), "f1_std": np.std(f)})
    return pd.DataFrame(rows)
