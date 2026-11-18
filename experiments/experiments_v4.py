"""
experiments_v4.py : improvement experiments for notebook 08 (Jian + Claude, Tue 17 Nov night).

Everything here runs on the DRESS-REHEARSAL 80% training split (rehearsal/train_features_*.parquet).
The hidden 20% neighbourhoods are only touched in confirm_on_hidden(), once, for the final check.

One function, transfer_run(), implements the CURRENT frozen method (blend3) with switches for each idea.
With every switch off it reproduces transfer_v3.adapt_to_amsterdam exactly (checked in the notebook).

Ideas (letters as in the chat / PROJECT_LOG):
  D  align   : quantile-align each Amsterdam feature to Madrid's distribution (no labels) before the
               RANDOM FOREST parts see it (stage 1 probabilities + fine-tuned forest). Own scaling stays
               for the logistic regression and prototypes.
  I  year    : a Random Forest REGRESSOR trained on Madrid's weighted_mean_year predicts a construction
               year for every Amsterdam pixel. Years mean the same in every city (class boundaries do
               not: Madrid class 1 = before 1960, Amsterdam = before 1945), so the predicted year is
               added to the stack layer as a city-independent age signal. Amsterdam support pixels still
               only use their AGE CLASS label.
  H  weights : training weights for Madrid pixels: more trust for high building coverage, less for a
               year right on a class boundary. Weights only, never features.
  E  boost   : a HistGradientBoosting model trained on Madrid adds its probabilities to the stack layer.
  A  rf      : Random Forest settings that memorise neighbourhoods less (chosen by Madrid spatial CV).
  C  prior   : label-free one-step correction of the class mix, estimated from the unlabelled pixels.
  B  smooth  : smoothing variants (alpha, radius); pure post-processing.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import transfer_v3 as tv3

SEED, CLASSES = tv3.SEED, tv3.CLASSES
BOUNDARIES = {"Madrid": (1960, 1984, 2004), "Amsterdam": (1945, 1984, 2004)}
SMOOTH_VARIANTS = [(0.0, 1), (0.5, 1), (0.5, 2), (0.7, 1), (0.7, 2), (0.3, 2)]   # (alpha, radius); (0,1) = off
BASE_SMOOTH = (0.5, 1)                                                            # the frozen setting


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load(rehearsal_dir, data_dir):
    """80% training split of both cities + Madrid's year and coverage (from the raw file, 3 columns only)."""
    m = pd.read_parquet(f"{rehearsal_dir}/train_features_madrid.parquet")
    a = pd.read_parquet(f"{rehearsal_dir}/train_features_amsterdam.parquet")
    cols = [c for c in m.columns if c not in tv3_forbidden()]
    meta = (pd.read_parquet(f"{data_dir}/madrid_train.parquet", columns=["px_key", "py_key", "weighted_mean_year", "coverage"])
            .groupby(["px_key", "py_key"], as_index=False).first())
    m = m.merge(meta, on=["px_key", "py_key"], how="left")
    assert m["weighted_mean_year"].notna().all(), "missing year for some Madrid pixels"
    return {"cols": cols,
            "Xm": m[cols].to_numpy(float), "ym": m["age_class"].to_numpy(int),
            "year_m": m["weighted_mean_year"].to_numpy(float), "cov_m": m["coverage"].to_numpy(float),
            "ids_m": m[["px_key", "py_key"]].to_numpy(np.int64),
            "Xa": a[cols].to_numpy(float), "ya": a["age_class"].to_numpy(int),
            "ids_a": a[["px_key", "py_key"]].to_numpy(np.int64)}


def tv3_forbidden():
    import features_v3 as fv3
    return fv3.FORBIDDEN | {"weighted_mean_year", "coverage"}


# ----------------------------------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------------------------------
def madrid_weights(D):
    """Idea H. Fixed in advance: coverage weight (0.15 -> low, 1.0 -> high) x boundary weight
    (0.5 for a year exactly on a class boundary, rising to 1.0 at 5+ years away). Mean = 1."""
    cov = np.clip(D["cov_m"], 0.15, 1.0)
    dist = np.min([np.abs(D["year_m"] - b) for b in BOUNDARIES["Madrid"]], axis=0)
    w = cov * np.minimum(1.0, 0.5 + dist / 10.0)
    return w / w.mean()


class QuantileAligner:
    """Idea D. Maps each feature of the new city onto the training city's distribution:
    value -> its rank in the new city -> the training city's value at the same rank. No labels."""
    def __init__(self, X_source, X_target, n_q=1001):
        q = np.linspace(0, 100, n_q)
        self.src = np.percentile(X_source, q, axis=0)
        self.tgt = np.percentile(X_target, q, axis=0)
        self.q = q

    def transform(self, X):
        out = np.empty_like(X, dtype=float)
        for j in range(X.shape[1]):
            t, s = self.tgt[:, j], self.src[:, j]
            # np.interp needs increasing x; constant stretches are fine (it takes the first match)
            r = np.interp(X[:, j], t, self.q)        # rank in the target city (0..100)
            out[:, j] = np.interp(r, self.q, s)      # same rank in the source city
        return out


def balanced_weights(y):
    cnt = np.bincount(y, minlength=CLASSES.max() + 1)
    return len(y) / (len(CLASSES) * cnt[y])


def build_stage1(D, cfg):
    """Stage 1 for one configuration. cfg keys: align, year, weights, boost, rf_params."""
    rf_params = cfg.get("rf_params", tv3.RF_PARAMS)
    w_m = madrid_weights(D) if cfg.get("weights") else np.ones(len(D["ym"]))
    rf = RandomForestClassifier(**rf_params).fit(D["Xm"], D["ym"], sample_weight=w_m)
    aligner = QuantileAligner(D["Xm"], D["Xa"]) if cfg.get("align") else None
    Xa_rf = aligner.transform(D["Xa"]) if aligner else D["Xa"]
    scaler = StandardScaler().fit(D["Xa"])                       # own scaling (unchanged)
    Za = scaler.transform(D["Xa"])
    parts = [np.log(rf.predict_proba(Xa_rf) + 1e-4)]
    reg = hgb = None
    yr_mu = yr_sd = None
    if cfg.get("year"):
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        reg.fit(D["Xm"], D["year_m"], sample_weight=w_m)
        yr = reg.predict(Xa_rf)
        yr_mu, yr_sd = yr.mean(), yr.std() + 1e-9                    # standardised on Amsterdam itself (no labels)
        parts.append(((yr - yr_mu) / yr_sd)[:, None])
    if cfg.get("boost"):
        hgb = HistGradientBoostingClassifier(random_state=SEED, max_iter=300, early_stopping=True)
        hgb.fit(D["Xm"], D["ym"], sample_weight=balanced_weights(D["ym"]) * w_m)
        parts.append(np.log(hgb.predict_proba(Xa_rf) + 1e-4))
    Zst = np.hstack(parts + [Za])
    return {"rf": rf, "w_m": w_m, "aligner": aligner, "Xa_rf": Xa_rf, "scaler": scaler, "Za": Za,
            "Zst": Zst, "reg": reg, "hgb": hgb, "yr_mu": yr_mu, "yr_sd": yr_sd, "cfg": cfg}


def stage1_transform(S, X_new):
    """Apply a fitted Stage 1 (aligner, scaler, Madrid models) to NEW Amsterdam pixels, e.g. the hidden test.
    Everything was fitted on training pixels only."""
    Xrf = S["aligner"].transform(X_new) if S["aligner"] else X_new
    Za = S["scaler"].transform(X_new)
    parts = [np.log(S["rf"].predict_proba(Xrf) + 1e-4)]
    if S["reg"] is not None:
        parts.append(((S["reg"].predict(Xrf) - S["yr_mu"]) / S["yr_sd"])[:, None])
    if S["hgb"] is not None:
        parts.append(np.log(S["hgb"].predict_proba(Xrf) + 1e-4))
    return Xrf, Za, np.hstack(parts + [Za])


def predict_new(S, parts, X_new):
    """blend3 probabilities for new pixels from one fitted transfer run (parts from transfer_run)."""
    Xrf, Za, Zst = stage1_transform(S, X_new)
    d = np.sqrt(((Za[:, None, :] - parts["protos"][None]) ** 2).sum(2))
    return (parts["ft"].predict_proba(Xrf) + parts["lr"].predict_proba(Zst) + tv3._soft(d)) / 3


def prior_correct(P):
    """Idea C. Label-free class-mix correction, ONE step: estimate the class mix of the unlabelled pixels as
    the share of each predicted class, then re-weight the probabilities by (estimated mix / training mix).
    Training mix = uniform (balanced class weights + balanced support).
    Note: the full EM version (Saerens et al. 2002) was tested first and collapsed to one or two classes,
    because blend3 probabilities are not calibrated. The one-step version cannot run away."""
    pi0 = np.full(P.shape[1], 1.0 / P.shape[1])
    pi = np.bincount(P.argmax(1), minlength=P.shape[1]) / len(P)
    pi = np.clip(pi, 0.02, None); pi /= pi.sum()                # never remove a class completely
    Q = P * (pi / pi0)
    return Q / Q.sum(1, keepdims=True), pi


def transfer_run(D, S, n, seed, return_parts=False):
    """blend3 for one support sample, exactly as transfer_v3 when every switch is off.
    Returns query indices and the blended probabilities of the query pixels."""
    cfg = S["cfg"]
    sup, qry = tv3.sample_support(D["ya"], n, seed)
    ys = D["ya"][sup]
    rf_params = dict(cfg.get("rf_params", tv3.RF_PARAMS), n_estimators=150)
    w = max(1.0, len(D["ym"]) / len(sup) * 0.0625)              # same weight rule as tv3.finetune_proba
    X = np.vstack([D["Xm"], S["Xa_rf"][sup]])
    y = np.concatenate([D["ym"], ys])
    sw = np.concatenate([S["w_m"], np.full(len(sup), w)])
    ft = RandomForestClassifier(**rf_params).fit(X, y, sample_weight=sw)
    Pf = ft.predict_proba(S["Xa_rf"][qry])
    lr = LogisticRegression(C=tv3._c_for(n), max_iter=5000).fit(S["Zst"][sup], ys)
    Ps = lr.predict_proba(S["Zst"][qry])
    Pp = tv3.proto_proba(S["Za"], ys, sup, qry)
    P = (Pf + Ps + Pp) / 3
    if return_parts:
        return sup, qry, P, {"ft": ft, "lr": lr, "protos": np.stack([S["Za"][sup][ys == c].mean(0) for c in CLASSES])}
    return sup, qry, P


def scores(P, y, ids, prior=False):
    """Macro F1 for every smoothing variant (with or without the class-mix correction)."""
    if prior:
        P = prior_correct(P)[0]
    out = {}
    for a, r in SMOOTH_VARIANTS:
        Ps = P if a == 0 else tv3.smooth_proba(P, ids, alpha=a, r=r)
        out[(a, r)] = f1_score(y, CLASSES[Ps.argmax(1)], average="macro")
    return out


# ----------------------------------------------------------------------------------------------
# Madrid cross-validation (random and spatial) for Stage 1 changes (ideas A, H, B on Madrid)
# ----------------------------------------------------------------------------------------------
def madrid_cv(D, rf_params, weights=False, block=50, k=5):
    """Returns dict with random and spatial CV F1 (mean, std) and the spatial out-of-fold probabilities."""
    X, y, ids = D["Xm"], D["ym"], D["ids_m"]
    w = madrid_weights(D) if weights else np.ones(len(y))
    res = {}
    for kind in ("random", "spatial"):
        if kind == "random":
            splits = StratifiedKFold(k, shuffle=True, random_state=SEED).split(X, y)
        else:
            groups = (ids[:, 0] // block) * 1_000_000 + ids[:, 1] // block
            splits = StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(X, y, groups)
        P = np.zeros((len(y), len(CLASSES)))
        f = []
        for tr, va in splits:
            rf = RandomForestClassifier(**rf_params).fit(X[tr], y[tr], sample_weight=w[tr])
            P[va] = rf.predict_proba(X[va])
            f.append(f1_score(y[va], CLASSES[P[va].argmax(1)], average="macro"))
        res[kind] = (np.mean(f), np.std(f))
        res[f"P_{kind}"] = P
    return res


# ----------------------------------------------------------------------------------------------
# Idea G (notebook 09): neighbourhood features
# ----------------------------------------------------------------------------------------------
def neighbour_means(F, ids, r):
    """For every pixel, the mean of each feature over the OTHER pixels in its (2r+1) x (2r+1) window.
    Uses satellite features only, never labels. Pixels with no neighbour get their own values, and the
    neighbour count is returned so the model can tell."""
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


def add_neighbourhood(F, ids, radii, n_own=None):
    """Own features + neighbour means for each radius + neighbour counts.
    The pixel set (F, ids) decides who counts as a neighbour: at training time the training pixels,
    at test time the test pixels plus our own pixels of that city (context), exactly like smoothing.
    n_own: return only the first n_own rows (the pixels we actually want)."""
    blocks = [F]
    for r in radii:
        m, c = neighbour_means(F, ids, r)
        blocks += [m, c[:, None] / ((2 * r + 1) ** 2 - 1)]
    out = np.hstack(blocks)
    return out if n_own is None else out[:n_own]
