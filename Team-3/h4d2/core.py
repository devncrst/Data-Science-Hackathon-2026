"""Shared building blocks: parquet -> annual time series, features, spatial graph, CNN, prior-LR, smoothing."""
from __future__ import annotations
import numpy as np, pandas as pd, scipy.sparse as sp
import torch, torch.nn as nn
from scipy.spatial import cKDTree

BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
YEARS = np.arange(1984, 2026)
CLASS_EVENTS = {"Amsterdam": 1945, "Madrid": 1960}
BLUE_MAX = 15_000
RADIUS, SIGMA = 3.0, 1.5          # spatial graph: neighbours within 3 pixels (90 m), Gaussian weights
SMOOTH_ALPHA, SMOOTH_ITERS = 3.0, 10


def load_pixels(df, city):
    """Wide parquet (one row per pixel-year) -> per-pixel annual series (N, T, 6).
    Best QA-valid observation per year (obs_1 > obs_2 > obs_3), Blue <= 15000, no zero bands."""
    if isinstance(df, str):
        df = pd.read_parquet(df)
    out = df[["px_key", "py_key", "year"]].copy()
    out["wmy"] = df["weighted_mean_year"].values if "weighted_mean_year" in df.columns else np.nan
    for b in BANDS:
        col = np.full(len(df), np.nan)
        for s in (1, 2, 3):
            v = df["qa_valid_%d" % s].fillna(False).values.astype(bool)
            col = np.where(np.isnan(col) & v, df["%s_%d" % (b, s)].astype(float).values, col)
        out[b] = col
    out = out.dropna(subset=BANDS)
    out = out[(out.Blue <= BLUE_MAX) & (out[BANDS] > 0).all(1) & out.year.between(YEARS[0], YEARS[-1])]
    pix = out.groupby(["px_key", "py_key"]).agg(wmy=("wmy", "first")).reset_index()
    pix["pid"] = np.arange(len(pix))
    out = out.merge(pix[["px_key", "py_key", "pid"]], on=["px_key", "py_key"])
    arr = np.full((len(pix), len(YEARS), 6), np.nan, np.float32)
    arr[out.pid.values, out.year.values - YEARS[0], :] = out[BANDS].values.astype(np.float32)
    wmy = pix.wmy.values.astype(float)
    y = None
    if len(wmy) and np.isfinite(wmy).all():
        y = np.digitize(wmy, [CLASS_EVENTS[city], 1984, 2004], right=True) + 1   # same as pd.cut right-closed
    return dict(arr=arr, obs=~np.isnan(arr[:, :, 0]), y=y, wmy=wmy,
                px=pix.px_key.values.astype(np.int64), py=pix.py_key.values.astype(np.int64))


def union_pixels(a, b):
    """Pixels of a, followed by the pixels of b not already in a. Returns (union, n_a)."""
    ka = set(zip(a["px"].tolist(), a["py"].tolist()))
    keep = np.array([(x, y) not in ka for x, y in zip(b["px"].tolist(), b["py"].tolist())], bool)
    out = {k: np.concatenate([a[k], b[k][keep]]) for k in ("arr", "obs", "wmy", "px", "py")}
    out["y"] = None
    return out, len(a["px"])


def gapfill(arr):
    """Linear interpolation of missing years; edges take the nearest observed value."""
    N, T, C = arr.shape
    df = pd.DataFrame(arr.transpose(1, 0, 2).reshape(T, N * C))
    df = df.interpolate(axis=0, limit_direction="both")
    return df.values.reshape(T, N, C).transpose(1, 0, 2).astype(np.float32)


def channels(v):
    """12 channels: log surface reflectance (6 bands), NDVI/NDBI/UI/MNDWI/BSI, log brightness."""
    r = np.clip(v * 2.75e-5 - 0.2, 1e-3, None)
    B, G, R, N, S1, S2 = [r[..., i] for i in range(6)]
    e = 1e-6
    idx = [(N - R) / (N + R + e), (S1 - N) / (S1 + N + e), (S2 - N) / (S2 + N + e),
           (G - S1) / (G + S1 + e), ((S1 + R) - (N + B)) / ((S1 + R) + (N + B) + e)]
    return np.concatenate([np.log(r), np.stack(idx, -1), np.log(r.mean(-1))[..., None]], -1)


def org_feats(arr, obs):
    """The organiser's 60 per-pixel temporal statistics (NB3 3.6), on the gap-filled series."""
    v = gapfill(arr).astype(np.float64)
    B, G, R, N_, S1, S2 = [v[..., i] for i in range(6)]
    e = 1e-6
    idx = np.stack([(N_ - R) / (N_ + R + e), (S1 - N_) / (S1 + N_ + e), (S2 - N_) / (S2 + N_ + e),
                    (G - S1) / (G + S1 + e), ((S1 + R) - (N_ + B)) / ((S1 + R) + (N_ + B) + e)], -1)
    early, late = YEARS <= 2003, YEARS >= 2004
    d = np.diff(v, axis=1)
    F = [v.mean(1), v.std(1, ddof=1), idx.mean(1), idx.std(1, ddof=1),
         v[:, early].mean(1), v[:, early].std(1, ddof=1), v[:, late].mean(1), v[:, late].std(1, ddof=1),
         d.mean(1), d.std(1, ddof=1),
         obs[:, early].any(1)[:, None].astype(float), obs[:, late].any(1)[:, None].astype(float)]
    return np.hstack(F)


def spatial_W(px, py, radius=RADIUS, sigma=SIGMA):
    """Sparse symmetric neighbour graph on the 30 m grid (no self loops)."""
    p = np.c_[px, py].astype(float)
    n = len(p)
    pairs = cKDTree(p).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return sp.csr_matrix((n, n))
    dd = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
    w = np.exp(-dd ** 2 / (2 * sigma ** 2))
    i = np.r_[pairs[:, 0], pairs[:, 1]]
    j = np.r_[pairs[:, 1], pairs[:, 0]]
    return sp.coo_matrix((np.r_[w, w], (i, j)), shape=(n, n)).tocsr()


def sfeat(X, Wm):
    """Neighbourhood-averaged features (self weight 1)."""
    deg = np.asarray(Wm.sum(1)).ravel()
    return (X + Wm @ X) / (1 + deg[:, None])


def smooth(P, Wm, alpha=SMOOTH_ALPHA, iters=SMOOTH_ITERS, sup=None, ysup=None):
    """Label-propagation smoothing of class probabilities over the spatial graph.
    Known support labels (if given) are clamped to one-hot and spread to their neighbours."""
    Q = P.copy()
    eye = np.eye(P.shape[1])
    has_sup = sup is not None and len(sup) > 0
    if has_sup:
        Q[sup] = eye[np.asarray(ysup) - 1]
    deg = np.asarray(Wm.sum(1)).ravel()
    for _ in range(iters):
        Q = (P + alpha * (Wm @ Q)) / (1 + alpha * deg[:, None])
        if has_sup:
            Q[sup] = eye[np.asarray(ysup) - 1]
    return Q


# ---------------------------------------------------------------- CNN
def cnn_input(arr, Wm, mu, sd):
    """(N, 24, T): 12 standardised channels + their neighbourhood average."""
    z = (channels(gapfill(arr)) - mu) / sd
    N, T, C = z.shape
    deg = np.asarray(Wm.sum(1)).ravel()
    s = ((z.reshape(N, -1) + Wm @ z.reshape(N, -1)) / (1 + deg[:, None])).reshape(N, T, C)
    return np.concatenate([z, s], -1).transpose(0, 2, 1).astype(np.float32)


class Net(nn.Module):
    """Small 1-D CNN over the 42-year series; f() is the 64-d embedding, head() the Madrid classifier."""
    def __init__(self, cin=24, h=64, e=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(cin, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Conv1d(h, h, 5, padding=4, dilation=2), nn.BatchNorm1d(h), nn.ReLU(),
            nn.Conv1d(h, h, 3, padding=1), nn.BatchNorm1d(h), nn.ReLU(),
            nn.AdaptiveAvgPool1d(6), nn.Flatten(), nn.Dropout(0.3), nn.Linear(h * 6, e), nn.ReLU())
        self.head = nn.Linear(e, 4)

    def forward(self, x):
        return self.head(self.f(x))


@torch.no_grad()
def embed(net, X, bs=4096):
    net.eval()
    return torch.cat([net.f(torch.tensor(X[i:i + bs])) for i in range(0, len(X), bs)]).numpy()


# ---------------------------------------------------------------- transfer head
def fit_prior_lr(X, y, W0, b0, lam, iters=200):
    """Multinomial logistic regression on the labelled target pixels, L2-penalised toward the
    Madrid head (W0, b0).  lam -> 0: plain LR on the target shots; lam -> inf: Madrid head unchanged."""
    Xt = torch.tensor(np.ascontiguousarray(X), dtype=torch.float64)
    yt = torch.tensor(np.asarray(y) - 1)
    W = torch.tensor(np.ascontiguousarray(W0, dtype=np.float64), requires_grad=True)
    b = torch.tensor(np.ascontiguousarray(b0, dtype=np.float64), requires_grad=True)
    W0t = torch.tensor(np.ascontiguousarray(W0, dtype=np.float64))
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(Xt @ W.T + b, yt, reduction="sum") + lam * ((W - W0t) ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def support_draws(y, n, seed, trials):
    """Balanced supports: n pixels per class; one RNG stream per n, default_rng([seed, n])."""
    rng = np.random.default_rng([int(seed), int(n)])
    classes = np.unique(y)
    return [np.concatenate([rng.choice(np.where(y == c)[0], int(n), replace=False) for c in classes])
            for _ in range(trials)]


def notebook_draws(y, seed, trials=10, shots=(5, 10, 25, 50, 100, 200)):
    """Organiser NB4 draw order: ONE default_rng(seed); n outer (incl. n=10), trial inner, classes sorted."""
    rng = np.random.default_rng(int(seed))
    classes = np.unique(y)
    return {int(n): [np.concatenate([rng.choice(np.where(y == c)[0], min(int(n), int((y == c).sum())), replace=False)
                                     for c in classes]) for _ in range(trials)] for n in shots}
