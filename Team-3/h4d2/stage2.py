"""Stage 2: adapt the Madrid models to a new city with n labelled pixels per class, then predict."""
from __future__ import annotations
import numpy as np, torch
from sklearn.metrics import f1_score
from . import core

LAM_FEAT = 2.0     # pull of the feature-LR toward the Madrid feature head
LAM_NET = 8.0      # pull of each CNN-embedding LR toward that CNN's Madrid head
W_FEAT = 1.0 / 3   # ensemble weight of the feature head (CNN heads share the rest equally)


def prepare(bundle, pix):
    """All representations of a pixel set: spatial graph, smoothed organiser features, CNN embeddings,
    and the Madrid heads expressed in each representation's standardised coordinates."""
    Wm = core.spatial_W(pix["px"], pix["py"])
    F = core.org_feats(pix["arr"], pix["obs"])
    XF = core.sfeat((F - bundle["f_mu"]) / bundle["f_sd"], Wm)
    reps = [dict(kind="feat", X=XF, W0=bundle["lr_W"], b0=bundle["lr_b"], lam=LAM_FEAT)]
    Xc = core.cnn_input(pix["arr"], Wm, bundle["c_mu"], bundle["c_sd"])
    for e in bundle["nets"]:
        net = core.Net(); net.load_state_dict(e["state"])
        E = (core.embed(net, Xc) - e["e_mu"]) / e["e_sd"]
        Wh = net.head.weight.detach().numpy().astype(np.float64)
        bh = net.head.bias.detach().numpy().astype(np.float64)
        reps.append(dict(kind="cnn", X=E, W0=Wh * e["e_sd"][None, :], b0=bh + Wh @ e["e_mu"], lam=LAM_NET))
    return dict(W=Wm, reps=reps, px=pix["px"], py=pix["py"])


def madrid_heads(prep):
    """Zero-shot: the Madrid heads unchanged."""
    return [(r["W0"], r["b0"]) for r in prep["reps"]]


def fit_heads(prep, sup, ysup):
    """Few-shot transfer: one prior-regularised LR per representation, fit on the labelled pixels."""
    return [core.fit_prior_lr(r["X"][sup], ysup, r["W0"], r["b0"], r["lam"]) for r in prep["reps"]]


def proba(prep, heads, sup=None, ysup=None, spatial=True):
    P = [core.softmax(r["X"] @ W.T + b) for r, (W, b) in zip(prep["reps"], heads)]
    Pf, Pn = P[0], (np.mean(P[1:], 0) if len(P) > 1 else P[0])
    P = W_FEAT * Pf + (1 - W_FEAT) * Pn
    return core.smooth(P, prep["W"], sup=sup, ysup=ysup) if spatial else P


def sweep(prep, y, shots=(5, 25, 50, 100, 200), seed=42, trials=10, log=print, rng_mode="per_n"):
    """Official protocol: for each n, several balanced supports; query = every other pixel."""
    rows = []
    nb = core.notebook_draws(y, seed, trials) if rng_mode == "notebook" else None
    for n in shots:
        f1s = []
        draws = nb[int(n)] if nb is not None else core.support_draws(y, n, seed, trials)
        for t, sup in enumerate(draws):
            P = proba(prep, fit_heads(prep, sup, y[sup]), sup, y[sup])
            q = np.ones(len(y), bool); q[sup] = False
            f = f1_score(y[q], P[q].argmax(1) + 1, average="macro")
            f1s.append(f)
            rows.append(dict(n=int(n), trial=t, seed=int(seed), rng_mode=rng_mode, f1=float(f)))
        log(f"  n={n}: macro-F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    return rows


def adapt(bundle, pool, n, seed=42, trial=0):
    """Draw the labelled support (n per class) from a labelled pool of the target city and fit the heads.
    Returns the Stage 2 model: heads + the support pixels and labels (reused at predict time)."""
    prep = prepare(bundle, pool)
    sup = core.support_draws(pool["y"], n, seed, trial + 1)[trial]
    heads = fit_heads(prep, sup, pool["y"][sup])
    return dict(version="h4d2-1", n=int(n), seed=int(seed), trial=int(trial), heads=heads,
                sup_px=pool["px"][sup], sup_py=pool["py"][sup], sup_y=pool["y"][sup])


def predict(bundle, test, stage2=None, context=None):
    """Predict age classes for the pixels in test.
    stage2=None: Madrid model (zero-shot).  context: optional extra pixels of the same city (e.g. the
    provided data file) used only as spatial neighbours; their labels are never read, except the
    stage-2 support labels, which are clamped during smoothing."""
    if context is None:
        pix, n_test = test, len(test["px"])
    else:
        pix, n_test = core.union_pixels(test, context)
    prep = prepare(bundle, pix)
    if stage2 is None:
        return proba(prep, madrid_heads(prep)).argmax(1)[:n_test] + 1
    key = {(a, b): i for i, (a, b) in enumerate(zip(pix["px"].tolist(), pix["py"].tolist()))}
    sup, ysup = [], []
    for a, b, c in zip(stage2["sup_px"].tolist(), stage2["sup_py"].tolist(), stage2["sup_y"].tolist()):
        if (a, b) in key:
            sup.append(key[(a, b)]); ysup.append(c)
    P = proba(prep, stage2["heads"], np.array(sup, int), np.array(ysup, int))
    return P.argmax(1)[:n_test] + 1
