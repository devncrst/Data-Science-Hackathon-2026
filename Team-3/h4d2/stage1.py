"""Stage 1: everything learned from Madrid (plus unlabelled Amsterdam pixels for the domain-adversarial nets)."""
from __future__ import annotations
import time
import numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from . import core


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def train_cnn(XM, yM, XA, seed, dann=0.0, epochs=40, threads=8, log=print):
    """Train one CNN on Madrid.  dann > 0 adds a domain discriminator through a gradient-reversal layer,
    fed with unlabelled Amsterdam pixels, so the embedding cannot tell the two cities apart.
    Checkpoint = best macro-F1 on a 10% Madrid validation split."""
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(threads)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yM)); va, tr = idx[:7600], idx[7600:]
    net = core.Net()
    disc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(list(net.parameters()) + list(disc.parameters()), 1e-3, weight_decay=1e-3)
    cw = torch.tensor(len(yM) / (4 * np.bincount(yM, minlength=5)[1:]), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)
    bce = nn.BCEWithLogitsLoss()
    Xt, yt, XAt = torch.tensor(XM), torch.tensor(yM - 1), torch.tensor(XA)
    steps = epochs * (len(tr) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    best, best_state, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        net.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            xb = Xt[b] + 0.05 * torch.randn_like(Xt[b])
            opt.zero_grad()
            h = net.f(xb)
            loss = lossf(net.head(h), yt[b])
            if dann > 0:
                ia = torch.randint(0, len(XAt), (len(b),))
                ha = net.f(XAt[ia] + 0.05 * torch.randn_like(XAt[ia]))
                ramp = min(1.0, (ep * len(perm) + i) / (epochs * len(perm) * 0.3))
                hh = _GradReverse.apply(torch.cat([h, ha]), dann * ramp)
                dom = torch.cat([torch.zeros(len(b)), torch.ones(len(b))])
                loss = loss + bce(disc(hh).squeeze(1), dom)
            loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            f = f1_score(yM[va], net(Xt[va]).argmax(1).numpy() + 1, average="macro")
        log(f"  cnn seed={seed} dann={dann} ep={ep} madrid_val_f1={f:.4f} ({time.time()-t0:.0f}s)")
        if f > best:
            best, best_state = f, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, best


def fit_stage1(mad, ams_unlab, nets=((42, 0.0), (1, 0.0), (1, 0.3), (2, 0.3), (3, 0.3)),
               epochs=40, threads=8, pretrained=None, log=print):
    """mad / ams_unlab: dicts from core.load_pixels.  pretrained: optional list of state_dicts
    (same order as nets) to skip CNN training."""
    WM = core.spatial_W(mad["px"], mad["py"])
    # (a) organiser features + spatial context -> Madrid logistic regression (the 'feature' head)
    F = core.org_feats(mad["arr"], mad["obs"])
    f_mu, f_sd = F.mean(0), F.std(0) + 1e-12
    XF = core.sfeat((F - f_mu) / f_sd, WM)
    lr = LogisticRegression(max_iter=3000).fit(XF, mad["y"])
    # (b) CNNs on the full time series
    ch = core.channels(core.gapfill(mad["arr"])).reshape(-1, 12)
    c_mu, c_sd = ch.mean(0), ch.std(0)
    XM = core.cnn_input(mad["arr"], WM, c_mu, c_sd)
    XA = core.cnn_input(ams_unlab["arr"], core.spatial_W(ams_unlab["px"], ams_unlab["py"]), c_mu, c_sd)
    net_entries = []
    for k, (seed, dann) in enumerate(nets):
        if pretrained is not None:
            net = core.Net(); net.load_state_dict(pretrained[k]); val = float("nan")
        else:
            net, val = train_cnn(XM, mad["y"], XA, seed, dann, epochs, threads, log)
        E = core.embed(net, XM)
        net_entries.append(dict(seed=seed, dann=dann, val_f1=val, state=net.state_dict(),
                                e_mu=E.mean(0), e_sd=np.where(E.std(0) < 1e-6, 1.0, E.std(0))))
    return dict(version="h4d2-1", f_mu=f_mu, f_sd=f_sd,
                lr_W=lr.coef_.astype(np.float64), lr_b=lr.intercept_.astype(np.float64),
                c_mu=c_mu, c_sd=c_sd, nets=net_entries)


def save(bundle, path):
    torch.save(bundle, path)


def load(path):
    return torch.load(path, weights_only=False)
