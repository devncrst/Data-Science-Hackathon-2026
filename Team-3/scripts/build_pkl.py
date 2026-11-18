"""Package the frozen h4d2 model (declared in 48a195f) into ONE self-contained pickle: h4d2_model.pkl.

Packaging only: no retraining, no tuning.  The pickle carries
  * the inference code (copied verbatim from h4d2/core.py, plus the stage-2 prepare/proba/predict logic),
  * the Stage 1 Madrid feature head (logistic regression), the six CNNs (CPU state dicts),
  * the five Stage 2 Amsterdam models (n = 5, 25, 50, 100, 200; seed 42),
  * the provided city files' pixel time series (labels removed) used as unlabelled spatial context,
so loading it needs only numpy / pandas / scipy / torch / joblib, NOT the h4d2 package.

Run on the hub:
    cd ~/Hack4dev-NCRST && H4D_DATA_DIR=~/Building_Age_Classification/data /shared/venv/bin/python scripts/build_pkl.py
"""
import os, sys, hashlib
from pathlib import Path
import numpy as np, torch, joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from h4d2 import core, stage1, stage2           # used only to READ the frozen artifacts

ART = REPO / "h4d2" / "artifacts"
DATA = Path(os.environ.get("H4D_DATA_DIR", "/home/participant-c1/Building_Age_Classification/data"))
OUT = REPO / "h4d2_model.pkl"
SHOTS = (5, 25, 50, 100, 200)

# ---------------------------------------------------------------- runtime source
core_src = (REPO / "h4d2" / "core.py").read_text(encoding="utf-8")
head, rest = core_src.split("# ---------------------------------------------------------------- transfer head")
softmax_src = rest[rest.index("def softmax("):rest.index("def support_draws(")]

MODEL_SRC = '''

# ---------------------------------------------------------------- h4d2 model object (stage2 logic, inference only)
_EXPR = "(exec(__src, __ns := {'__name__': 'h4d2_pkl_runtime'}), __ns['H4D2Model'](__state, __src))[1]"


class H4D2Model:
    """Frozen h4d2 (commit 48a195f).  predict(df, n) -> DataFrame(px_key, py_key, pred_class).

    df : raw test parquet as a pandas DataFrame (organiser wide format, one row per pixel-year).
    n  : 5, 25, 50, 100 or 200 -> Amsterdam Stage 2 model with n labelled pixels per class;
         None or "madrid"      -> Madrid Stage 1 model.
    context : True (default) uses the provided city file, stored inside this pickle without labels,
              as unlabelled spatial neighbours, exactly as scripts/submit_predict.sh does.
              False -> test pixels only.  A DataFrame -> use that file as context instead.
    """
    W_FEAT = 1.0 / 3

    def __init__(self, state, src):
        self.s = state
        self._src = src
        self._nets = None

    def __reduce__(self):
        return (eval, (_EXPR, {"__src": self._src, "__state": self.s}))

    @property
    def available_n(self):
        return sorted(self.s["stage2"])

    def _load_nets(self):
        if self._nets is None:
            nets = []
            for e in self.s["nets"]:
                net = Net()
                net.load_state_dict(e["state"])
                net.eval()
                nets.append(net)
            self._nets = nets
        return self._nets

    def _prepare(self, pix):
        s = self.s
        Wm = spatial_W(pix["px"], pix["py"])
        F = org_feats(pix["arr"], pix["obs"])
        XF = sfeat((F - s["f_mu"]) / s["f_sd"], Wm)
        reps = [(XF, s["lr_W"], s["lr_b"])]
        Xc = cnn_input(pix["arr"], Wm, s["c_mu"], s["c_sd"])
        for e, net in zip(s["nets"], self._load_nets()):
            E = (embed(net, Xc) - e["e_mu"]) / e["e_sd"]
            Wh = net.head.weight.detach().numpy().astype(np.float64)
            bh = net.head.bias.detach().numpy().astype(np.float64)
            reps.append((E, Wh * e["e_sd"][None, :], bh + Wh @ e["e_mu"]))
        return Wm, reps

    def _proba(self, Wm, reps, heads, sup=None, ysup=None):
        P = [softmax(X @ W.T + b) for (X, _, _), (W, b) in zip(reps, heads)]
        Pf, Pn = P[0], (np.mean(P[1:], 0) if len(P) > 1 else P[0])
        P = self.W_FEAT * Pf + (1 - self.W_FEAT) * Pn
        return smooth(P, Wm, sup=sup, ysup=ysup)

    def predict(self, df, n=None, context=True):
        madrid = n is None or (isinstance(n, str) and n.lower() == "madrid")
        city = "Madrid" if madrid else "Amsterdam"
        test = load_pixels(df, city)
        if context is True:
            ctx = self.s["context"][city]
        elif context is False or context is None:
            ctx = None
        else:
            ctx = load_pixels(context, city)
        if ctx is None:
            pix, n_test = test, len(test["px"])
        else:
            pix, n_test = union_pixels(test, ctx)
        Wm, reps = self._prepare(pix)
        if madrid:
            P = self._proba(Wm, reps, [(W0, b0) for (_, W0, b0) in reps])
        else:
            S = self.s["stage2"][int(n)]
            key = {(a, b): i for i, (a, b) in enumerate(zip(pix["px"].tolist(), pix["py"].tolist()))}
            sup, ysup = [], []
            for a, b, c in zip(S["sup_px"].tolist(), S["sup_py"].tolist(), S["sup_y"].tolist()):
                if (a, b) in key:
                    sup.append(key[(a, b)]); ysup.append(c)
            P = self._proba(Wm, reps, S["heads"], np.array(sup, int), np.array(ysup, int))
        pred = P.argmax(1)[:n_test] + 1
        return pd.DataFrame(dict(px_key=test["px"], py_key=test["py"], pred_class=pred))
'''

SRC = head + softmax_src + MODEL_SRC


class _Pack:
    """Dump-time wrapper: unpickles straight into an H4D2Model built from the embedded source."""
    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        expr = "(exec(__src, __ns := {'__name__': 'h4d2_pkl_runtime'}), __ns['H4D2Model'](__state, __src))[1]"
        return (eval, (expr, {"__src": SRC, "__state": self.state}))


def context_pixels(city, fname):
    p = core.load_pixels(str(DATA / fname), city)
    return dict(arr=p["arr"], obs=p["obs"], wmy=np.full(len(p["px"]), np.nan), px=p["px"], py=p["py"], y=None)


def main():
    B = stage1.load(ART / "stage1_madrid.pt")
    nets = [dict(e_mu=e["e_mu"], e_sd=e["e_sd"], seed=e["seed"], dann=e["dann"],
                 state={k: v.detach().cpu().clone() for k, v in e["state"].items()}) for e in B["nets"]]
    s2 = {}
    for n in SHOTS:
        S = torch.load(ART / "models" / f"stage2_amsterdam_n{n}_seed42.pt", weights_only=False)
        s2[n] = dict(heads=[(np.asarray(W), np.asarray(b)) for W, b in S["heads"]],
                     sup_px=S["sup_px"], sup_py=S["sup_py"], sup_y=S["sup_y"])
    state = dict(version="h4d2-1 (frozen, commit 48a195f)",
                 f_mu=B["f_mu"], f_sd=B["f_sd"], lr_W=B["lr_W"], lr_b=B["lr_b"],
                 c_mu=B["c_mu"], c_sd=B["c_sd"], nets=nets, stage2=s2,
                 context={"Amsterdam": context_pixels("Amsterdam", "amsterdam_data.parquet"),
                          "Madrid": context_pixels("Madrid", "madrid_train.parquet")})
    joblib.dump(_Pack(state), OUT, compress=("zlib", 3))
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT}  {OUT.stat().st_size/1e6:.1f} MB  sha256 {h}")


if __name__ == "__main__":
    main()
