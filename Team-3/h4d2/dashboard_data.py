"""Build the dashboard payload (JSON): headline numbers, curves, confusion matrices and Amsterdam maps."""
from __future__ import annotations
import json, os, base64
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from . import core, stage1, stage2

OUT = Path(os.environ.get("H4D2_OUT", "./h4d2/artifacts"))
DATA = Path(os.environ.get("H4D_DATA_DIR", "../Building_Age_Classification/data"))
SHOTS = (5, 25, 50, 100, 200)


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def cm_norm(y, p):
    c = confusion_matrix(y, p, labels=[1, 2, 3, 4]).astype(float)
    return c / c.sum(1, keepdims=True)


def main():
    B = stage1.load(OUT / "stage1_madrid.pt")
    ams = core.load_pixels(str(DATA / "amsterdam_data.parquet"), "Amsterdam")
    y = ams["y"]
    prep = stage2.prepare(B, ams)
    Xa = (core.org_feats(ams["arr"], ams["obs"]) - B["f_mu"]) / B["f_sd"]   # organiser features, Madrid scaling
    draws = core.notebook_draws(y, 42, 10)
    cms = {"h4d2": {}, "lr_sup": {}, "proto": {}}
    maps, sup_idx = {}, {}
    for n in SHOTS:
        acc = {k: np.zeros((4, 4)) for k in cms}
        for t, sup in enumerate(draws[n]):
            q = np.ones(len(y), bool); q[sup] = False
            ph = stage2.proba(prep, stage2.fit_heads(prep, sup, y[sup]), sup, y[sup]).argmax(1) + 1
            pl = LogisticRegression(max_iter=2000).fit(Xa[sup], y[sup]).predict(Xa)
            protos = np.stack([Xa[sup][y[sup] == c].mean(0) for c in (1, 2, 3, 4)])
            pp = np.linalg.norm(Xa[:, None] - protos[None], axis=2).argmin(1) + 1
            acc["h4d2"] += cm_norm(y[q], ph[q]); acc["lr_sup"] += cm_norm(y[q], pl[q]); acc["proto"] += cm_norm(y[q], pp[q])
            if t == 0:
                maps[n], sup_idx[n] = ph, sup
        for k in cms:
            cms[k][str(n)] = (acc[k] / len(draws[n])).round(4).tolist()
        print("cm done", n, flush=True)
    zero = stage2.proba(prep, stage2.madrid_heads(prep)).argmax(1) + 1
    px = ams["px"] - ams["px"].min(); py = ams["py"] - ams["py"].min()
    res = OUT / "results"
    payload = dict(
        official=json.load(open(res / "official_and_seed1000.json", encoding="utf-8")),
        madrid_cv=dict(folds=[0.7172, 0.7199, 0.7228, 0.7026, 0.7144], mean=0.7154, std=0.0070, rf=0.628),
        zero_shot=dict(h4d2=0.5935, rf=0.4433),
        cms=cms,
        madrid_cm=np.loadtxt(res / "madrid_cv_confusion_proportions.csv", delimiter=",", skiprows=1)[:, 1:].round(4).tolist(),
        map=dict(n=int(len(y)), w=int(px.max() + 1), h=int(py.max() + 1),
                 px=b64(px, np.uint16), py=b64(py, np.uint16), y=b64(y, np.uint8), zero=b64(zero, np.uint8),
                 pred={str(n): b64(maps[n], np.uint8) for n in SHOTS},
                 sup={str(n): b64(sup_idx[n], np.uint32) for n in SHOTS}),
    )
    out = OUT / "dashboard_payload.json"
    json.dump(payload, open(out, "w", encoding="utf-8"))
    print("wrote", out, out.stat().st_size // 1024, "KB")
    from . import explain
    explain.add_to_payload(SHOTS)


if __name__ == "__main__":
    main()
