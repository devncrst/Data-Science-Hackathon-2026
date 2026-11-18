"""Explain the neighbour-blending step of h4d2 on real pixels.

For one support draw it records the model's own guess (no blending) and the blended guess,
picks a real 7 x 7 window where blending changed the centre pixel, and measures the overall effect
(pixels fixed and broken, macro F1 before and after). Used by the dashboard and the model bench.
Uses the model's own guesses and the support labels only; true ages are added for display.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from . import core, stage1, stage2

R = 3   # window half-width in pixels: matches the neighbour radius used by the model


def _r(v):
    return [round(float(x), 3) for x in v]


def effect(y, raw, smo, sup):
    q = np.ones(len(y), bool); q[np.asarray(sup, int)] = False
    yq, a, b = y[q], raw[q], smo[q]
    return dict(changed=int((a != b).sum()), scored=int(q.sum()),
                fixed=int(((a != yq) & (b == yq)).sum()), broken=int(((a == yq) & (b != yq)).sum()),
                f1_raw=round(float(f1_score(yq, a, average="macro")), 4),
                f1_blend=round(float(f1_score(yq, b, average="macro")), 4))


def patch(pix, W, raw, smo, Praw, Psm, sup):
    """A real window around a pixel whose guess blending changed (preferably fixed), with every cell's numbers."""
    y, px, py = pix["y"], pix["px"], pix["py"]
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(px, py))}
    supset = set(int(s) for s in sup)
    deg = np.diff(W.indptr)
    ok = np.array([i not in supset for i in range(len(y))])
    best = None
    for kind, pool in (("fixed", np.where(ok & (raw != y) & (smo == y))[0]),
                       ("changed", np.where(ok & (raw != smo))[0]),
                       ("none", np.where(ok)[0])):
        for i in pool:
            js = W.indices[W.indptr[i]:W.indptr[i + 1]]; ws = W.data[W.indptr[i]:W.indptr[i + 1]]
            if len(js) < 4:
                continue
            share = ws[raw[js] == smo[i]].sum() / ws.sum()
            win = [key.get((int(px[i]) + dx, int(py[i]) + dy)) for dx in range(-R, R + 1) for dy in range(-R, R + 1)]
            win = [j for j in win if j is not None]
            # a clear example: neighbours that explain the change, several changes and mixed true ages in view
            sc = share * min(len(js), 20) + 3 * min(int((raw[win] != smo[win]).sum()), 6) + 4 * len(set(y[win].tolist()))
            if best is None or sc > best[0]:
                best = (sc, int(i))
        if best:
            break
    if best is None:
        return None
    i = best[1]
    cx, cy = int(px[i]), int(py[i])
    js = W.indices[W.indptr[i]:W.indptr[i + 1]]; ws = W.data[W.indptr[i]:W.indptr[i + 1]]
    vote = np.array([ws[raw[js] == c].sum() for c in (1, 2, 3, 4)]); vote = vote / max(vote.sum(), 1e-9)
    cells = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            j = key.get((cx + dx, cy + dy))
            cells.append(None if j is None else dict(t=int(y[j]), r=int(raw[j]), s=int(smo[j]), sup=j in supset,
                                                     pr=_r(Praw[j]), ps=_r(Psm[j])))
    return dict(kind=kind, cx=cx, cy=cy, nb=int(deg[i]), vote=_r(vote), cells=cells)


def h4d2_explain(prep, heads, pix, sup):
    """Everything the dashboard panel needs for one h4d2 run on one support draw."""
    y = pix["y"]; sup = np.asarray(sup, int)
    Praw = stage2.proba(prep, heads, spatial=False)
    Psm = stage2.proba(prep, heads, sup, y[sup], spatial=True)
    raw, smo = Praw.argmax(1) + 1, Psm.argmax(1) + 1
    return dict(effect=effect(y, raw, smo, sup), patch=patch(pix, prep["W"], raw, smo, Praw, Psm, sup))


def draws_table(prep, y, draws, shots):
    """Macro F1 before and after blending, averaged over every draw of every budget."""
    out = {}
    for n in shots:
        a, b = [], []
        for sup in draws[n]:
            sup = np.asarray(sup, int)
            h = stage2.fit_heads(prep, sup, y[sup])
            q = np.ones(len(y), bool); q[sup] = False
            a.append(f1_score(y[q], stage2.proba(prep, h, spatial=False).argmax(1)[q] + 1, average="macro"))
            b.append(f1_score(y[q], stage2.proba(prep, h, sup, y[sup], spatial=True).argmax(1)[q] + 1, average="macro"))
        out[str(n)] = dict(raw=round(float(np.mean(a)), 4), blend=round(float(np.mean(b)), 4),
                           helped=int(sum(x < z for x, z in zip(a, b))), draws=len(a))
    return out


def add_to_payload(shots=(5, 25, 50, 100, 200)):
    """Add the official-results explainer (Amsterdam, official draws, seed 42) to dashboard_payload.json."""
    out = Path(os.environ.get("H4D2_OUT", "./h4d2/artifacts"))
    data = Path(os.environ.get("H4D_DATA_DIR", "../Building_Age_Classification/data"))
    B = stage1.load(out / "stage1_madrid.pt")
    ams = core.load_pixels(str(data / "amsterdam_data.parquet"), "Amsterdam")
    y = ams["y"]
    prep = stage2.prepare(B, ams)
    draws = core.notebook_draws(y, 42, 10)
    ex = {}
    for n in shots:
        sup = np.asarray(draws[n][0], int)
        ex[str(n)] = h4d2_explain(prep, stage2.fit_heads(prep, sup, y[sup]), ams, sup)
        print("explain", n, ex[str(n)]["effect"], flush=True)
    table = draws_table(prep, y, draws, shots)
    print("table", table, flush=True)
    p = out / "dashboard_payload.json"
    payload = json.load(open(p, encoding="utf-8"))
    payload["explain"] = dict(city="Amsterdam", draw="first official draw (seed 42)", per_n=ex, table=table)
    json.dump(payload, open(p, "w", encoding="utf-8"))
    json.dump(payload["explain"], open(out / "results" / "blending_effect.json", "w", encoding="utf-8"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    add_to_payload()
