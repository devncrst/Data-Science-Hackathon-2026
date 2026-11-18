"""Run h4d2 on an uploaded parquet and render the results dashboard (used by h4d2/Dashboard.ipynb)."""
from __future__ import annotations
import html, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score
from . import core, stage1, stage2

ROOT = Path(__file__).resolve().parent
ART = Path(os.environ.get("H4D2_OUT", ROOT / "artifacts"))
DATA = Path(os.environ.get("H4D_DATA_DIR", "/home/participant-c1/Building_Age_Classification/data"))
RUNS = Path(os.environ.get("H4D2_RUNS", Path.home() / "h4d2_out" / "dashboard_runs"))
SHOTS = (5, 25, 50, 100, 200)
_cache = {}


def _bundle():
    if "B" not in _cache:
        _cache["B"] = stage1.load(ART / "stage1_madrid.pt")
    return _cache["B"]


def _context(city):
    key = "ctx_" + city
    if key not in _cache:
        f = "madrid_train.parquet" if city == "Madrid" else "amsterdam_data.parquet"
        _cache[key] = core.load_pixels(str(DATA / f), city)
    return _cache[key]


def run(test_path, city="Amsterdam", ns=SHOTS, use_context=True, log=print):
    """Predict every requested n (Amsterdam) or the Madrid model, preparing the pixels once.
    Returns a list of dataset dicts for the dashboard and writes one CSV per run."""
    t0 = time.time()
    B = _bundle()
    test = core.load_pixels(str(test_path), city)
    log(f"loaded {len(test['px']):,} pixels from {Path(test_path).name}")
    if use_context:
        pix, n_test = core.union_pixels(test, _context(city))
    else:
        pix, n_test = test, len(test["px"])
    prep = stage2.prepare(B, pix)
    log(f"features + networks ready ({time.time()-t0:.0f}s)")
    key = {(a, b): i for i, (a, b) in enumerate(zip(pix["px"].tolist(), pix["py"].tolist()))}
    out, RUNS_ = [], RUNS
    RUNS_.mkdir(parents=True, exist_ok=True)
    stem = Path(test_path).stem
    jobs = [None] if city == "Madrid" else list(ns)
    for n in jobs:
        if n is None:
            P = stage2.proba(prep, stage2.madrid_heads(prep)); name = f"{stem}_madrid"
        else:
            S = torch.load(ART / "models" / f"stage2_amsterdam_n{n}_seed42.pt", weights_only=False)
            sup, ysup = [], []
            for a, b, c in zip(S["sup_px"].tolist(), S["sup_py"].tolist(), S["sup_y"].tolist()):
                if (a, b) in key:
                    sup.append(key[(a, b)]); ysup.append(c)
            P = stage2.proba(prep, S["heads"], np.array(sup, int), np.array(ysup, int))
            name = f"{stem}_amsterdam_n{n}"
        pred = P.argmax(1)[:n_test] + 1
        df = pd.DataFrame(dict(px_key=test["px"], py_key=test["py"], pred_class=pred))
        ds = dict(name=name + ".csv", city=city.lower(), n=n, px=test["px"].tolist(), py=test["py"].tolist(),
                  pred=pred.astype(int).tolist(), y=None)
        if test["y"] is not None:
            df["true_class"] = test["y"]; ds["y"] = test["y"].astype(int).tolist()
            log(f"  {name}: macro-F1 {f1_score(test['y'], pred, average='macro'):.4f}")
        else:
            log(f"  {name}: done")
        df.to_csv(RUNS_ / (name + ".csv"), index=False)
        out.append(ds)
    log(f"finished in {time.time()-t0:.0f}s; CSVs in {RUNS_}")
    return out


def page(datasets=None, note=""):
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    payload = (ART / "dashboard_payload.json").read_text(encoding="utf-8")
    live = ""
    if datasets:
        live = "<script>window.H4D2_LIVE=" + json.dumps(dict(note=note, datasets=datasets)) + ";</script>"
    return tpl.replace('<script id="payload"', live + '<script id="payload"', 1).replace("__PAYLOAD__", payload)


def show(datasets=None, note="", height=1500):
    from IPython.display import HTML, display
    doc = page(datasets, note)
    RUNS.mkdir(parents=True, exist_ok=True)
    f = RUNS / ("dashboard_" + time.strftime("%Y%m%d_%H%M%S") + ".html")
    f.write_text(doc, encoding="utf-8")
    display(HTML(f'<iframe srcdoc="{html.escape(doc, quote=True)}" style="width:100%;height:{height}px;border:0;border-radius:12px"></iframe>'
                 f'<div style="font-size:12px;color:#5D6685">Saved a standalone copy: {f}</div>'))
