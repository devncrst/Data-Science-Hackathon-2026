"""h4d2 model bench: run the official few-shot protocol for any model on any city.

Datasets: built-in Amsterdam (class 1/2 boundary 1945) and Madrid (1960), plus any city parquet you add
(same columns as the organiser files, with weighted_mean_year, and that city's class 1/2 boundary year).

Models: built-in h4d2, lr_sup and the organiser prototype baseline, plus uploaded .pkl files.
An uploaded model is a pickled object with ONE of:
  predict_few_shot(df, support) -> DataFrame(px_key, py_key, pred_class)   [few-shot: adapts from support]
      df      = the city's raw parquet as a DataFrame
      support = DataFrame(px_key, py_key, age_class): n labelled pixels per class, drawn by the bench
      (optional) prepare(df) is called once per dataset before the draws.
  predict(df, n) -> DataFrame(px_key, py_key, pred_class)                   [fixed: pre-fitted per n, e.g. h4d2_model.pkl]

Our own h4d2_model.pkl (it has predict(df, n=...) and available_n) runs in one of two modes, never on fresh draws:
  "saved"  : the saved Amsterdam models, n = 5, 25, 50, 100, 200, each scored on every pixel except its own
             labelled support pixels;
  "madrid" : the Madrid model, predict(df, n="madrid"), scored on every pixel (no labels of the city are used).
The default is "madrid" for the Madrid city and "saved" otherwise.  parity() checks the pickle against the
built-in h4d2 on identical pixels with the same saved models.  The pickle file is only read, never written.

Protocol (official harness): organiser NB4 draw order, default_rng(seed), n = 5, 25, 50, 100, 200, 10 draws each,
macro F1 on every non-support pixel.  Only load .pkl files you trust: unpickling runs code.
"""
from __future__ import annotations
import json, os, time, uuid, threading, traceback
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from . import core, stage1, stage2

ROOT = Path(__file__).resolve().parent
ART = Path(os.environ.get("H4D2_OUT", ROOT / "artifacts"))
DATA = Path(os.environ.get("H4D_DATA_DIR", Path.home() / "Building_Age_Classification" / "data"))
HOME = Path(os.environ.get("H4D2_BENCH_HOME", Path.home() / "h4d2_bench"))
SHOTS = (5, 25, 50, 100, 200)
BUILTIN_DATASETS = {"Amsterdam": dict(file="amsterdam_data.parquet", event=1945),
                    "Madrid": dict(file="madrid_train.parquet", event=1960)}
BUILTIN_MODELS = {"h4d2": "h4d2 (ours, frozen)", "lr_sup": "lr_sup (previous champion)",
                  "proto": "Organiser baseline (prototypes)"}
for d in ("datasets", "models", "runs"):
    (HOME / d).mkdir(parents=True, exist_ok=True)
_bundle = {}


def _B():
    if "B" not in _bundle:
        _bundle["B"] = stage1.load(ART / "stage1_madrid.pt")
    return _bundle["B"]


# ---------------------------------------------------------------- datasets
def _ds_meta(name):
    if name in BUILTIN_DATASETS:
        m = dict(BUILTIN_DATASETS[name]); return dict(name=name, event=m["event"], path=str(DATA / m["file"]), builtin=True)
    meta = json.loads((HOME / "datasets" / (name + ".json")).read_text(encoding="utf-8"))
    meta["builtin"] = False
    return meta


def list_datasets():
    out = []
    for name, m in BUILTIN_DATASETS.items():
        if (DATA / m["file"]).exists():
            out.append(dict(name=name, event=m["event"], builtin=True))
    for f in sorted((HOME / "datasets").glob("*.json")):
        m = json.loads(f.read_text(encoding="utf-8")); out.append(dict(name=m["name"], event=m["event"], builtin=False,
                                                      pixels=m.get("pixels"), counts=m.get("counts")))
    return out


def load_city(name):
    m = _ds_meta(name)
    core.CLASS_EVENTS[name] = int(m["event"])          # class 1/2 boundary for this city
    df = pd.read_parquet(m["path"])
    pix = core.load_pixels(df, name)
    if pix["y"] is None:
        raise ValueError(name + ": weighted_mean_year is missing or incomplete; the bench needs labels to score")
    return df, pix


def add_dataset(name, event, raw: bytes):
    name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "city"
    if name in BUILTIN_DATASETS:
        raise ValueError(name + " is a built-in dataset name")
    p = HOME / "datasets" / (name + ".parquet"); p.write_bytes(raw)
    meta = dict(name=name, event=int(event), path=str(p))
    (HOME / "datasets" / (name + ".json")).write_text(json.dumps(meta), encoding="utf-8")
    try:
        _, pix = load_city(name)
    except Exception:
        p.unlink(missing_ok=True); (HOME / "datasets" / (name + ".json")).unlink(missing_ok=True); raise
    meta.update(pixels=int(len(pix["y"])), counts=np.bincount(pix["y"], minlength=5)[1:].tolist())
    (HOME / "datasets" / (name + ".json")).write_text(json.dumps(meta), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- models (common internal interface)
class _H4D2:
    kind = "few-shot"
    def prepare(self, df, pix):
        self.prep = stage2.prepare(_B(), pix)
    def zero_shot(self, pix):
        return stage2.proba(self.prep, stage2.madrid_heads(self.prep)).argmax(1) + 1
    def few_shot(self, df, pix, sup):
        y = pix["y"]
        self._last = (stage2.fit_heads(self.prep, sup, y[sup]), sup)
        return stage2.proba(self.prep, self._last[0], sup, y[sup]).argmax(1) + 1
    def explain(self, pix, sup):
        from . import explain
        return explain.h4d2_explain(self.prep, self._last[0], pix, self._last[1])


class _Feat:
    kind = "few-shot"
    def prepare(self, df, pix):
        B = _B(); self.X = (core.org_feats(pix["arr"], pix["obs"]) - B["f_mu"]) / B["f_sd"]
    zero_shot = None


class _LRSup(_Feat):
    def few_shot(self, df, pix, sup):
        return LogisticRegression(max_iter=2000).fit(self.X[sup], pix["y"][sup]).predict(self.X)


class _Proto(_Feat):
    def few_shot(self, df, pix, sup):
        y = pix["y"]; P = np.stack([self.X[sup][y[sup] == c].mean(0) for c in (1, 2, 3, 4)])
        return np.linalg.norm(self.X[:, None] - P[None], axis=2).argmin(1) + 1


def _align(out, pix):
    """Model output -> class array in the bench's pixel order."""
    if isinstance(out, pd.DataFrame):
        col = "pred_class" if "pred_class" in out else out.columns[-1]
        m = dict(zip(zip(out["px_key"].astype(int), out["py_key"].astype(int)), out[col].astype(int)))
        return np.array([m.get((a, b), 0) for a, b in zip(pix["px"].tolist(), pix["py"].tolist())], int)
    a = np.asarray(out).astype(int).ravel()
    if len(a) != len(pix["px"]):
        raise ValueError("model returned %d predictions for %d pixels; return a DataFrame with px_key, py_key, pred_class" % (len(a), len(pix["px"])))
    return a


PKL_MODES = {"saved": "Amsterdam saved models", "madrid": "Madrid model"}


def default_mode(dataset):
    return "madrid" if dataset == "Madrid" else "saved"


def _pkl_info(obj):
    """Our h4d2 pickle: predict(df, n=...) plus the saved Amsterdam stage-2 models (available_n)."""
    an = getattr(obj, "available_n", None)
    if an is None or not callable(getattr(obj, "predict", None)) or not isinstance(getattr(obj, "s", None), dict):
        return None
    return dict(h4d2pkl=True, available_n=[int(x) for x in an])


def _sup_mask(pix, S):
    """True on the city's pixels that are the saved model's own labelled support pixels."""
    key = set(zip(np.asarray(S["sup_px"]).astype(int).tolist(), np.asarray(S["sup_py"]).astype(int).tolist()))
    return np.array([(a, b) in key for a, b in zip(pix["px"].tolist(), pix["py"].tolist())], bool)


class _User:
    def __init__(self, obj, mode=None):
        self.obj = obj
        self.pkl = _pkl_info(obj)
        self.mode = (mode if mode in PKL_MODES else "saved") if self.pkl else None
        if self.pkl:
            self.kind = "fixed"
        elif hasattr(obj, "predict_few_shot"):
            self.kind = "few-shot"
        elif hasattr(obj, "predict"):
            self.kind = "fixed"
        else:
            raise ValueError("the object has neither predict_few_shot(df, support) nor predict(df, n)")
    def prepare(self, df, pix):
        if self.kind == "few-shot" and hasattr(self.obj, "prepare"):
            self.obj.prepare(df)
    zero_shot = None
    def few_shot(self, df, pix, sup):
        support = pd.DataFrame(dict(px_key=pix["px"][sup], py_key=pix["py"][sup], age_class=pix["y"][sup]))
        return _align(self.obj.predict_few_shot(df, support), pix)
    def fixed(self, df, pix, n):
        return _align(self.obj.predict(df, n), pix)


def add_model(name, raw: bytes):
    name = "".join(c for c in name if c.isalnum() or c in "-_.").replace(".pkl", "") or "model"
    p = HOME / "models" / (name + ".pkl"); p.write_bytes(raw)
    try:
        obj = joblib.load(p)
    except Exception as e:
        p.unlink(missing_ok=True)
        raise ValueError("could not open this file as a joblib/pickle model (%s: %s). If it pickles a custom class, that class must be importable here, or the pickle must be self-contained like h4d2_model.pkl." % (type(e).__name__, e))
    try:
        m = _User(obj)
    except Exception:
        p.unlink(missing_ok=True); raise
    info = dict(kind=m.kind, **(m.pkl or {}))
    (HOME / "models" / (name + ".json")).write_text(json.dumps(info), encoding="utf-8")
    return dict(name=name, **info)


def _model_info(f):
    """Cached facts about an uploaded model (kind, and whether it is our h4d2 pickle)."""
    side = f.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        m = _User(joblib.load(f)); info = dict(kind=m.kind, **(m.pkl or {}))
    except Exception as e:
        info = dict(kind=None, error=str(e))
    side.write_text(json.dumps(info), encoding="utf-8")
    return info


def list_models():
    out = [dict(name=k, label=v, kind="few-shot", builtin=True) for k, v in BUILTIN_MODELS.items()]
    for f in sorted((HOME / "models").glob("*.pkl")):
        info = _model_info(f)
        out.append(dict(name=f.stem, label=f.stem + " (uploaded)", builtin=False, **info))
    return out


def _make_model(name, mode=None):
    if name == "h4d2": return _H4D2()
    if name == "lr_sup": return _LRSup()
    if name == "proto": return _Proto()
    return _User(joblib.load(HOME / "models" / (name + ".pkl")), mode)


# ---------------------------------------------------------------- jobs
JOBS, _lock = {}, threading.Lock()


def _cm(y, p):
    c = confusion_matrix(y, p, labels=[1, 2, 3, 4]).astype(float)
    return (c / np.maximum(c.sum(1, keepdims=True), 1)).round(4).tolist()


PREP_TEXT = {"h4d2": "building 90 m neighbourhood features and running 6 neural networks over every pixel's 42-year series",
             "lr_sup": "computing the 60 temporal features for every pixel",
             "proto": "computing the 60 temporal features for every pixel"}


def plan(models, shots, modes=None):
    st = [dict(id="load", group="Data", label="Load city", status="pending")]
    for m in models:
        g = BUILTIN_MODELS.get(m, m)
        st.append(dict(id="prep:" + m, group=g, label="Prepare", status="pending"))
        if (modes or {}).get(m) == "madrid":
            st.append(dict(id="n:%s:madrid" % m, group=g, label="n = madrid", status="pending"))
        else:
            st += [dict(id="n:%s:%d" % (m, n), group=g, label="n = %d" % n, status="pending") for n in shots]
    return st


def _run_pkl(m, name, r, df, pix, shots, trials, dataset, progress, done, total):
    """Our h4d2 pickle on its own saved models: no fresh support draws."""
    y = pix["y"]; obj = m.obj
    r["mode"] = m.mode
    if m.mode == "madrid":
        r["label"] = "%s.pkl · Madrid mode" % name
        progress(0.02 + 0.96 * done / total, "%s: predicting %s with the Madrid model (n = 'madrid'), no labels of this city" % (r["label"], dataset), "n:%s:madrid" % name)
        p = _align(obj.predict(df, n="madrid"), pix)
        r["zero_shot"] = float(f1_score(y, p, average="macro"))
        r["pred"]["0"] = p.astype(int).tolist(); r["cms"]["0"] = _cm(y, p)
        r["scored"] = dict(pixels=int(len(y)), rule="every pixel" + (" (these are the Madrid model's training pixels)" if dataset == "Madrid" else ""))
        r["avg"] = None
        return done + len(shots) * trials
    avail = set(m.pkl["available_n"])
    used = [n for n in shots if n in avail]
    r["label"] = "%s.pkl · Amsterdam saved n = %s" % (name, ", ".join(map(str, used)) if used else "none")
    r["scored"] = dict(rule="every pixel except that saved model's own labelled support pixels", per_n={})
    for n in shots:
        if n not in avail:
            progress(0.02 + 0.96 * done / total, "%s has no saved model for n = %d" % (name, n), "skip:n:%s:%d" % (name, n))
            done += trials; continue
        progress(0.02 + 0.96 * done / total, "%s: predicting %s with the saved Amsterdam n = %d model" % (r["label"], dataset, n), "n:%s:%d" % (name, n))
        p = _align(obj.predict(df, n=n), pix)
        q = ~_sup_mask(pix, obj.s["stage2"][int(n)])
        r.setdefault("sup", {})[str(n)] = np.where(~q)[0].astype(int).tolist()   # the saved model's labelled pixels, ringed on the map
        r["per_n"][str(n)] = [float(f1_score(y[q], p[q], average="macro")), 0.0, 1]
        r["cms"][str(n)] = _cm(y[q], p[q]); r["pred"][str(n)] = p.astype(int).tolist()
        r["scored"]["per_n"][str(n)] = int(q.sum())
        done += trials
    r["avg"] = float(np.mean([v[0] for v in r["per_n"].values()])) if r["per_n"] else None
    return done


def run(dataset, models, shots=SHOTS, trials=10, seed=42, progress=lambda f, msg, stage=None: None, modes=None):
    t0 = time.time()
    progress(0.01, "Reading %s: 42 years of Landsat observations per pixel" % dataset, "load")
    df, pix = load_city(dataset)
    progress(0.02, "Loaded %s pixels from %s" % (format(len(pix["px"]), ","), dataset), "load")
    y = pix["y"]
    counts = np.bincount(y, minlength=5)[1:]
    shots_req = [int(n) for n in shots]
    shots = [n for n in shots_req if n < counts.min()]
    skipped = [n for n in shots_req if n not in shots]
    if skipped:
        progress(0.02, "Skipping n = %s: the smallest class has only %d pixels" % (", ".join(map(str, skipped)), counts.min()),
                 "skip:" + "|".join("n:%s:%d" % (m, n) for m in models for n in skipped))
    draws = core.notebook_draws(y, seed, trials)          # official stream: n = 5, 10, 25, 50, 100, 200
    res = dict(dataset=dataset, event=_ds_meta(dataset)["event"], seed=int(seed), trials=int(trials), shots=shots,
               pixels=int(len(y)), counts=counts.tolist(), models={}, created=time.strftime("%Y-%m-%d %H:%M"))
    px = pix["px"] - pix["px"].min(); py = pix["py"] - pix["py"].min()
    res["map"] = dict(w=int(px.max() + 1), h=int(py.max() + 1), px=px.tolist(), py=py.tolist(), y=y.tolist(),
                      sup={str(n): draws[n][0].tolist() for n in shots})
    total = len(models) * (1 + len(shots) * trials)
    done = 0
    for mi, name in enumerate(models):
        label = BUILTIN_MODELS.get(name, name)
        progress(0.02 + 0.96 * done / total, "%s: %s" % (label, PREP_TEXT.get(name, "loading and preparing the model")), "prep:" + name)
        m = _make_model(name, (modes or {}).get(name) or default_mode(dataset))
        m.prepare(df, pix)
        done += 1
        r = dict(label=label, kind=m.kind, per_n={}, cms={}, pred={})
        if getattr(m, "pkl", None):
            done = _run_pkl(m, name, r, df, pix, shots, trials, dataset, progress, done, total)
            res["models"][name] = r
            continue
        if getattr(m, "zero_shot", None):
            z = m.zero_shot(pix); r["zero_shot"] = float(f1_score(y, z, average="macro")); r["pred"]["0"] = z.astype(int).tolist()
        for n in shots:
            f1s, acc = [], np.zeros((4, 4))
            reps = draws[n] if m.kind == "few-shot" else draws[n][:1]
            for t, sup in enumerate(reps):
                progress(0.02 + 0.96 * done / total, "%s: learning %s from %d labelled pixels per class (draw %d of %d)" % (label, dataset, n, t + 1, len(reps)), "n:%s:%d" % (name, n))
                p = m.few_shot(df, pix, sup) if m.kind == "few-shot" else m.fixed(df, pix, n)
                q = np.ones(len(y), bool); q[sup] = False
                if m.kind == "fixed":
                    q[:] = True                       # fixed models choose their own support; score every pixel
                f1s.append(float(f1_score(y[q], p[q], average="macro")))
                acc += np.array(_cm(y[q], p[q]))
                if t == 0:
                    r["pred"][str(n)] = np.asarray(p).astype(int).tolist()
                    if hasattr(m, "explain"):
                        r.setdefault("explain", {})[str(n)] = m.explain(pix, sup)
                done += 1
            done += (trials - len(reps))
            r["per_n"][str(n)] = [float(np.mean(f1s)), float(np.std(f1s)), len(f1s)]
            r["cms"][str(n)] = (acc / len(reps)).round(4).tolist()
        r["avg"] = float(np.mean([v[0] for v in r["per_n"].values()])) if r["per_n"] else None
        res["models"][name] = r
    res["seconds"] = round(time.time() - t0, 1)
    progress(1.0, "Done in %.0f s" % res["seconds"], "end")
    return res


def submit(dataset, models, shots=SHOTS, trials=10, seed=42, modes=None):
    jid = uuid.uuid4().hex[:10]
    shots_ok = [int(n) for n in shots]
    modes = {m: (modes or {}).get(m) or default_mode(dataset) for m in models
             if m not in BUILTIN_MODELS and _model_info(HOME / "models" / (m + ".pkl")).get("h4d2pkl")}
    job = dict(id=jid, status="queued", progress=0.0, message="Queued", started=time.time(), stages=plan(models, shots_ok, modes), eta=None)
    with _lock:
        JOBS[jid] = job
    def prog(f, msg, stage=None):
        el = time.time() - job["started"]
        job["progress"] = round(float(f), 4); job["message"] = msg; job["elapsed"] = round(el, 1)
        job["eta"] = round(el / f - el, 0) if f > 0.04 and f < 1 else None
        if stage and stage.startswith("skip:"):
            ids = set(stage[5:].split("|"))
            for st in job["stages"]:
                if st["id"] in ids: st["status"] = "skipped"
            return
        if stage:
            hit = False
            for st in job["stages"]:
                if st["id"] == stage:
                    st["status"] = "done" if stage == "end" else "running"; hit = True
                elif not hit and st["status"] != "skipped":
                    st["status"] = "done"
            if stage == "end":
                for st in job["stages"]:
                    if st["status"] != "skipped": st["status"] = "done"
    def work():
        job["status"] = "running"
        try:
            res = run(dataset, models, shots, trials, seed, prog, modes)
            res["id"] = jid
            (HOME / "runs" / (jid + ".json")).write_text(json.dumps(res), encoding="utf-8")
            job["status"] = "done"
        except Exception as e:
            job["status"] = "error"; job["message"] = str(e); job["trace"] = traceback.format_exc()[-2000:]
    threading.Thread(target=work, daemon=True).start()
    return jid


def parity(dataset, model, progress=lambda f, msg, stage=None: None):
    """Built-in h4d2 vs our pickle on IDENTICAL pixels (this city's file, no extra context) with the SAME
    saved models: Amsterdam saved n = 5 and 25, or the Madrid model.  Nothing is refitted or redrawn."""
    import torch
    t0 = time.time()
    progress(0.02, "Loading %s and the uploaded %s.pkl (read only)" % (dataset, model), "load")
    df, pix = load_city(dataset)
    obj = joblib.load(HOME / "models" / (model + ".pkl"))
    info = _pkl_info(obj)
    if not info:
        raise ValueError("%s.pkl is not the h4d2 pickle (it has no predict(df, n=...) with saved models)" % model)
    y = pix["y"]
    mode = default_mode(dataset)
    targets = ["madrid"] if mode == "madrid" else [n for n in (5, 25) if n in info["available_n"]]
    progress(0.08, "Built-in h4d2: 90 m neighbourhood features and 6 networks on %s pixels" % format(len(y), ","), "prep:builtin")
    prep = stage2.prepare(_B(), pix)
    progress(0.35, "Pickle: the same steps with its own embedded code on the identical pixels", "prep:pkl")
    Wm, reps = obj._prepare(pix)
    out = []
    for i, t in enumerate(targets):
        tag = "madrid" if t == "madrid" else "saved n = %d" % t
        progress(0.6 + 0.35 * i / len(targets), "Comparing %s: built-in vs pickle, then the pickle's public predict()" % tag, "cmp:%s" % t)
        if t == "madrid":
            Pb = stage2.proba(prep, stage2.madrid_heads(prep))
            Pp = obj._proba(Wm, reps, [(W0, b0) for (_, W0, b0) in reps])
            hd = max(float(np.abs(np.asarray(a[0]) - np.asarray(b[1])).max()) for a, b in zip(stage2.madrid_heads(prep), reps))
            q = np.ones(len(y), bool); nsup = 0
            pub = _align(obj.predict(df, n="madrid", context=False), pix)
        else:
            Sb = torch.load(ART / "models" / ("stage2_amsterdam_n%d_seed42.pt" % t), weights_only=False)
            Sp = obj.s["stage2"][int(t)]
            key = {(a, b): k for k, (a, b) in enumerate(zip(pix["px"].tolist(), pix["py"].tolist()))}
            def supmap(S):
                sup, ys = [], []
                for a, b, c in zip(np.asarray(S["sup_px"]).tolist(), np.asarray(S["sup_py"]).tolist(), np.asarray(S["sup_y"]).tolist()):
                    if (a, b) in key: sup.append(key[(a, b)]); ys.append(c)
                return np.array(sup, int), np.array(ys, int)
            sb, yb = supmap(Sb); sp_, yp = supmap(Sp)
            Pb = stage2.proba(prep, Sb["heads"], sb, yb)
            Pp = obj._proba(Wm, reps, Sp["heads"], sp_, yp)
            hd = max(max(float(np.abs(np.asarray(Wa) - np.asarray(Wc)).max()), float(np.abs(np.asarray(ba) - np.asarray(bc)).max()))
                     for (Wa, ba), (Wc, bc) in zip(Sb["heads"], Sp["heads"]))
            q = np.ones(len(y), bool); q[sb] = False; nsup = int(len(sb))
            same_sup = bool(np.array_equal(sb, sp_) and np.array_equal(yb, yp))
            pub = _align(obj.predict(df, n=int(t), context=False), pix)
        pb, pp = Pb.argmax(1) + 1, Pp.argmax(1) + 1
        row = dict(target=tag, pixels=int(len(y)), scored=int(q.sum()), support_in_city=nsup,
                   agree=float((pb == pp).mean()), agree_public=float((pb == pub).mean()),
                   f1_builtin=float(f1_score(y[q], pb[q], average="macro")), f1_pkl=float(f1_score(y[q], pp[q], average="macro")),
                   f1_pkl_public=float(f1_score(y[q], pub[q], average="macro")),
                   max_abs_dp=float(np.abs(Pb - Pp).max()), max_abs_dheads=hd)
        if t != "madrid":
            row["same_support"] = same_sup
        out.append(row)
    res = dict(dataset=dataset, model=model, mode=mode, pixels=int(len(y)), rows=out, seconds=round(time.time() - t0, 1),
               note="Identical pixels: this city's file only, no stored context, on both sides. Built-in = stage2 with the "
                    "saved artifacts; pickle = its embedded code and models. F1 on every pixel except the saved model's support pixels.")
    progress(1.0, "Parity check done in %.0f s" % res["seconds"], "end")
    return res


def submit_parity(dataset, model):
    jid = uuid.uuid4().hex[:10]
    mode = default_mode(dataset)
    tg = ["madrid"] if mode == "madrid" else [5, 25]
    st = [dict(id="load", group="Data", label="Load city + pickle", status="pending"),
          dict(id="prep:builtin", group="Built-in h4d2", label="Prepare", status="pending"),
          dict(id="prep:pkl", group=model + ".pkl", label="Prepare", status="pending")]
    st += [dict(id="cmp:%s" % t, group="Compare", label=("madrid" if t == "madrid" else "n = %d" % t), status="pending") for t in tg]
    job = dict(id=jid, kind="parity", status="queued", progress=0.0, message="Queued", started=time.time(), stages=st, eta=None)
    with _lock:
        JOBS[jid] = job
    def prog(f, msg, stage=None):
        el = time.time() - job["started"]
        job["progress"] = round(float(f), 4); job["message"] = msg; job["elapsed"] = round(el, 1)
        job["eta"] = round(el / f - el, 0) if f > 0.04 and f < 1 else None
        hit = False
        for s_ in job["stages"]:
            if s_["id"] == stage: s_["status"] = "running"; hit = True
            elif not hit: s_["status"] = "done"
        if stage == "end":
            for s_ in job["stages"]: s_["status"] = "done"
    def work():
        job["status"] = "running"
        try:
            job["result"] = parity(dataset, model, prog); job["status"] = "done"
        except Exception as e:
            job["status"] = "error"; job["message"] = str(e); job["trace"] = traceback.format_exc()[-2000:]
    threading.Thread(target=work, daemon=True).start()
    return jid


def leaderboard():
    rows = []
    for f in sorted((HOME / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for name, m in r["models"].items():
            rows.append(dict(run=r["id"], dataset=r["dataset"], model=m["label"], kind=m["kind"], avg=m.get("avg"),
                             n25=(m["per_n"].get("25") or [None])[0], peak=max([v[0] for v in m["per_n"].values()] or [None]),
                             zero_shot=m.get("zero_shot"), seed=r["seed"], trials=r["trials"], created=r["created"]))
    return rows


def get_run(rid):
    return json.loads((HOME / "runs" / (rid + ".json")).read_text(encoding="utf-8"))
