"""h4d2 command line.

  python -m h4d2.cli stage1   [--pretrained DIR]         train (or assemble) the Madrid Stage 1 bundle
  python -m h4d2.cli madrid-cv [--folds 5]                Stage 1 cross-validation on Madrid (retrains per fold)
  python -m h4d2.cli sweep    [--seed 42 --trials 10]    Amsterdam F1 for n = 5, 25, 50, 100, 200
  python -m h4d2.cli adapt    --n 25 [--seed 42]         fit and save the Stage 2 model for one n
  python -m h4d2.cli predict  --city Amsterdam --n 25 --input test.parquet --out preds.csv

Env: H4D_DATA_DIR (folder with the two parquet files), H4D2_OUT (outputs, default ./outputs_h4d2).
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, confusion_matrix
from . import core, stage1, stage2

DATA = Path(os.environ.get("H4D_DATA_DIR", "../Building_Age_Classification/data"))
OUT = Path(os.environ.get("H4D2_OUT", "./outputs_h4d2"))
NETS = ((42, 0.0), (1, 0.0), (2, 0.0), (1, 0.3), (2, 0.3), (3, 0.3))   # (seed, domain-adversarial weight)
SHOTS = (5, 25, 50, 100, 200)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _load(city):
    f = "madrid_train.parquet" if city == "Madrid" else "amsterdam_data.parquet"
    return core.load_pixels(str(DATA / f), city)


def cmd_stage1(a):
    torch.manual_seed(a.seed)
    mad, ams = _load("Madrid"), _load("Amsterdam")
    pre = None
    if a.pretrained:
        pre = [torch.load(Path(a.pretrained) / f"cnn_{tag}.pt") for tag in a.tags.split(",")]
    B = stage1.fit_stage1(mad, ams, nets=NETS, epochs=a.epochs, threads=a.threads, pretrained=pre, log=log)
    OUT.mkdir(parents=True, exist_ok=True)
    stage1.save(B, OUT / "stage1_madrid.pt")
    log("saved", OUT / "stage1_madrid.pt")


def cmd_sweep(a):
    B = stage1.load(OUT / "stage1_madrid.pt")
    ams = _load("Amsterdam")
    prep = stage2.prepare(B, ams)
    zs = f1_score(ams["y"], stage2.proba(prep, stage2.madrid_heads(prep)).argmax(1) + 1, average="macro")
    log(f"zero-shot (no Amsterdam labels): {zs:.4f}")
    rows = []
    for s in a.seeds:
        log(f"seed {s}")
        rows += stage2.sweep(prep, ams["y"], SHOTS, s, a.trials, log=log, rng_mode=a.rng_mode)
    df = pd.DataFrame(rows)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results" / f"amsterdam_trials_{a.rng_mode}.csv", index=False)
    if a.rng_mode == "per_n":
        df.to_csv(OUT / "results" / "amsterdam_trials.csv", index=False)
    summ = df.groupby("n").f1.agg(["mean", "std", "count"]).reset_index()
    summ["std"] = df.groupby("n").f1.std(ddof=0).values
    summ["log2_n"] = np.log2(summ.n)
    summ.to_csv(OUT / "results" / f"amsterdam_summary_{a.rng_mode}.csv", index=False)
    print(summ.to_string(index=False))
    print(f"avg5 = {summ['mean'].mean():.4f}   peak = {summ['mean'].max():.4f}   n25 = {summ.loc[summ.n == 25, 'mean'].item():.4f}")
    json.dump(dict(zero_shot=zs), open(OUT / "results" / "zero_shot.json", "w", encoding="utf-8"))


def cmd_adapt(a):
    B = stage1.load(OUT / "stage1_madrid.pt")
    S = stage2.adapt(B, _load(a.city), a.n, a.seed)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    p = OUT / "models" / f"stage2_{a.city.lower()}_n{a.n}_seed{a.seed}.pt"
    torch.save(S, p)
    log("saved", p)


def cmd_predict(a):
    B = stage1.load(OUT / "stage1_madrid.pt")
    test = core.load_pixels(a.input, a.city)
    context = None if a.no_context else _load(a.city)
    S = None
    if a.city != "Madrid" or a.n:
        S = torch.load(OUT / "models" / f"stage2_{a.city.lower()}_n{a.n}_seed{a.seed}.pt", weights_only=False)
    pred = stage2.predict(B, test, S, context)
    out = pd.DataFrame(dict(px_key=test["px"], py_key=test["py"], pred_class=pred))
    if test["y"] is not None:
        out["true_class"] = test["y"]
        log(f"macro-F1 on input labels: {f1_score(test['y'], pred, average='macro'):.4f}")
    out.to_csv(a.out, index=False)
    log("wrote", a.out, len(out), "pixels")


def cmd_madrid_cv(a):
    """K-fold CV of the full Stage 1 pipeline on Madrid (feature LR + CNNs retrained per fold).
    Held-out pixels are predicted with the Madrid heads; training-fold pixels serve only as
    unlabelled spatial context (their labels are not used for the held-out predictions)."""
    from sklearn.model_selection import StratifiedKFold
    mad, ams = _load("Madrid"), _load("Amsterdam")
    y = mad["y"]
    rows, cm = [], np.zeros((4, 4))
    for rep in range(a.repeats):
        skf = StratifiedKFold(a.folds, shuffle=True, random_state=a.seed + rep)
        for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            sub = lambda idx: {key: (v[idx] if v is not None else None) for key, v in mad.items()}
            B = stage1.fit_stage1(sub(tr), ams, nets=tuple(NETS[i] for i in a.nets), epochs=a.epochs,
                                  threads=a.threads, log=lambda *x: None)
            pred = stage2.predict(B, sub(te), None, sub(tr))
            f = f1_score(y[te], pred, average="macro")
            cm += confusion_matrix(y[te], pred, labels=[1, 2, 3, 4])
            rows.append(dict(repeat=rep, fold=k, f1=f))
            log(f"repeat {rep} fold {k}: macro-F1 {f:.4f}")
    df = pd.DataFrame(rows)
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "results" / "madrid_cv_folds.csv", index=False)
    prop = cm / cm.sum(1, keepdims=True)
    pd.DataFrame(prop, index=[1, 2, 3, 4], columns=[1, 2, 3, 4]).to_csv(OUT / "results" / "madrid_cv_confusion_proportions.csv")
    log(f"Madrid CV macro-F1 {df.f1.mean():.4f} +/- {df.f1.std(ddof=0):.4f} over {len(df)} fits")


def main():
    p = argparse.ArgumentParser(prog="h4d2")
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("stage1"); s.add_argument("--pretrained"); s.add_argument("--tags", default="s42,plain1,plain2,dann1,dann2,dann3")
    s.add_argument("--epochs", type=int, default=40); s.add_argument("--threads", type=int, default=8); s.add_argument("--seed", type=int, default=42)
    s = sp.add_parser("sweep"); s.add_argument("--seeds", type=int, nargs="+", default=[42]); s.add_argument("--trials", type=int, default=10)
    s.add_argument("--rng-mode", choices=["per_n", "notebook"], default="notebook")
    s = sp.add_parser("adapt"); s.add_argument("--n", type=int, required=True); s.add_argument("--seed", type=int, default=42); s.add_argument("--city", default="Amsterdam")
    s = sp.add_parser("predict"); s.add_argument("--city", required=True); s.add_argument("--n", type=int, default=0); s.add_argument("--seed", type=int, default=42)
    s.add_argument("--input", required=True); s.add_argument("--out", required=True); s.add_argument("--no-context", action="store_true")
    s = sp.add_parser("madrid-cv"); s.add_argument("--folds", type=int, default=5); s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--seed", type=int, default=42); s.add_argument("--epochs", type=int, default=40); s.add_argument("--threads", type=int, default=8)
    s.add_argument("--nets", type=int, nargs="+", default=[0, 3])
    a = p.parse_args()
    dict(stage1=cmd_stage1, sweep=cmd_sweep, adapt=cmd_adapt, predict=cmd_predict)[a.cmd](a) if a.cmd != "madrid-cv" else cmd_madrid_cv(a)


if __name__ == "__main__":
    main()
