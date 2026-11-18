"""
quick_test.py - does a feature/scaling option help Madrid AND Amsterdam?
Default scaling = per city. Few-shot: Amsterdam features + Madrid-model probabilities
-> logistic regression, 5 draws per n (seed 42+r). Appends a row to /shared/collab/results.csv
"""
import argparse, csv, os, time
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import features as F

DATA = "/shared/collab/preprocessed_data.pkl"
RESULTS = "/shared/collab/results.csv"
SEED = 42
NS = [5, 25, 50, 100, 200]
REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--change", action="store_true")
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--madridscaler", action="store_true")
    ap.add_argument("--data", default=DATA)
    a = ap.parse_args()
    t0 = time.time()

    d = F.load(a.data)
    names = d["names"]
    Xm, Xa = d["X_madrid"], d["X_amsterdam"]
    ym, ya = d["y_madrid"], d["y_amsterdam"]
    added = []
    if a.change:
        Xm, n2, add = F.add_change(Xm, names)
        Xa, _, _ = F.add_change(Xa, names)
        names, added = n2, added + add
    if a.spatial:
        base = list(names)
        Xm, n2, add = F.add_spatial(Xm, d["xy_madrid"], base)
        Xa, _, _ = F.add_spatial(Xa, d["xy_amsterdam"], base)
        names, added = n2, added + add
    Xm, Xa = F.fill_nan(Xm), F.fill_nan(Xa)

    if a.madridscaler:
        mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
        Xm, Xa = (Xm - mu) / sd, (Xa - mu) / sd
    else:
        Xm, Xa = F.standardise_city(Xm), F.standardise_city(Xa)

    classes = np.unique(ym)
    print(f"[{a.name}] features={Xm.shape[1]}  Madrid={len(ym)}  Amsterdam={len(ya)}  classes={classes.tolist()}", flush=True)
    if added:
        with open(f"feature_table_{a.name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["feature", "reason"]); w.writerows(F.feature_table(added))

    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
                  random_state=SEED, verbose=-1, n_jobs=8)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    f1m = []
    for tr, te in cv.split(Xm, ym):
        m = lgb.LGBMClassifier(**params).fit(Xm[tr], ym[tr])
        f1m.append(f1_score(ym[te], m.predict(Xm[te]), average="macro"))
    print(f"Madrid CV macro-F1     : {np.mean(f1m):.3f} ± {np.std(f1m):.3f}", flush=True)

    mad = lgb.LGBMClassifier(**params).fit(Xm, ym)
    zs = f1_score(ya, mad.predict(Xa), average="macro")
    print(f"Amsterdam zero-shot    : {zs:.3f}", flush=True)
    Za = np.hstack([Xa, mad.predict_proba(Xa)])

    res = {}
    for n in NS:
        s = []
        for r in range(REPEATS):
            rng = np.random.default_rng(SEED + r)
            tr = np.concatenate([rng.choice(np.where(ya == k)[0], n, replace=False) for k in classes])
            te = np.setdiff1d(np.arange(len(ya)), tr)
            clf = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
            clf.fit(Za[tr], ya[tr])
            s.append(f1_score(ya[te], clf.predict(Za[te]), average="macro"))
        res[n] = (float(np.mean(s)), float(np.std(s)))
        print(f"Amsterdam n={n:3d}        : {res[n][0]:.3f} ± {res[n][1]:.3f}", flush=True)
    ams_mean = float(np.mean([res[n][0] for n in NS]))
    print(f"Amsterdam mean (5 n's) : {ams_mean:.3f}    [baseline 0.596]   ({time.time()-t0:.0f}s)", flush=True)

    row = {"name": a.name,
           "what_changed": f"change={a.change} spatial={a.spatial} madridscaler={a.madridscaler}",
           "n_features": Xm.shape[1], "madrid_f1": round(np.mean(f1m), 4),
           "madrid_std": round(np.std(f1m), 4), "ams_zeroshot": round(zs, 4)}
    for n in NS:
        row[f"ams_n{n}"] = round(res[n][0], 4)
        row[f"ams_n{n}_std"] = round(res[n][1], 4)
    row["ams_mean"] = round(ams_mean, 4)
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    print(f"Logged to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
