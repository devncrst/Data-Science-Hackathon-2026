#!/usr/bin/env python
"""Model Lead's paste-ready results table (CSV + markdown) from the harness trials.

python scripts/results_table.py --trials outputs/results/methods/final_trials.csv \
    --methods proto,rf_retrain_a0.5 --out outputs/results/RESULTS
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL = (5, 25, 50, 100, 200)
ap = argparse.ArgumentParser()
ap.add_argument("--trials", required=True)
ap.add_argument("--methods", default=None, help="comma list; default = all in file")
ap.add_argument("--out", required=True, help="path stem; writes .csv and .md")
a = ap.parse_args()

out = Path(os.environ.get("H4D_OUT_DIR", "./outputs")).resolve()
t = pd.read_csv(a.trials)
if "method" not in t.columns:
    t = t.copy()
    t["method"] = "proto"
if "seed" not in t.columns:
    t["seed"] = int(os.environ.get("H4D_SEED", 42))
if a.methods:
    t = t[t["method"].isin(a.methods.split(","))]
rows = []
for m, g in t.groupby("method", sort=False):
    seed = int(g["seed"].iloc[0])
    per_n = {}
    for n in OFFICIAL:
        f = g.loc[g["n"] == n, "f1"].to_numpy()
        if len(f) == 0:
            continue
        per_n[n] = (f.mean(), np.std(f), len(f))
        rows.append(
            {
                "method": m,
                "n": n,
                "mean_macro_f1": f.mean(),
                "std": np.std(f),
                "trials": len(f),
                "seed": seed,
                "rng_mode": g["rng_mode"].iloc[0] if "rng_mode" in g else "per_n",
                "model_file": f"models/stage2_amsterdam_n{n}.npz"
                if m == "proto"
                else f"models/stage2/{m}/amsterdam_n{n}.joblib",
            }
        )
    if set(OFFICIAL) - set(per_n):
        raise SystemExit(f"{m} missing official n: {set(OFFICIAL) - set(per_n)}")
    means = np.array([per_n[n][0] for n in OFFICIAL])
    stds = np.array([per_n[n][1] for n in OFFICIAL])
    pk = OFFICIAL[int(means.argmax())]
    rows += [
        {
            "method": m,
            "n": "avg5",
            "mean_macro_f1": means.mean(),
            "std": np.sqrt((stds ** 2).sum()) / 5,
            "trials": 50,
            "seed": seed,
            "model_file": "",
        },
        {
            "method": m,
            "n": f"peak(n={pk})",
            "mean_macro_f1": means.max(),
            "std": per_n[pk][1],
            "trials": per_n[pk][2],
            "seed": seed,
            "model_file": "",
        },
        {
            "method": m,
            "n": "25-shot",
            "mean_macro_f1": per_n[25][0],
            "std": per_n[25][1],
            "trials": per_n[25][2],
            "seed": seed,
            "model_file": "models/stage2_amsterdam_n25.npz"
            if m == "proto"
            else f"models/stage2/{m}/amsterdam_n25.joblib",
        },
    ]
cv = json.loads((out / "results" / "madrid_cv_summary.json").read_text(encoding="utf-8"))
zs = json.loads((out / "results" / "zero_shot.json").read_text(encoding="utf-8"))
rows += [
    {
        "method": "stage1_rf (Madrid CV 5x5)",
        "n": "madrid_cv",
        "mean_macro_f1": cv["f1_mean"],
        "std": cv["f1_std"],
        "trials": 25,
        "seed": 42,
        "model_file": "models/stage1_madrid.joblib",
    },
    {
        "method": "stage1_rf (zero-shot)",
        "n": 0,
        "mean_macro_f1": zs["macro_f1"],
        "std": np.nan,
        "trials": 1,
        "seed": 42,
        "model_file": "models/stage1_madrid.joblib",
    },
]
df = pd.DataFrame(rows)
df.to_csv(a.out + ".csv", index=False, float_format="%.4f")
md = [
    "| method | n | mean macro-F1 | std | trials | seed | model file |",
    "|---|---|---|---|---|---|---|",
]
for r in df.itertuples():
    sd = "" if pd.isna(r.std) else f"{r.std:.4f}"
    md.append(
        f"| {r.method} | {r.n} | {r.mean_macro_f1:.4f} | {sd} | {r.trials} | {r.seed} | {r.model_file} |"
    )
modes = sorted(set(t["rng_mode"])) if "rng_mode" in t.columns else ["per_n"]
proto = {"notebook": "organiser NB4 draw order (one default_rng(seed) over n = 5, 10, 25, 50, 100, 200)",
         "per_n": "per_n protocol, support drawn with default_rng([seed, n])"}
md.append("\n*Amsterdam: " + "; ".join(proto[m] for m in modes) + ", 10 trials, "
          "query = all non-support pixels; std = ddof 0 over trials; avg5 std = sqrt(sum std^2)/5.*")
Path(a.out + ".md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
