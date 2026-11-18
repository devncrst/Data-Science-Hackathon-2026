#!/usr/bin/env python
"""Score Stage 2 plug-ins with the official per_n protocol; save five models per method.

python scripts/run_methods.py --methods all --seed 42 --tag final
python scripts/run_methods.py --methods proto,proto_cityz_all --seed 1000 --tag select --no-save
python scripts/run_methods.py --methods proto --seed 42 --tag parity --rng-mode notebook --no-save
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h4d.harness import run_method, summarise  # noqa: E402
from h4d.methods import registry  # noqa: E402
from h4d.preprocess import load_preprocessed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--methods", default="all")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", required=True)
ap.add_argument("--trees", type=int, default=int(os.environ.get("H4D_RF_TREES_E1", 200)))
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--rng-mode", choices=["per_n", "notebook"], default="per_n")
a = ap.parse_args()

out = Path(os.environ.get("H4D_OUT_DIR", "./outputs")).resolve()
d = load_preprocessed(out / "preprocessed" / "preprocessed_data.pkl")
rf = joblib.load(out / "models" / "stage1_madrid.joblib")["model"]
ctx = dict(Xm=d["X_madrid"], ym=d["y_madrid"], Xa=d["X_amsterdam"],
           Xm_raw=d["X_madrid_raw"], Xa_raw=d["X_amsterdam_raw"], rf=rf,
           pid_m=d.get("pixel_ids_madrid"))  # Madrid ids: spatial pseudo-target for round-2 selection
y = d["y_amsterdam"]  # stays in this script; methods only get y[support]
reg = registry(n_estimators=a.trees, n_jobs=int(os.environ.get("H4D_N_JOBS", -1)))
names = list(reg) if a.methods == "all" else a.methods.split(",")

res = out / "results" / "methods"
res.mkdir(parents=True, exist_ok=True)
models = None if a.no_save else out / "models" / "stage2"
stem = f"{a.tag}_{a.rng_mode}"
allt, sel = [], {}
for name in names:
    t0 = time.time()
    trials, cms = run_method(reg[name], ctx, y, seed=a.seed, models_dir=models,
                             rng_mode=a.rng_mode)
    trials["seed"] = a.seed
    trials["rng_mode"] = a.rng_mode
    allt.append(trials)
    if hasattr(reg[name], "selection"):  # round 2: Madrid-only / unlabelled choices, for the log
        sel[name] = reg[name].selection
    np.savez(res / f"{stem}_{name}_cms_trial0.npz", **{f"n{k}": v for k, v in cms.items()})
    print(f"{name:22s} {time.time() - t0:7.1f}s", flush=True)
trials = pd.concat(allt)
trials.to_csv(res / f"{stem}_trials.csv", index=False)
tab = summarise(trials)
tab.to_csv(res / f"{stem}_summary.csv")
(res / f"{stem}_meta.json").write_text(json.dumps(
    {"seed": a.seed, "methods": names, "trees": a.trees, "rng_mode": a.rng_mode,
     "featureset": os.environ.get("H4D_FEATURESET", "base"),
     "feature_names": list(d["feature_names"])}, indent=2), encoding="utf-8")
if sel:
    (res / f"{stem}_selection.json").write_text(json.dumps(sel, indent=2, default=float), encoding="utf-8")
print(tab.to_string())
