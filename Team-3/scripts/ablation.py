#!/usr/bin/env python
"""Feature-group ablation: prototypes on column subsets, same per_n harness (seconds, no RF refit)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h4d.changepoint import CP_COLS  # noqa: E402
from h4d.features import FEAT_GROUPS  # noqa: E402
from h4d.harness import run_method, summarise  # noqa: E402
from h4d.methods import Proto  # noqa: E402
from h4d.preprocess import load_preprocessed  # noqa: E402
from h4d.sensor import SENSOR_COLS  # noqa: E402
from h4d.temporal import TCP_COLS  # noqa: E402


class SubsetProto(Proto):
    def __init__(self, name, cols_idx):
        self.name, self.ci = name, np.asarray(cols_idx)

    def embed(self, st, Xs, Xraw, rf):
        return Xs[:, self.ci]


d = load_preprocessed(Path(os.environ.get("H4D_OUT_DIR", "./outputs")) / "preprocessed" / "preprocessed_data.pkl")
names = list(d["feature_names"])
ctx = {"Xa": d["X_amsterdam"], "Xa_raw": d["X_amsterdam_raw"], "rf": None}
y = d["y_amsterdam"]
groups = dict(FEAT_GROUPS)
for g, cols in (("changepoint", CP_COLS), ("sensor", SENSOR_COLS), ("tcp", TCP_COLS)):
    if set(cols) <= set(names):
        groups[g] = cols
variants = {"all": names, **{f"minus_{g}": [c for c in names if c not in cols] for g, cols in groups.items()}}
BASE = sum(FEAT_GROUPS.values(), [])  # the 60 NB3 columns
for g in ("changepoint", "sensor", "tcp"):
    if g in groups:
        variants[f"base_plus_{g}"] = BASE + groups[g]
trials = pd.concat([run_method(SubsetProto(k, [names.index(c) for c in v]), ctx, y)[0]
                    for k, v in variants.items()])
print(summarise(trials).sort_values("avg5", ascending=False).to_string())
