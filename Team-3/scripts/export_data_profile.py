#!/usr/bin/env python
"""Madrid-vs-Amsterdam data panel: results/data_profile.json from the raw parquets (one row per pixel)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h4d import config as c  # noqa: E402
from h4d.labels import assign_age_class  # noqa: E402

BINS = list(range(1800, 2031, 10))
out = {"bins": BINS, "cities": {}}
for city, path in (("Madrid", c.MADRID_PATH), ("Amsterdam", c.AMSTERDAM_PATH)):
    df = pd.read_parquet(path, columns=["px_key", "py_key", "year", "weighted_mean_year"])
    px = df.drop_duplicates(["px_key", "py_key"]).assign(city=city)
    px = assign_age_class(px, events=c.CLASS_EVENTS)
    w = px["weighted_mean_year"].dropna().clip(BINS[0], BINS[-1] - 1)
    out["cities"][city] = {
        "rows": int(len(df)), "pixels": int(len(px)),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "class_counts": {str(k): int(v) for k, v in px["age_class"].value_counts().sort_index().items()},
        "wmy_hist": np.histogram(w, bins=BINS)[0].tolist(),
        "wmy_quantiles": {q: float(w.quantile(float(q))) for q in ("0.1", "0.5", "0.9")},
    }
dst = Path(os.environ.get("H4D_OUT_DIR", "./outputs")) / "results" / "data_profile.json"
dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("wrote", dst)
