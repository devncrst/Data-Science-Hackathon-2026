#!/usr/bin/env python
"""python scripts/gate_check.py --final r3_nb --confirm r3_nb_s1000 [--robust r3_pn] --champion lr_sup [--extra-dir DIR:fs ...]
Prints the gate table (markdown) and writes results/methods/<final>_gate.csv."""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h4d.gate import gate_table  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True)
ap.add_argument("--confirm")
ap.add_argument("--champion", default="lr_sup")
ap.add_argument("--robust", help="tag scored under the OTHER draw protocol (per_n), same seed")
ap.add_argument("--candidates")
ap.add_argument("--extra-dir", action="append", default=[], help="OUT_DIR:featureset (e.g. ~/h4d_out/tcp:tcp)")
a = ap.parse_args()
res = Path(os.environ.get("H4D_OUT_DIR", "./outputs")).resolve() / "results" / "methods"


def load(tag, base=res, fs=None):
    p = base / f"{tag}_trials.csv"
    if not p.exists():
        return None
    t = pd.read_csv(p)
    if fs:
        t["method"] = t["method"] + "@" + fs
    return t


tr, cf = [load(a.final)], [load(a.confirm)] if a.confirm else [None]
rb = [load(a.robust)] if a.robust else [None]
for spec in a.extra_dir:
    d, fs = spec.rsplit(":", 1)
    base = Path(os.path.expanduser(d)) / "results" / "methods"
    tr.append(load(a.final, base, fs))
    cf.append(load(a.confirm, base, fs) if a.confirm else None)
    rb.append(load(a.robust, base, fs) if a.robust else None)
trials = pd.concat([t for t in tr if t is not None])
confirm = pd.concat([t for t in cf if t is not None]) if any(t is not None for t in cf) else None
robust = pd.concat([t for t in rb if t is not None]) if any(t is not None for t in rb) else None
cands = a.candidates.split(",") if a.candidates else None
tab = gate_table(trials, confirm, a.champion, cands, robust)
tab.to_csv(res / f"{a.final}_gate.csv", index=False)
print(tab.to_string(index=False))
