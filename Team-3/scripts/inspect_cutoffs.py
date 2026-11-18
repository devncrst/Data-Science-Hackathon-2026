#!/usr/bin/env python
"""On-site: find where the organisers put the class cutoffs. Read-only."""
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, ".")
from h4d.io import REQUIRED  # noqa: E402

for path in sys.argv[1:]:
    sch = pq.read_schema(path)
    print(f"\n=== {path}\ncolumns ({len(sch.names)}): {sch.names}")
    extra = [c for c in sch.names if c not in REQUIRED]
    print("not in pipeline schema:", extra)
    for k, v in (sch.metadata or {}).items():
        if k != b"pandas":
            print("metadata:", k.decode(), "=", v.decode()[:300])
    df = pd.read_parquet(path, columns=[c for c in extra + ["city"] if c in sch.names])
    for c in extra:
        nun = df.groupby("city")[c].nunique() if "city" in df.columns else df[c].nunique()
        print(
            f"  {c:28s} dtype={df[c].dtype}  nunique/city={dict(nun) if hasattr(nun, 'items') else nun}"
            f"  sample={df[c].dropna().unique()[:5].tolist()}"
        )
