#!/usr/bin/env python
"""Write schema-faithful synthetic parquet files for local testing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.conftest import make_wide  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pixels", type=int, default=None, help="pixels for both cities")
    p.add_argument("--pixels-madrid", type=int, default=None)
    p.add_argument("--pixels-amsterdam", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_m = args.pixels_madrid or args.pixels or 400
    n_a = args.pixels_amsterdam or args.pixels or 400
    args.out.mkdir(parents=True, exist_ok=True)

    mad = make_wide("Madrid", n_pixels=n_m, seed=args.seed)
    ams = make_wide("Amsterdam", n_pixels=n_a, seed=args.seed + 1, city_shift=300.0)
    mad_path = args.out / "madrid_train.parquet"
    ams_path = args.out / "amsterdam_data.parquet"
    mad.to_parquet(mad_path, index=False)
    ams.to_parquet(ams_path, index=False)
    print(f"Wrote {mad_path} shape={mad.shape}")
    print(f"Wrote {ams_path} shape={ams.shape}")
    print(f"Madrid cols={len(mad.columns)} Amsterdam cols={len(ams.columns)}")


if __name__ == "__main__":
    main()
