import argparse, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import features as F

SRC = "/shared/collab/preprocessed_data.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orig = pickle.load(open(SRC, "rb"))
    d = F.load(SRC)
    names = d["names"]
    Xm, Xa = d["X_madrid"], d["X_amsterdam"]
    groups = dict(orig.get("feat_groups", {}))
    if a.spatial:
        base = list(names)
        Xm, n2, add = F.add_spatial(Xm, d["xy_madrid"], base)
        Xa, _, _ = F.add_spatial(Xa, d["xy_amsterdam"], base)
        names = n2; groups["spatial"] = add
    Xm, Xa = F.fill_nan(Xm), F.fill_nan(Xa)
    sm, sa = StandardScaler().fit(Xm), StandardScaler().fit(Xa)

    out = dict(orig)
    out.update({
        "X_madrid": sm.transform(Xm).astype(np.float32),
        "X_amsterdam": sa.transform(Xa).astype(np.float32),
        "feature_names": names,
        "scaler": sm,
        "scalers": {"madrid": sm, "amsterdam": sa},
        "scaling": "per_city",
        "feat_groups": groups,
        "built_with": f"spatial={a.spatial} scaling=per_city",
    })
    pickle.dump(out, open(a.out, "wb"))
    print(f"Saved {a.out}: Madrid {out['X_madrid'].shape}, Amsterdam {out['X_amsterdam'].shape}")


if __name__ == "__main__":
    main()
