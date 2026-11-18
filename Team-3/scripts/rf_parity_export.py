# run in the repo venv (sklearn 1.9.1): dump the exact arrays the RF sees, plus quick sanity facts
import os, pickle, numpy as np, sklearn
OUT = os.environ["H4D_OUT_DIR"]
d = pickle.load(open(f"{OUT}/preprocessed/preprocessed_data.pkl", "rb"))
np.savez(f"{OUT}/rf_arrays.npz", Xm=d["X_madrid"], ym=d["y_madrid"], Xa=d["X_amsterdam"], ya=d["y_amsterdam"])
names = list(d["feature_names"]); pm = np.asarray(d["pixel_ids_madrid"])
print("sklearn", sklearn.__version__, "| X", d["X_madrid"].shape, d["X_madrid"].dtype)
print("y_madrid counts", np.bincount(d["y_madrid"])[1:].tolist())           # organiser 13794/27152/18672/16645
print("Madrid rows sorted by (px,py):", bool((np.lexsort((pm[:, 1], pm[:, 0])) == np.arange(len(pm))).all()))
for c in ("has_early_data", "has_late_data"):
    raw = d["X_madrid_raw"][:, names.index(c)] if "X_madrid_raw" in d else None
    print(c, "raw unique:", None if raw is None else np.unique(raw).tolist(),
          "| scaled unique (first 3):", np.unique(d["X_madrid"][:, names.index(c)])[:3].round(4).tolist())
