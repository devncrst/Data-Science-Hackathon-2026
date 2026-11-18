# run in a throwaway sklearn-1.8 venv: if this prints 0.3427 (and fits 0.6171/0.6249/0.6182), the gap is sklearn 1.9
import os, sys, time, numpy as np, sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
a = np.load(f"{os.environ['H4D_OUT_DIR']}/rf_arrays.npz")
Xm, ym, Xa, ya = a["Xm"], a["ym"], a["Xa"], a["ya"]
P = dict(n_estimators=int(os.environ.get("TREES", 500)), class_weight="balanced", random_state=42, n_jobs=-1)
t = time.time()
rf = RandomForestClassifier(**P).fit(Xm, ym)
print(f"sklearn {sklearn.__version__}: zero-shot macro-F1 {f1_score(ya, rf.predict(Xa), average='macro'):.4f}"
      f" (organiser 0.3427) | in-sample acc {np.mean(rf.predict(Xm) == ym):.4f} | {time.time()-t:.0f}s")
k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for i, (tr, va) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(Xm, ym)):
    if i == k: break
    f = f1_score(ym[va], RandomForestClassifier(**P).fit(Xm[tr], ym[tr]).predict(Xm[va]), average="macro")
    print(f"  CV fit {i+1}: {f:.4f}  (organiser fits 1-3: 0.6171 0.6249 0.6182)")
