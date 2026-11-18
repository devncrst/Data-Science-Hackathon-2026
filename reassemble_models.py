"""
reassemble_models.py : rebuild the models/ folder from the pieces in models_split/.

Why: GitHub refuses files over 100 MB, so each trained model was saved compressed
(joblib compress=3, the same model: checked to give identical predictions) and cut into 95 MB pieces.

Usage (from the folder that contains this script):
    python reassemble_models.py
Result: models/*.joblib, ready for PREDICT_with_saved_models.ipynb and predict.py.
Each rebuilt file is checked against the SHA-256 in models_split/MANIFEST.txt.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(HERE, "models_split")
OUT = os.path.join(HERE, "models")
os.makedirs(OUT, exist_ok=True)

for line in open(os.path.join(SPLIT, "MANIFEST.txt")):
    if not line.strip():
        continue
    name, sha, n_parts = line.split()
    target = os.path.join(OUT, name)
    digest = hashlib.sha256()
    with open(target, "wb") as out:
        for i in range(int(n_parts)):                      # join the pieces in order
            chunk = open(os.path.join(SPLIT, f"{name}.part{i:02d}"), "rb").read()
            out.write(chunk)
            digest.update(chunk)
    ok = digest.hexdigest() == sha                         # same bytes as the file we split
    print(f"{name}: {int(n_parts)} pieces joined, checksum {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"{name} is damaged: download models_split/ again")
print("models/ is ready")
