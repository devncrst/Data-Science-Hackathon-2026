"""
predict.py  :  Apply our saved models to NEW pixels (e.g. the organisers' held-out test set).

Usage from a notebook:
    import predict
    out = predict.predict_city("data/amsterdam_test.parquet", city="Amsterdam",
                               stage1_path="models/stage1_madrid_model.joblib",
                               amsterdam_model_path="models/amsterdam_blend3_n025_seed42.joblib")
    out.to_csv("predictions_amsterdam_n025.csv", index=False)

Command line:
    python predict.py data/amsterdam_test.parquet Amsterdam models/amsterdam_blend3_n025_seed42.joblib

Steps (exactly the same as training, so test pixels are treated like training pixels):
 1. data_prep.load_city + clean_long_table   (same cloud masking + temporal haze rule)
 2. features_v3.build_features                (same 132 features; works without labels)
    + neighbourhood features (tv3.add_neighbourhood): averages over the 3x3 and 7x7 windows, with our own
      pixels of that city as extra neighbours (the setting is read from the saved Stage 1 model)
 3. Madrid  -> Stage 1 Random Forest
    Amsterdam -> the saved blend3 model for the chosen n (fine-tuned forest + stack + prototypes),
                 using the scaler saved in stage 1 (fitted on the Amsterdam TRANSFER data, never on test)
 4. 5x5 smoothing of the class probabilities (ON by default; smooth=False for the plain model)
 5. If the file has weighted_mean_year, labels are added and macro F1 is printed (for checking only).
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score
import data_prep as dp
import features_v3 as fv3
import transfer_v3 as tv3

EVENTS = {"Madrid": 1960, "Amsterdam": 1945}


def predict_city(parquet_path, city, stage1_path="models/stage1_madrid_model.joblib",
                 amsterdam_model_path=None, smooth=True, alpha=tv3.SMOOTH_ALPHA,
                 radius=tv3.SMOOTH_RADIUS, context_features=None):
    """smooth=True (default, the final method) mixes each pixel's probabilities with its neighbours' in a
    (2*radius+1) x (2*radius+1) window (see tv3.smooth_proba; final setting 5x5, strength 0.5).
    context_features = a features parquet of OUR OWN pixels of the same city (default: features_<city>_v3.parquet
    if it exists). Used as extra neighbours for the neighbourhood features and for smoothing (features and
    predictions only, never labels). Only the test pixels are returned."""
    raw = dp.load_city(parquet_path)
    has_labels = "weighted_mean_year" in raw.columns and raw["weighted_mean_year"].notna().all()
    all_pix = raw[["px_key", "py_key"]].drop_duplicates()

    clean = dp.clean_long_table(raw)
    # SAFETY NET: a pixel whose photos ALL fail our cleaning rules would get no prediction.
    # For those pixels only, fall back to the organisers' simpler rule (any qa_valid photo).
    kept = clean[["px_key", "py_key"]].drop_duplicates()
    lost = all_pix.merge(kept, how="left", indicator=True).query("_merge == 'left_only'")[["px_key", "py_key"]]
    if len(lost):
        sub = raw.merge(lost, on=["px_key", "py_key"])
        fb = sub[[c for c in ["px_key", "py_key", "year", "weighted_mean_year"] if c in sub]].copy()
        for b in dp.BANDS:
            col = np.full(len(sub), np.nan)
            for slot in (1, 2, 3):
                ok = sub[f"qa_valid_{slot}"].fillna(False).to_numpy(bool) & (sub[f"{b}_{slot}"].to_numpy(float) > 0)
                col = np.where(np.isnan(col) & ok, sub[f"{b}_{slot}"].to_numpy(float), col)
            fb[b] = col
        fb = fb.dropna(subset=dp.BANDS)
        clean = pd.concat([clean, fb], ignore_index=True)
        print(f"safety net: {len(lost)} pixels had no photo passing our rules; "
              f"{fb[['px_key','py_key']].drop_duplicates().shape[0]} recovered with the qa_valid-only rule")
    if has_labels:
        clean = dp.add_age_class(clean, EVENTS[city])
    feats, cols = fv3.build_features(clean)

    s1 = joblib.load(stage1_path)
    own_cols = s1.get("own_feature_cols", s1["feature_cols"])       # v1 models: no neighbourhood features
    radii = tuple(s1.get("neighbour_radii", ()))
    assert cols == own_cols, "feature columns differ from the ones the model was trained on"
    X_own = feats[cols].to_numpy(float)
    ids = feats[["px_key", "py_key"]].to_numpy(np.int64)
    stage1 = {"rf": s1["rf"], "scaler_a": s1["scaler_amsterdam"]}

    # CONTEXT = our own pixels of this city (features only, never labels). They serve as extra neighbours
    # for the neighbourhood features and for smoothing, so scattered test pixels still have neighbours.
    if context_features is None:
        default = f"features_{city.lower()}_v3.parquet"
        context_features = default if os.path.exists(default) else None
    ctx_all = pd.read_parquet(context_features) if context_features else None
    if ctx_all is not None:
        is_test = ctx_all.merge(feats[["px_key", "py_key"]].drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").to_numpy()
        ctx_ids_all = ctx_all[["px_key", "py_key"]].to_numpy(np.int64)
        ctx_X_all = ctx_all[cols].to_numpy(float)
        print(f"context: {(~is_test).sum():,} of our own {city} pixels used as neighbours")
        if not (~is_test).any():          # every context pixel is a test pixel: nothing extra to add
            ctx_all = None

    # model inputs for the test pixels: own features (+ neighbourhood features over test + context pixels)
    if radii:
        if ctx_all is not None:
            X = tv3.add_neighbourhood(np.vstack([X_own, ctx_X_all[~is_test]]), np.vstack([ids, ctx_ids_all[~is_test]]),
                                      radii, n_own=len(X_own))
        else:
            X = tv3.add_neighbourhood(X_own, ids, radii)
    else:
        X = X_own
    assert X.shape[1] == len(s1["feature_cols"]), "model input width differs from training"

    if city == "Madrid":
        proba = lambda Z: s1["rf"].predict_proba(Z)
    else:
        assert amsterdam_model_path, "give the saved Amsterdam model for the chosen n"
        am = joblib.load(amsterdam_model_path)
        proba = lambda Z: tv3.predict_proba_blend3(am, stage1, Z)
    P = proba(X)
    if smooth:
        if ctx_all is not None:
            # context pixels keep the inputs they had in training (neighbours = our own pixels only)
            ctx_in = tv3.add_neighbourhood(ctx_X_all, ctx_ids_all, radii)[~is_test] if radii else ctx_X_all[~is_test]
            P_all = np.vstack([P, proba(ctx_in)])
            ids_all = np.vstack([ids, ctx_ids_all[~is_test]])
            P = tv3.smooth_proba(P_all, ids_all, alpha, radius)[: len(P)]
        else:
            P = tv3.smooth_proba(P, ids, alpha, radius)
    pred = tv3.CLASSES[P.argmax(1)]

    out = feats[["px_key", "py_key"]].copy()
    out["predicted_age_class"] = pred.astype(int)
    # LAST RESORT: a pixel with no usable photo at all still gets an answer (Stage 1's most common
    # class in training), flagged so it can be reported honestly.
    missing = all_pix.merge(out[["px_key", "py_key"]], how="left", indicator=True).query("_merge == 'left_only'")
    out["fallback"] = False
    if len(missing):
        default = int(s1["rf"].classes_[np.argmax(np.bincount(np.argmax(s1["rf"].predict_proba(X), 1)))])
        extra = missing[["px_key", "py_key"]].assign(predicted_age_class=default, fallback=True)
        out = pd.concat([out, extra], ignore_index=True)
        print(f"last resort: {len(missing)} pixels with no usable photo got class {default} (flagged)")
    assert len(out) == len(all_pix), "every input pixel must get exactly one prediction"
    if has_labels:
        truth = raw[["px_key", "py_key", "weighted_mean_year"]].drop_duplicates(["px_key", "py_key"])
        truth = dp.add_age_class(truth.assign(city=city), EVENTS[city])[["px_key", "py_key", "age_class"]]
        out = out.merge(truth.rename(columns={"age_class": "true_age_class"}), on=["px_key", "py_key"])
        print(f"{city}: macro F1 on ALL {len(out):,} pixels = "
              f"{f1_score(out.true_age_class, out.predicted_age_class, average='macro'):.4f}")
    print(f"predicted {len(out):,} pixels; class counts:",
          out.predicted_age_class.value_counts().sort_index().to_dict())
    return out


if __name__ == "__main__":
    path, city = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else None
    res = predict_city(path, city, amsterdam_model_path=model)
    res.to_csv(f"predictions_{city.lower()}.csv", index=False)
    print("saved", f"predictions_{city.lower()}.csv")
