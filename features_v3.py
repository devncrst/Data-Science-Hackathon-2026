"""
features_v3.py  :  Feature table version 3 (Fenny + Jian, with Claude)

Builds ONE row per pixel from the clean pixel-year tables made in step 1.
Use the SAME function for Madrid and Amsterdam so a feature means the same thing in both cities.

    import features_v3 as fv3
    feat_mad, cols = fv3.build_features(pd.read_parquet("clean/clean_madrid.parquet"))
    feat_ams, _    = fv3.build_features(pd.read_parquet("clean/clean_amsterdam.parquet"))

WHAT IS NEW COMPARED TO v2
1. Every band is converted to real reflectance first (DN * 0.0000275 - 0.2), so indices are correct.
2. Two kinds of statistics:
   LEVEL statistics  use the gap-filled series (interpolated), because averages need every year.
   CHANGE statistics use OBSERVED years only. Interpolated years are straight lines, which make a
   pixel look calmer than it is. Amsterdam has more missing years than Madrid, so using interpolated
   years would make Amsterdam look "older" for reasons that have nothing to do with age.
3. New change features: trend slope, biggest jump, and the YEAR of the biggest jump
   (a building finished in 1995 should show its biggest change near 1995).
4. An honest, label-free check of which features shift between cities (shift_report). It uses
   only feature values, never Amsterdam labels.

NO-LEAKAGE GUARANTEE: the label (age_class) and IDs (px_key, py_key) are returned as separate
columns and are NEVER in the feature list. weighted_mean_year and coverage are not used at all.
"""

import numpy as np
import pandas as pd

BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
ID_COLS = ["px_key", "py_key"]
YEARS = np.arange(1984, 2026)                     # 1984 to 2025
EARLY_END, LATE_START = 2003, 2004                # same split as the official notebooks
SR_SCALE, SR_OFFSET = 0.0000275, -0.2             # Landsat Collection 2 surface reflectance scaling

# Normalised-difference indices: (first band - second band) / (first band + second band)
# Each one has a physical reason, which goes on the slides.
INDICES = {
    "NDVI":   ("NIR", "Red",   "vegetation; bare or built land has less"),
    "NDBI":   ("SWIR1", "NIR", "built-up materials reflect SWIR1 more than NIR"),
    "MNDWI":  ("Green", "SWIR1", "moisture and water; canals, wet surfaces"),
    "NDSWIR": ("SWIR1", "SWIR2", "shape of the SWIR spectrum; differs between roof/wall materials"),
}
BSI_REASON = "bare soil vs vegetation; high on construction sites"
CHANNELS = BANDS + list(INDICES) + ["BSI"]        # 6 bands + 5 indices = 11 channels

# Columns we must never use as features (checked at the end of build_features)
FORBIDDEN = {"weighted_mean_year", "age_class", "coverage", "px_key", "py_key",
             "pixel_x", "pixel_y", "pixel_id", "city", "year", "n_clean_obs", "n_obs_years"}


def _nd(a, b):
    """Normalised difference, with division-by-zero turned into NaN (not infinity)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def _cube(df):
    """Pivot the long table to cube[pixel, year, channel] (NaN where a year was not observed).
    Returns the cube, the pixel index (px_key, py_key) and the label per pixel."""
    df = df[(df["year"] >= YEARS[0]) & (df["year"] <= YEARS[-1])].copy()
    if "age_class" not in df.columns:      # unlabelled data (e.g. the organisers' test set): label 0 = unknown
        df["age_class"] = 0
    # one class per pixel, otherwise the data is broken
    assert df.groupby(ID_COLS)["age_class"].nunique().max() == 1, "a pixel has two classes"

    # reflectance for each band
    refl = {b: df[b].to_numpy(dtype=np.float64) * SR_SCALE + SR_OFFSET for b in BANDS}
    vals = dict(refl)
    for name, (a, b, _) in INDICES.items():
        vals[name] = _nd(refl[a], refl[b])
    vals["BSI"] = _nd(refl["SWIR1"] + refl["Red"], refl["NIR"] + refl["Blue"])

    pix = pd.MultiIndex.from_frame(df[ID_COLS]).unique()          # stable pixel order
    labels = df.groupby(ID_COLS)["age_class"].first().reindex(pix).to_numpy()
    layers = []
    for ch in CHANNELS:
        wide = (pd.DataFrame({"px_key": df["px_key"].values, "py_key": df["py_key"].values,
                              "year": df["year"].values, "v": vals[ch]})
                .pivot_table(index=ID_COLS, columns="year", values="v", aggfunc="mean")
                .reindex(index=pix, columns=YEARS))
        layers.append(wide.to_numpy())
    return np.stack(layers, axis=2), pix, labels


def build_features(clean_df):
    """Clean pixel-year table -> (feature table with one row per pixel, list of feature columns)."""
    cube, pix, labels = _cube(clean_df)                 # (pixels, 42 years, 11 channels)
    observed = ~np.isnan(cube)
    n_pix, n_years, n_ch = cube.shape

    # ---- gap-filled copy for LEVEL statistics (linear in time, edges copied outwards) ----
    filled = np.empty_like(cube)
    for c in range(n_ch):
        filled[:, :, c] = (pd.DataFrame(cube[:, :, c], columns=YEARS)
                           .interpolate(axis=1, limit_direction="both").to_numpy())

    early = YEARS <= EARLY_END
    late = YEARS >= LATE_START
    feats, names = [], []

    def add(arr2d, suffix):                              # arr2d: (pixels, channels)
        feats.append(arr2d)
        names.extend([f"{ch}_{suffix}" for ch in CHANNELS])

    # LEVEL statistics (what the surface looks like, before and after 2004)
    add(filled.mean(axis=1), "mean")
    add(filled.std(axis=1), "std")
    add(filled[:, early].mean(axis=1), "early_mean")
    add(filled[:, late].mean(axis=1), "late_mean")
    add(filled[:, early].std(axis=1), "early_std")
    add(filled[:, late].std(axis=1), "late_std")
    add(filled[:, late].mean(axis=1) - filled[:, early].mean(axis=1), "late_minus_early")

    # Trend slope per year, least squares on OBSERVED years only
    t = np.broadcast_to(YEARS[None, :, None], cube.shape).astype(np.float64)
    w = observed.astype(np.float64)
    nobs = w.sum(axis=1)
    t_mean = (t * w).sum(axis=1) / np.maximum(nobs, 1)
    x0 = np.where(observed, cube, 0.0)
    x_mean = x0.sum(axis=1) / np.maximum(nobs, 1)
    cov = ((t - t_mean[:, None, :]) * (x0 - x_mean[:, None, :]) * w).sum(axis=1)
    var = (((t - t_mean[:, None, :]) ** 2) * w).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(var > 0, cov / var, np.nan)
    add(slope, "slope")

    # CHANGE statistics on OBSERVED years only.
    # For each observed year, compare with the previous OBSERVED year (skipping cloudy gaps).
    #  * step  = raw difference. For a stable building the difference between two photos is just
    #            noise, whatever the gap, so its spread does NOT depend on how cloudy the city is.
    #            A construction step also shows up whole, even if the years around it are missing.
    #  * rate  = difference divided by the gap in years, used only for the average yearly change.
    steps = np.full(cube.shape, np.nan)
    rates = np.full(cube.shape, np.nan)
    for c in range(n_ch):
        v = pd.DataFrame(cube[:, :, c], columns=YEARS)
        yr = pd.DataFrame(np.where(observed[:, :, c], YEARS[None, :], np.nan), columns=YEARS)
        pv = v.ffill(axis=1).shift(1, axis=1)            # previous observed value
        py = yr.ffill(axis=1).shift(1, axis=1)           # year of that previous observation
        steps[:, :, c] = (v - pv).to_numpy()
        rates[:, :, c] = ((v - pv) / (yr - py)).to_numpy()
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)    # pixels with 1 observation -> NaN, filled below
        add(np.nanmean(rates, axis=1), "chg_mean")       # average change per year
        add(np.nanstd(steps, axis=1), "chg_std")         # how jumpy (gap-proof)
        abs_s = np.abs(steps)
        add(np.nanmax(abs_s, axis=1), "max_jump")        # biggest single change
        idx = np.nanargmax(np.where(np.isnan(abs_s), -np.inf, abs_s), axis=1)
        add(YEARS[idx].astype(np.float64), "year_of_max_jump")   # WHEN the biggest change happened

    X = np.concatenate(feats, axis=1)
    table = pd.DataFrame(X, columns=names)
    # Pixels with too few observations can give NaN in a change feature: fill with the median of
    # that feature IN THE SAME CITY (no labels used). Report how many so nothing is hidden.
    n_nan = int(np.isnan(X).sum())
    if n_nan:
        table = table.fillna(table.median())
    front = pd.DataFrame({"px_key": pix.get_level_values(0), "py_key": pix.get_level_values(1),
                          "age_class": labels.astype(int),
                          "n_obs_years": observed[:, :, 0].sum(axis=1)})  # info only, NOT a feature
    table = pd.concat([front, table], axis=1)

    assert not (set(names) & FORBIDDEN), "a forbidden column ended up in the features"
    print(f"built {len(table):,} pixels x {len(names)} features "
          f"({n_nan} NaN values filled with the city median)")
    return table, names


def shift_report(feat_a, feat_b, cols):
    """LABEL-FREE check of how much each feature's distribution moves between two cities.

    Uses the Kolmogorov-Smirnov statistic (0 = identical distributions, 1 = no overlap) and the
    shift of the median measured in Madrid standard deviations. No labels are used at all.

    Caution when reading it: part of any shift is real, because Amsterdam has fewer class 4
    buildings than Madrid. So a high shift is a warning to test, not an automatic drop.
    """
    from scipy.stats import ks_2samp
    rows = []
    for c in cols:
        a, b = feat_a[c].to_numpy(), feat_b[c].to_numpy()
        sd = a.std() if a.std() > 0 else 1.0
        rows.append({"feature": c,
                     "ks": ks_2samp(a, b).statistic,
                     "median_shift_sd": abs(np.median(b) - np.median(a)) / sd})
    return pd.DataFrame(rows).sort_values("ks").reset_index(drop=True)
