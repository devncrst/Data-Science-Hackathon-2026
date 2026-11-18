"""
data_prep.py  :  Data loading + cloud masking for the Building Age hackathon.

Use it from any notebook:

    from data_prep import load_city, mask_report, clean_long_table

    raw   = load_city("../data/madrid_train.parquet")         # 1. load (memory friendly)
    print(mask_report(raw))                                   # 2. see what each mask rule removes
    clean = clean_long_table(raw, combine="median")           # 3. one clean value per pixel-year

`clean` has one row per (pixel, year) with the 6 bands, and is what the
feature engineering step (Notebook 3 style) should start from.

WHY OUR OWN CLOUD MASK?
The files already have qa_valid_N, but Notebook 2 found haze that it missed
(Landsat 7 scene, 29 May 2003, Blue up to 64,126). qa_valid ignores the cirrus
bit. So we decode the raw QA_PIXEL bits ourselves and add sanity checks.
"""

import numpy as np
import pandas as pd

BANDS = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
SLOTS = (1, 2, 3)

# ---------------------------------------------------------------------------
# Landsat Collection 2 QA_PIXEL bits (USGS). Same layout for L5/L7 and L8/L9,
# except bit 2 (cirrus) only exists on L8/L9 (always 0 on L5/L7).
# ---------------------------------------------------------------------------
BIT_FILL = 0            # 1 = no data
BIT_DILATED_CLOUD = 1   # 1 = next to a cloud (cloud edges are often hazy)
BIT_CIRRUS = 2          # 1 = high confidence cirrus (thin high cloud), L8/L9 only
BIT_CLOUD = 3           # 1 = high confidence cloud
BIT_SHADOW = 4          # 1 = high confidence cloud shadow
BIT_SNOW = 5            # 1 = snow / ice
# bits 8-9 = cloud confidence (0 none, 1 low, 2 medium, 3 high)

# Collection 2 surface reflectance scaling: reflectance = DN * 0.0000275 - 0.2
SR_SCALE, SR_OFFSET = 0.0000275, -0.2
SR_VALID_MIN, SR_VALID_MAX = 7273, 43636     # valid DN range per USGS

BLUE_MAX = 15_000       # old per-pixel haze threshold from Notebook 2 (only used with haze_mode="pixel")
BLUE_HARD_MAX = 30_000  # any single value above this is cloud/haze whatever the mode (no roof is this bright)
SCENE_MAD_K = 4.0       # a scene is hazy if its median Blue is > 4 robust std devs above the city's typical scene
TEMPORAL_FACTOR = 1.35  # temporal haze rule: drop a photo whose Blue is > 1.35x that pixel's own typical Blue


def bit_set(qa, bit):
    """True where the given bit is 1 in the QA integer. Example: bit_set(5440, 6) -> True."""
    qa = np.asarray(qa, dtype=np.int64)          # numpy array (pandas Series has no >> operator)
    return ((qa >> bit) & 1).astype(bool)


# ---------------------------------------------------------------------------
# 1. LOADING
# ---------------------------------------------------------------------------
def load_city(path, columns=None, float32=True):
    """Read one city's parquet file.

    * columns=None reads everything. Pass a list to read fewer columns (faster, less RAM).
    * float32=True halves the memory of the band columns (plenty of precision for reflectance).
    """
    df = pd.read_parquet(path, columns=columns)
    if float32:
        band_cols = [f"{b}_{s}" for b in BANDS for s in SLOTS if f"{b}_{s}" in df.columns]
        df[band_cols] = df[band_cols].astype(np.float32)
    return df


def quick_look(df):
    """Print the basic facts the team needs on Day 1."""
    n_pix = df.groupby(["px_key", "py_key"]).ngroups
    print(f"City: {df['city'].unique().tolist()}")
    print(f"Rows (pixel-years): {len(df):,}   Unique pixels: {n_pix:,}")
    print(f"Years: {df['year'].min()} to {df['year'].max()} ({df['year'].nunique()} distinct)")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    print(f"Coverage: min {df['coverage'].min():.2f}, median {df['coverage'].median():.2f}")
    print(f"weighted_mean_year: {df['weighted_mean_year'].min():.0f} to {df['weighted_mean_year'].max():.0f}")


# ---------------------------------------------------------------------------
# 2. CLOUD MASKING (one boolean per observation slot)
# ---------------------------------------------------------------------------

def hazy_scenes(df, k=SCENE_MAD_K):
    """Find whole satellite scenes (photos of the whole city on one day) that are hazy.

    WHY: haze covers the whole city at once, so it lifts the MEDIAN Blue of every pixel in that scene.
    A bright new roof or a construction site only lifts ONE pixel. So we judge haze per scene, not per
    pixel, and keep bright buildings (which are real age signal, especially for class 4).

    How: for every scene_id, take the median Blue of all its pixels. Compare each scene to the typical
    scene using the median and MAD (median absolute deviation, a spread measure that outliers can't
    distort). Scenes more than k robust std devs above typical are flagged.
    Returns a Python set of hazy scene_ids.
    """
    parts = []
    for s in SLOTS:
        if f"scene_id_{s}" in df and f"Blue_{s}" in df:
            parts.append(df[[f"scene_id_{s}", f"Blue_{s}"]].set_axis(["scene", "blue"], axis=1))
    if not parts:
        return set()
    long = pd.concat(parts).dropna()
    long = long[(long["blue"] >= SR_VALID_MIN) & (long["blue"] <= SR_VALID_MAX)]   # ignore fill zeros
    med = long.groupby("scene")["blue"].median()
    centre = med.median()
    mad = (med - centre).abs().median() * 1.4826          # 1.4826 turns MAD into a std-dev equivalent
    return set(med[med > centre + k * max(mad, 1.0)].index)

def slot_mask(df, slot, use_qa_valid=True, strict_dilated=True,
              max_cloud_conf=1, blue_max=BLUE_MAX, haze_mode="temporal", bad_scenes=None):
    """Return a dict of boolean Series: which rows of observation `slot` pass each rule.

    A row passes overall only if it passes EVERY rule. Rules:
      present    : the 6 band values exist (not NaN)
      qa_valid   : the organisers' own flag (kept as a first filter)
      not_fill / not_cloud / not_shadow / not_snow / not_cirrus : QA_PIXEL bits
      not_dilated: not on a cloud edge (strict_dilated=True)
      cloud_conf : cloud confidence (bits 8-9) <= max_cloud_conf (1 = low)
      not_saturated : QA_RADSAT == 0 (no band hit the sensor's maximum)
      in_range   : every band inside the valid DN range 7273 to 43636. On the real data this mostly
                   catches fill photos (all 6 bands "too low" together = value 0, no data), the same
                   photos not_fill catches. So it is safe and does not bias any class.
      not_haze   : haze the flags missed. haze_mode="temporal" (DEFAULT, chosen by test on Tue 17 Nov):
                   drop a photo whose Blue is more than 1.35x that pixel's own typical (median) Blue
                   over all its years. Haze is TEMPORARY; a bright new roof is PERMANENT, so it stays.
                   Removes only 0.4% (Madrid) / 0.7% (Amsterdam) of photos and catches patchy haze.
                   haze_mode="scene" drops whole hazy scenes plus any
                   value above 30,000. haze_mode="pixel" is the old rule Blue <= 15,000, which we found
                   also deletes bright NEW buildings (removes 12x more class 4 than class 1 in Madrid).
      Pass bad_scenes (from hazy_scenes) to avoid recomputing it for every slot.
    """
    s = slot
    rules = {}
    bands = df[[f"{b}_{s}" for b in BANDS]]
    rules["present"] = bands.notna().all(axis=1)

    if use_qa_valid and f"qa_valid_{s}" in df:
        rules["qa_valid"] = df[f"qa_valid_{s}"].fillna(False).astype(bool)

    qa_col = f"QA_PIXEL_{s}"
    if qa_col in df:
        qa = df[qa_col].fillna(1).to_numpy(dtype=np.int64)  # missing QA counts as fill
        rules["not_fill"] = ~bit_set(qa, BIT_FILL)
        rules["not_cloud"] = ~bit_set(qa, BIT_CLOUD)
        rules["not_shadow"] = ~bit_set(qa, BIT_SHADOW)
        rules["not_snow"] = ~bit_set(qa, BIT_SNOW)
        rules["not_cirrus"] = ~bit_set(qa, BIT_CIRRUS)
        if strict_dilated:
            rules["not_dilated"] = ~bit_set(qa, BIT_DILATED_CLOUD)
        rules["cloud_conf"] = ((qa >> 8) & 3) <= max_cloud_conf
        # turn numpy arrays back into Series with the same row index
        for k in ["not_fill", "not_cloud", "not_shadow", "not_snow", "not_cirrus", "not_dilated", "cloud_conf"]:
            if k in rules:
                rules[k] = pd.Series(rules[k], index=df.index)

    sat_col = f"QA_RADSAT_{s}"
    if sat_col in df:
        rules["not_saturated"] = df[sat_col].fillna(0).astype(np.int64) == 0

    vals = bands.values
    with np.errstate(invalid="ignore"):
        rules["in_range"] = pd.Series(((vals >= SR_VALID_MIN) & (vals <= SR_VALID_MAX)).all(axis=1),
                                      index=df.index)
        if haze_mode == "pixel":
            rules["not_haze"] = df[f"Blue_{s}"] <= blue_max
        elif haze_mode == "temporal":
            # per photo we can only apply the hard limit here; the temporal check needs every year of
            # the pixel at once, so clean_long_table applies it after this mask
            rules["not_haze"] = df[f"Blue_{s}"] <= BLUE_HARD_MAX
        else:
            if bad_scenes is None:
                bad_scenes = hazy_scenes(df)
            ok = df[f"Blue_{s}"] <= BLUE_HARD_MAX
            if f"scene_id_{s}" in df and bad_scenes:
                ok &= ~df[f"scene_id_{s}"].isin(bad_scenes)
            rules["not_haze"] = ok
    return rules


def mask_report(df, **kw):
    """Table: for each slot, how many observations each rule removes. Great for the EDA slide."""
    out = {}
    if kw.get("haze_mode", "temporal") == "scene" and "bad_scenes" not in kw:
        kw["bad_scenes"] = hazy_scenes(df)
        print(f"hazy scenes found: {len(kw['bad_scenes'])}")
    for s in SLOTS:
        rules = slot_mask(df, s, **kw)
        present = rules["present"]
        row = {"present": int(present.sum())}
        for name, ok in rules.items():
            if name != "present":
                row[f"fails_{name}"] = int((present & ~ok).sum())   # among observations that exist
        all_ok = np.logical_and.reduce([r.values for r in rules.values()])
        row["kept"] = int(all_ok.sum())
        row["kept_%"] = round(100 * all_ok.sum() / max(present.sum(), 1), 1)
        out[f"obs_{s}"] = row
    return pd.DataFrame(out).T


# ---------------------------------------------------------------------------
# 3. ONE CLEAN VALUE PER PIXEL-YEAR
# ---------------------------------------------------------------------------
def clean_long_table(df, combine="median", to_reflectance=False, **mask_kw):
    """Apply the mask and collapse the 3 observation slots into one value per band.

    combine = "first"  : first clean observation (obs 1, else 2, else 3). Same as the notebooks.
            = "median" : median of all clean observations that year. More robust: one leftover
                         hazy scene gets outvoted by the clean ones.
    to_reflectance     : convert DN to real reflectance (DN * 0.0000275 - 0.2).
                         Matters for ratios/indices (see note at the bottom of this file).
    Returns: one row per (pixel, year) with metadata, the 6 bands, and n_clean_obs.
    """
    keep_meta = [c for c in ["pixel_id", "city", "year", "px_key", "py_key", "pixel_x", "pixel_y",
                             "coverage", "weighted_mean_year"] if c in df.columns]
    out = df[keep_meta].copy()

    # stack[slot, row, band] with NaN wherever an observation failed the mask
    if mask_kw.get("haze_mode", "temporal") == "scene" and "bad_scenes" not in mask_kw:
        mask_kw["bad_scenes"] = hazy_scenes(df)
    stack = np.full((len(SLOTS), len(df), len(BANDS)), np.nan, dtype=np.float32)
    for i, s in enumerate(SLOTS):
        rules = slot_mask(df, s, **mask_kw)
        ok = np.logical_and.reduce([r.values for r in rules.values()])
        vals = df[[f"{b}_{s}" for b in BANDS]].values.astype(np.float32)
        stack[i][ok] = vals[ok]

    if mask_kw.get("haze_mode", "temporal") == "temporal":
        # Temporal haze check: compare every photo's Blue with the median Blue of ALL clean photos of
        # the same pixel (all years, all 3 slots). Temporarily much brighter = haze -> removed.
        key = (df["px_key"].astype(np.int64) * 10_000_000 + df["py_key"].astype(np.int64)).to_numpy()
        blue_all = stack[:, :, 0].reshape(-1)                      # slot-major: slot 1 rows, slot 2 rows, ...
        key_all = np.tile(key, len(SLOTS))
        typical = (pd.Series(blue_all).groupby(key_all).transform("median")
                   .to_numpy().reshape(len(SLOTS), -1))
        with np.errstate(invalid="ignore"):
            hazy = stack[:, :, 0] > TEMPORAL_FACTOR * typical
        n_valid = int(np.isfinite(stack[:, :, 0]).sum())
        stack[hazy] = np.nan
        print(f"temporal haze filter removed {int(hazy.sum()):,} photos "
              f"({100 * hazy.sum() / max(n_valid, 1):.2f}% of clean photos)")

    if combine == "median":
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # rows with no clean obs -> NaN
            combined = np.nanmedian(stack, axis=0)
    elif combine == "first":
        combined = stack[0].copy()
        for i in range(1, len(SLOTS)):
            empty = np.isnan(combined)
            combined[empty] = stack[i][empty]
    else:
        raise ValueError("combine must be 'median' or 'first'")

    if to_reflectance:
        combined = combined * SR_SCALE + SR_OFFSET

    for j, b in enumerate(BANDS):
        out[b] = combined[:, j]
    out["n_clean_obs"] = (~np.isnan(stack[:, :, 0])).sum(axis=0)
    return out.dropna(subset=BANDS).reset_index(drop=True)


def add_age_class(df, event_year):
    """Bins (-inf, event], (event, 1984], (1984, 2004], (2004, inf) -> classes 1 to 4.
    Right-closed like the official notebooks (exactly 1984 -> class 2). Confirm with organisers."""
    out = df.copy()
    out["age_class"] = pd.cut(out["weighted_mean_year"], [-np.inf, event_year, 1984, 2004, np.inf],
                              labels=[1, 2, 3, 4]).astype(int)
    return out


# NOTE ON to_reflectance:
# Band values like Blue = 9,184 are Collection 2 scaled integers, not reflectance.
# Real reflectance = DN * 0.0000275 - 0.2. Because of the -0.2 offset, an index such as
# NDVI = (NIR - Red) / (NIR + Red) gives a DIFFERENT answer on DN than on reflectance.
# Example: NIR 16,000, Red 11,000 -> NDVI on DN = 0.19, on reflectance = 0.40.
# Ask the organisers whether the files are already harmonised reflectance or DN, and tell
# the features person (Fenny): indices should be computed on reflectance.
