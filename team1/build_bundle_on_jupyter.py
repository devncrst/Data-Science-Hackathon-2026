"""Run on the trusted hosted JupyterLab server to build an actual model bundle.

This program has no access to remote files until you upload this source directory
and run it on the JupyterLab server containing the saved hackathon_results.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import hashlib
import platform
import zipfile
import numpy as np
import joblib
import sklearn
import lightgbm
from age_hackathon.predictor import load_bundle, feature_inputs_from_archives

ROOT = Path.home() / 'hackathon_results'
DEST = ROOT / 'presentation_day_submission_20261118'
BUNDLE = DEST / 'model_bundle'
ZIP_PATH = DEST / 'BuildingAge_Stage2_Submission.zip'

RF_PATH = ROOT / 'exp015a_model_freeze_v1/models/madrid_rf60.joblib'
LGBM_PATH = ROOT / 'exp017d_full_history_transfer/madrid_lgbm354_full.joblib'
FEAT_PATH = ROOT / 'exp013h_102feature_candidate.npz'
ANNUAL_PATH = ROOT / 'exp017c_amsterdam_annual_history/amsterdam_annual_252features.npz'
REF_RESULTS = ROOT / 'exp017f_random_support_354/lgbm102_vs354_10trials.npz'
REF_CHECKPOINT = ROOT / 'exp017f_random_support_354/trial_checkpoints/trial_01.npz'
BACKUP = ROOT / 'private_recovery/verified_fixed_data_backup.npz'
SOURCE_FOLDER = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build():
    for p in (RF_PATH, LGBM_PATH, FEAT_PATH, ANNUAL_PATH, REF_RESULTS, REF_CHECKPOINT, BACKUP):
        if not p.is_file():
            raise FileNotFoundError(f'Missing required trusted server artifact: {p}')
    if DEST.exists():
        raise FileExistsError(f'Won\'t overwrite existing submission: {DEST}')
    BUNDLE.mkdir(parents=True)
    print('Source artifacts: FOUND', flush=True)

    with np.load(FEAT_PATH, allow_pickle=False) as d:
        names102 = d['feature_names'].astype(str)
        ams_ids = d['pixel_ids_amsterdam'].copy()
    with np.load(ANNUAL_PATH, allow_pickle=False) as d:
        names252 = d['feature_names'].astype(str)
        np.testing.assert_array_equal(ams_ids, d['pixel_ids'])
    with np.load(REF_RESULTS, allow_pickle=False) as d:
        lda_columns = d['lda78_columns'].copy()
        np.testing.assert_allclose(d['ensemble_weights'], [0.2, 0.35, 0.45], atol=0, rtol=0)
        assert int(d['continuation_rounds'][0]) == 120
        assert d['scores'].shape == (10, 2)
        np.testing.assert_allclose(d['scores'][:, 1].mean(), 0.7232, atol=0.0001)
    assert names102.shape == (102,) and names252.shape == (252,)
    assert lda_columns.shape == (78,) and np.unique(lda_columns).size == 78
    np.testing.assert_array_equal(lda_columns[:60], np.arange(60))
    print('Exact 354-feature schema and LDA columns: VERIFIED', flush=True)

    for source, name in ((RF_PATH, 'madrid_rf60.joblib'), (LGBM_PATH, 'madrid_lgbm354.joblib')):
        shutil.copy2(source, BUNDLE / name)
        assert sha(source) == sha(BUNDLE / name)
    np.savez_compressed(BUNDLE / 'schema.npz',
                        feature_names102=names102, feature_names252=names252,
                        lda_columns=lda_columns, classes=np.arange(1, 5),
                        ensemble_weights=np.array([0.20, 0.35, 0.45]),
                        continuation_rounds=np.array([120]),
                        low_shot_rounds=np.array([0]))

    model_names = ['madrid_rf60.joblib', 'madrid_lgbm354.joblib', 'schema.npz']
    manifest = {
        'name': 'Madrid -> Amsterdam building-age few-shot transfer',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'feature-level Stage 1/Stage 2 reproducibility bundle',
        'classes': [1, 2, 3, 4], 'accepted_shots_per_class': [5, 25, 50, 100, 200],
        'feature_columns': {'original102': 102, 'raw_annual': 252, 'combined': 354},
        'mixture': {'RF60': 0.20, 'continued_LGBM354': 0.35, 'support_LDA78': 0.45},
        'continuation_policy': {'shots_5': 0, 'shots_25_or_more': 120},
        'baseline_reference': 'Experiment 017F, 200 shots/class, development only, mean Macro F1 0.7232',
        'limitations': 'Raw Parquet to 102-feature provenance must be separately verified for arbitrary new hidden pixels; do not claim an end-to-end raw-Parquet interface.',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scikit_learn': sklearn.__version__, 'lightgbm': lightgbm.__version__,
                        'joblib': joblib.__version__},
        'sha256': {name: sha(BUNDLE / name) for name in model_names}
    }
    (BUNDLE / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    model = load_bundle(BUNDLE)
    print('Models reloaded and SHA256 verified', flush=True)

    # Exact reproducibility against previous model outputs; QUERY LABELS ARE NOT READ.
    with np.load(REF_CHECKPOINT, allow_pickle=False) as d:
        support_idx, query_idx = d['support_indices'].copy(), d['query_indices'].copy()
        expected_predictions = d['predictions'][1].copy()
    ids, x102, annual = feature_inputs_from_archives(FEAT_PATH, ANNUAL_PATH, 'amsterdam')
    np.testing.assert_array_equal(ids, ams_ids)
    with np.load(BACKUP, allow_pickle=False) as d:
        y_support = d['y_amsterdam'][support_idx].copy()  # ONLY support labels
    assert len(support_idx) == 800 and len(query_idx) == 25192
    print('Reproducing trial 1 from 017F (200 shots/class) ...', flush=True)
    model.adapt(x102[support_idx], annual[support_idx], y_support, n=200)
    predicted = model.predict(x102[query_idx], annual[query_idx])
    matches = np.mean(predicted == expected_predictions)
    print(f'Exact reproduction against stored 017F predictions: {matches:.4%}', flush=True)
    if not np.array_equal(predicted, expected_predictions):
        raise AssertionError('Stage 2 predictions DO NOT reproduce Experiment 017F; do not submit this bundle.')

    # Five-shot sanity check, without querying any query labels.
    sub = np.concatenate([np.flatnonzero(y_support == c)[:5] for c in [1, 2, 3, 4]])
    model.adapt(x102[support_idx[sub]], annual[support_idx[sub]], y_support[sub], n=5)
    small = model.predict(x102[query_idx[:20]], annual[query_idx[:20]])
    assert small.shape == (20,) and np.isin(small, [1, 2, 3, 4]).all()
    print('Five-shot no-continuation smoke test: PASSED', flush=True)

    # Copy code and ready-to-use README; exclude all target labels / archived predictions.
    pkg = DEST / 'age_hackathon'
    shutil.copytree(SOURCE_FOLDER / 'age_hackathon', pkg)
    shutil.copy2(SOURCE_FOLDER / 'README.md', DEST / 'README.md')
    shutil.copy2(SOURCE_FOLDER / 'example_predict.py', DEST / 'example_predict.py')
    shutil.copy2(SOURCE_FOLDER / 'run_stage2.py', DEST / 'run_stage2.py')
    shutil.copy2(SOURCE_FOLDER / 'requirements.txt', DEST / 'requirements.txt')
    files = [p for p in DEST.rglob('*') if p.is_file() and p != ZIP_PATH and '__pycache__' not in p.parts]
    with zipfile.ZipFile(ZIP_PATH, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in files:
            z.write(p, p.relative_to(DEST))
    print(f'COMPLETE: {ZIP_PATH}', flush=True)
    print(f'Model files included: {[(p.name, round(p.stat().st_size/1e6,1)) for p in BUNDLE.glob("*.joblib")]}')
    print('PRIVATE Amsterdam labels and preprocessed datasets INCLUDED IN ZIP: NO')
    print('IMPORTANT: raw-to-feature verification STILL needed for arbitrary hidden Parquet rows.')
    return ZIP_PATH


if __name__ == '__main__':
    build()
