"""Building-age Stage 1 / Stage 2 prediction from VERIFIED feature matrices.

No Amsterdam query labels are accepted by the inference API.
The caller supplies engineered 102-feature + observed 252-year-band arrays.
Raw parquet preprocessing must be validated separately before arbitrary-city deployment.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

CLASSES = np.array([1, 2, 3, 4], dtype=np.int64)
ALLOWED_SHOTS = (5, 25, 50, 100, 200)
WEIGHTS = np.array([0.20, 0.35, 0.45], dtype=np.float64)
CONTINUATION = dict(objective='multiclass', n_estimators=120, learning_rate=0.03,
                    num_leaves=15, min_child_samples=20, reg_lambda=2.0,
                    random_state=20261117, n_jobs=2, verbosity=-1)


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _feature_pair(x102, annual):
    """Reject wrong feature dimensions, infinities and missing original features."""
    x102 = np.asarray(x102, dtype=np.float64)
    annual = np.asarray(annual, dtype=np.float32)
    if x102.ndim != 2 or x102.shape[1] != 102:
        raise ValueError(f'Expected 102 original features, got {x102.shape}')
    if annual.ndim != 2 or annual.shape != (len(x102), 252):
        raise ValueError('Annual matrix must be (same rows, 252)')
    if not np.isfinite(x102).all():
        raise ValueError('Original 102 features must be finite')
    if np.isinf(annual).any():
        raise ValueError('Annual 252 features may contain NaN, never infinity')
    return x102, annual


def _prob(model, x, name):
    p = np.asarray(model.predict_proba(x), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=int)
    if p.shape != (len(x), 4) or not np.array_equal(classes, CLASSES):
        raise ValueError(f'{name} model has wrong class order or output shape')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError(f'{name} probabilities are invalid')
    return p


@dataclass
class Stage2Predictor:
    """Frozen Madrid RF60/LGBM354 + Amsterdam support-only LDA78.

    Stage 2 adaptation changes only the LightGBM continuation and local LDA;
    the two stored Madrid models remain unchanged.
    """
    rf: object
    lgbm: object
    lda_columns: np.ndarray
    feature_names102: np.ndarray
    feature_names252: np.ndarray

    def __post_init__(self):
        self.lda_columns = np.asarray(self.lda_columns, dtype=np.int64)
        if self.lda_columns.shape != (78,):
            raise ValueError('Expected exactly 78 LDA feature indices')
        if not np.array_equal(self.lda_columns[:60], np.arange(60)):
            raise ValueError('First 60 LDA indices are not the original 60 features')
        if len(np.unique(self.lda_columns)) != 78 or np.any(self.lda_columns >= 102) or np.any(self.lda_columns < 0):
            raise ValueError('Invalid LDA feature indices')
        if len(self.feature_names102) != 102 or len(self.feature_names252) != 252:
            raise ValueError('Incorrect saved feature schema')
        if self.rf.n_features_in_ != 60 or self.lgbm.n_features_in_ != 354:
            raise ValueError('Madrid model feature dimensions do not match')
        for model in (self.rf, self.lgbm):
            if not np.array_equal(model.classes_, CLASSES):
                raise ValueError('Unexpected Madrid class order')
        self.local_lda = None
        self.local_scaler = None
        self.active_columns = None
        self.local_lgbm = None
        self.shots_per_class = None

    def predict_stage1(self, x102, annual):
        """Madrid-trained Stage 1 model without Amsterdam adaptation."""
        a, b = _feature_pair(x102, annual)
        full = np.column_stack([a, b]).astype(np.float32)
        p = _prob(self.lgbm, full, 'Madrid LightGBM354')
        return CLASSES[np.argmax(p, axis=1)], p

    def adapt(self, support102, support_annual, support_labels, n=None):
        """Adapt only on n labelled target-city examples per class."""
        x102, annual = _feature_pair(support102, support_annual)
        y = np.asarray(support_labels, dtype=int)
        if y.shape != (len(x102),):
            raise ValueError('Support labels must align with support rows')
        if len(x102) == 0 or not np.isin(y, CLASSES).all():
            raise ValueError('Support labels must belong to classes 1..4')
        counts = np.array([(y == c).sum() for c in CLASSES])
        if not np.all(counts == counts[0]):
            raise ValueError(f'Support is unbalanced: {counts.tolist()}')
        n_actual = int(counts[0])
        if n is not None and int(n) != n_actual:
            raise ValueError(f'Requested n={n}, but actual n={n_actual} per class')
        if n_actual not in ALLOWED_SHOTS:
            raise ValueError(f'n must be one of {ALLOWED_SHOTS}, got {n_actual}')

        local_train = x102[:, self.lda_columns]
        self.active_columns = np.var(local_train, axis=0) > 1e-12
        if not np.any(self.active_columns):
            raise ValueError('No informative LDA features in support set')
        self.local_scaler = StandardScaler()
        transformed = self.local_scaler.fit_transform(local_train[:, self.active_columns])
        self.local_lda = LinearDiscriminantAnalysis(
            solver='lsqr', shrinkage='auto', priors=np.full(4, 0.25))
        self.local_lda.fit(transformed, y)
        if not np.array_equal(self.local_lda.classes_, CLASSES):
            raise ValueError('Amsterdam LDA classes mismatch')

        # Fixed development policy: 5 shots -> frozen Madrid model;
        # 25/50/100/200 shots -> 120 additional support-only LGBM rounds.
        self.local_lgbm = None
        if n_actual >= 25:
            features = np.column_stack([x102, annual]).astype(np.float32)
            continued = LGBMClassifier(**CONTINUATION)
            continued.fit(features, y, init_model=self.lgbm.booster_)
            if not np.array_equal(continued.classes_, CLASSES):
                raise ValueError('Adapted LightGBM class order mismatch')
            self.local_lgbm = continued
        self.shots_per_class = n_actual
        return self

    def predict_proba(self, query102, query_annual):
        """Predict without ANY access to query labels."""
        if self.shots_per_class is None:
            raise RuntimeError('Call adapt(...) before predict(...)')
        x102, annual = _feature_pair(query102, query_annual)
        rf_p = _prob(self.rf, x102[:, :60], 'Madrid RF60')
        lgbm_features = np.column_stack([x102, annual]).astype(np.float32)
        gbm = self.local_lgbm if self.local_lgbm is not None else self.lgbm
        gbm_p = _prob(gbm, lgbm_features, 'LightGBM354')
        lda_x = x102[:, self.lda_columns][:, self.active_columns]
        lda_p = _prob(self.local_lda, self.local_scaler.transform(lda_x), 'Amsterdam LDA78')
        p = WEIGHTS[0] * rf_p + WEIGHTS[1] * gbm_p + WEIGHTS[2] * lda_p
        if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
            raise ValueError('Ensemble probabilities do not sum to 1')
        return p

    def predict(self, query102, query_annual):
        p = self.predict_proba(query102, query_annual)
        return CLASSES[np.argmax(p, axis=1)]


def load_bundle(directory):
    """Only load a bundle created by our build script, from a TRUSTED location."""
    folder = Path(directory)
    with (folder / 'manifest.json').open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    for name, hexdigest in manifest['sha256'].items():
        actual = _hash(folder / name)
        if actual != hexdigest:
            raise ValueError(f'Bundle integrity failure for {name}')
    rf = joblib.load(folder / 'madrid_rf60.joblib')
    lgbm = joblib.load(folder / 'madrid_lgbm354.joblib')
    schema = np.load(folder / 'schema.npz', allow_pickle=False)
    return Stage2Predictor(
        rf, lgbm,
        schema['lda_columns'].copy(),
        schema['feature_names102'].copy(),
        schema['feature_names252'].copy())


def feature_inputs_from_archives(base_npz, annual_npz, city):
    """For organizer-provided / previously verified feature matrices only.

    This helper is for reproducibility and demonstrations; it is NOT a
    substitute for raw-parquet preprocessing on unseen pixels.
    """
    if city not in ('madrid', 'amsterdam'):
        raise ValueError('city must be madrid or amsterdam')
    with np.load(base_npz, allow_pickle=False) as d:
        x102 = d[f'X_{city}'].copy()
        ids = d[f'pixel_ids_{city}'].copy()
    with np.load(annual_npz, allow_pickle=False) as d:
        if not np.array_equal(ids, d['pixel_ids']):
            raise ValueError('Annual features are not aligned by pixel IDs')
        annual = d['features'].copy()
    _feature_pair(x102, annual)
    return ids, x102, annual
