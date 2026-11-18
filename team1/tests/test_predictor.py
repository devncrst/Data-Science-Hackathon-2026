import numpy as np
import pytest
from age_hackathon.predictor import _feature_pair, CLASSES, Stage2Predictor, WEIGHTS, ALLOWED_SHOTS


def test_dimensions_and_nan_policy():
    x102 = np.ones((2,102))
    a = np.ones((2,252))
    _feature_pair(x102, a)
    a[0,1] = np.nan
    _feature_pair(x102, a)  # annual missing years allowed
    with pytest.raises(ValueError):
        _feature_pair(x102[:, :60], a)
    with pytest.raises(ValueError):
        _feature_pair(x102, a[:1])
    x102[0,0] = np.nan
    with pytest.raises(ValueError):
        _feature_pair(x102, a)


def test_weights_classes_and_budgets():
    assert list(CLASSES) == [1,2,3,4]
    assert list(ALLOWED_SHOTS) == [5,25,50,100,200]
    np.testing.assert_allclose(WEIGHTS.sum(), 1.0)


def test_api_has_no_query_labels():
    import inspect
    names = list(inspect.signature(Stage2Predictor.predict).parameters)
    assert 'y_query' not in names and 'query_labels' not in names
