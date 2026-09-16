import numpy as np
import pytest

from backend.src.benchmark_flow import score


def test_endpoint_error_and_nonfinite_coverage():
    target = np.zeros((2, 2, 2), dtype=np.float32)
    prediction = np.array([[[3, 4], [0, 0]], [[np.nan, 0], [99, 99]]])
    mask = np.array([[True, True], [True, False]])
    r = score(prediction, target, mask)
    assert r['eligible'] == 3 and r['predicted'] == 2
    assert r['prediction_coverage'] == pytest.approx(2 / 3)
    assert r['mean_epe'] == 2.5
    assert r['median_epe'] == 2.5
    assert r['fraction_gt3px'] == .5


def test_empty_support_is_not_zero_error():
    flow = np.zeros((2, 3, 2), dtype=np.float32)
    r = score(flow, flow, np.zeros((2, 3), dtype=bool))
    assert r['eligible'] == 0
    assert r['mean_epe'] is None and r['prediction_coverage'] is None


def test_truth_magnitude_does_not_depend_on_prediction_failures():
    target = np.array([[[3., 4.], [0., 0.]]])
    prediction = np.array([[[np.nan, 0.], [0., 0.]]])
    r = score(prediction, target, np.ones((1, 2), dtype=bool))
    assert r['mean_truth_magnitude'] == 2.5
    assert r['prediction_coverage'] == .5


def test_malformed_arrays_rejected():
    flow = np.zeros((2, 3, 2))
    with pytest.raises(ValueError):
        score(flow, flow, np.ones((2, 3)))
    with pytest.raises(ValueError):
        score(flow, flow, np.ones((3, 3), dtype=bool))
