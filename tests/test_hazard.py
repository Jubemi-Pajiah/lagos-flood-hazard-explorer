import numpy as np

from flood.hazard import apply_bounds, classify, hazard_index, normalise


def test_normalise_clips_and_reverses():
    arr = np.arange(101, dtype="float32")
    norm, (lo, hi) = normalise(arr, 2, 98, "benefit")
    assert (lo, hi) == (2.0, 98.0)
    assert norm[0] == 0 and norm[-1] == 1
    assert norm[50] == np.float32(0.5)
    cost, _ = normalise(arr, 2, 98, "cost")
    assert np.allclose(cost, 1 - norm)


def test_normalise_keeps_nan_and_dtype():
    arr = np.array([1, 2, np.nan, 4], dtype="float32")
    norm, _ = normalise(arr, 0, 100, "benefit")
    assert np.isnan(norm[2])
    assert norm.dtype == np.float32


def test_flat_factor_is_zero():
    assert np.all(apply_bounds(np.ones(4, "float32"), 1.0, 1.0, "benefit") == 0)


def test_index_is_weighted_sum_and_needs_every_factor():
    a = np.array([[1.0, 0.0], [0.5, np.nan]], dtype="float32")
    b = np.array([[0.0, 1.0], [0.5, 0.5]], dtype="float32")
    index, valid = hazard_index({"a": a, "b": b}, {"a": 0.75, "b": 0.25})
    assert np.allclose(index[0], [0.75, 0.25])
    assert index[1, 0] == np.float32(0.5)
    assert np.isnan(index[1, 1]) and not valid[1, 1]


def test_classify_breaks_lower_inclusive():
    index = np.array([0.0, 0.2, 0.3999, 0.8, 1.0, np.nan], dtype="float32")
    zones = classify(index, [0.0, 0.2, 0.4, 0.6, 0.8, 1.01])
    assert zones.tolist() == [1, 2, 2, 5, 5, 0]
