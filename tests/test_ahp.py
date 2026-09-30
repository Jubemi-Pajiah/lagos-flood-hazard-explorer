import numpy as np
import pytest

from flood.config import ConfigError
from flood.hazard import ahp_weights, pairwise_matrix, parse_judgement, weights


def test_parse_judgement():
    assert parse_judgement("1/3") == pytest.approx(1 / 3)
    assert parse_judgement(5) == 5.0


def test_reciprocal_matrix():
    a = pairwise_matrix([[1, 3, "1/2"], [1, 2], [1]])
    assert a[1, 0] == pytest.approx(1 / 3)
    assert a[2, 0] == pytest.approx(2)
    assert np.allclose(a * a.T, 1.0)


def test_consistent_matrix_has_zero_cr():
    w = np.array([0.5, 0.3, 0.2])
    result = ahp_weights(w[:, None] / w[None, :])
    assert np.allclose(result["weights"], w)
    assert result["cr"] == pytest.approx(0.0, abs=1e-9)


def test_lagos_matrix(region):
    result = ahp_weights(pairwise_matrix(region["hazard"]["pairwise_matrix"]))
    assert result["lambda_max"] == pytest.approx(8.2861, abs=1e-4)
    assert result["cr"] == pytest.approx(0.0290, abs=1e-4)
    assert result["cr"] < 0.10


def test_weights_sum_to_one(region):
    assert sum(weights(region).values()) == pytest.approx(1.0)
    region.settings["hazard"]["factors"][0]["weight"] = 0.3
    with pytest.raises(ConfigError, match="sum"):
        weights(region)


def test_bad_matrix_row_length():
    with pytest.raises(ConfigError, match="row 1"):
        pairwise_matrix([[1, 2], [1, 3], [1]])
