import numpy as np
import pandas as pd
import pytest

from flood.climate import (
    gumbel_fit,
    gumbel_return_level,
    mann_kendall,
    rainfall_products,
    rate_to_monthly_total,
)


def test_rate_to_total_uses_days_in_month():
    rate = pd.DataFrame({2020: [1.0, 1.0], 2021: [1.0, 1.0]}, index=[1, 2])
    total = rate_to_monthly_total(rate)
    assert total.loc[2, 2020] == 29
    assert total.loc[2, 2021] == 28
    assert total.loc[1, 2021] == 31


def test_mann_kendall_detects_monotonic_trend():
    up = mann_kendall(np.arange(20, dtype=float))
    assert up["Z"] > 0 and up["p"] < 0.001
    flat = mann_kendall(np.array([1.0, 3.0, 2.0, 3.0, 1.0, 2.0]))
    assert flat["p"] > 0.5


def test_gumbel_return_levels():
    mu, beta = gumbel_fit(np.array([1500.0, 1700, 1800, 1900, 2100, 2500]))
    assert gumbel_return_level(mu, beta, 100) > gumbel_return_level(mu, beta, 10)
    # the 2-year level is the median of the Gumbel distribution
    assert gumbel_return_level(mu, beta, 2) == pytest.approx(mu - beta * np.log(np.log(2)))


def test_rainfall_products():
    stack = np.array(
        [[[10.0, np.nan]], [[20.0, np.nan]], [[30.0, np.nan]], [[40.0, np.nan]]], dtype="float32"
    )
    out = rainfall_products(stack, [2020, 2021], {2020: [0, 1], 2021: [2, 3]}, 95)
    assert out["rainfall_total_2020"][0, 0] == 30
    assert out["rainfall_total_2021"][0, 0] == 70
    assert out["rainfall_mean_monthly"][0, 0] == 25
    assert out["rainfall_max_monthly"][0, 0] == 40
    assert np.isnan(out["rainfall_p95_monthly"][0, 1])


def test_fill_nearest_only_inside():
    from flood.preprocess import fill_nearest

    arr = np.array([[1.0, np.nan, np.nan], [2.0, 2.0, np.nan]], dtype="float32")
    inside = np.array([[True, True, False], [True, True, True]])
    out = fill_nearest(arr, inside)
    assert out[0, 1] == 1.0
    assert out[1, 2] == 2.0
    assert np.isnan(out[0, 2])
