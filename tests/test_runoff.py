import numpy as np
import pytest

from flood.runoff import curve_numbers, hydrologic_soil_group, scs_runoff
from flood.validation import amc_dry, amc_wet


def test_soil_groups_from_clay():
    clay = np.array([5, 10, 19.9, 20, 39, 40, np.nan], dtype="float32")
    assert hydrologic_soil_group(clay, [10, 20, 40]).tolist() == [1, 2, 2, 3, 3, 4, 0]
    assert hydrologic_soil_group(clay, [10, 20, 40], missing_group=1)[-1] == 1


def test_curve_number_lookup_and_default():
    lulc = np.array([1, 2, 3, 9, 0], dtype="uint8")
    hsg = np.array([4, 1, 2, 2, 2], dtype="uint8")
    table = {1: [77, 85, 90, 92], 2: [55, 70, 77, 83], 3: [98, 98, 98, 98]}
    cn = curve_numbers(lulc, hsg, table, 75)
    assert cn[:4].tolist() == [92, 55, 98, 75]
    assert np.isnan(cn[4])


def test_scs_runoff_textbook_value():
    # CN 80: S = 63.5 mm, Ia = 12.7 mm; P = 100 mm gives Q = 87.3^2 / 150.8
    q = scs_runoff(np.array([100.0], "float32"), np.array([80.0], "float32"))
    assert q[0] == pytest.approx(87.3**2 / 150.8, rel=1e-5)


def test_no_runoff_below_initial_abstraction_and_nan_without_cn():
    q = scs_runoff(np.array([5.0, 50.0], "float32"), np.array([80.0, np.nan], "float32"))
    assert q[0] == 0
    assert np.isnan(q[1])


def test_amc_conversions_bracket_amc2():
    cn = np.array([60.0, 80.0])
    assert np.all(amc_dry(cn) < cn)
    assert np.all(amc_wet(cn) > cn)
