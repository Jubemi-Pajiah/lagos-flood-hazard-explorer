"""Check a full run from raw data against the published baseline in tests/golden/lagos/baseline.

Run the pipeline first (``python -m flood run --region lagos``); these tests read
``outputs/lagos`` and are skipped when it is missing.

Tolerances: tables exact; landcover accuracy to 3 decimals; runoff means to 0.1 mm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from flood.config import load_region

BASELINE = Path(__file__).parent / "golden" / "lagos" / "baseline"

pytestmark = pytest.mark.golden


@pytest.fixture(scope="module")
def outputs():
    region = load_region("lagos")
    out = region.out()
    if not (out / "validation" / "hotspot_check.json").exists():
        pytest.skip("no pipeline outputs in outputs/lagos")
    return out


@pytest.mark.parametrize(
    "name",
    [
        "hazard/hazard_zone_areas.csv",
        "exposure/exposure_by_zone.csv",
        "exposure/exposure_by_lga.csv",
    ],
)
def test_tables(outputs, name):
    pd.testing.assert_frame_equal(
        pd.read_csv(outputs / name), pd.read_csv(BASELINE / Path(name).name)
    )


def test_hotspots(outputs):
    new = json.loads((outputs / "validation" / "hotspot_check.json").read_text())
    assert new == json.loads((BASELINE / "hotspot_check.json").read_text())


def test_landcover_accuracy(outputs):
    new = json.loads((outputs / "landcover" / "accuracy.json").read_text())
    old = json.loads((BASELINE / "landcover_accuracy.json").read_text())
    assert new["overall_accuracy"] == pytest.approx(old["overall_accuracy"], abs=1e-3)
    assert new["kappa"] == pytest.approx(old["kappa"], abs=1e-3)


def test_runoff_summary(outputs):
    new = json.loads((outputs / "runoff" / "runoff_summary.json").read_text())
    old = json.loads((BASELINE / "runoff_summary.json").read_text())
    assert new["runoff_mm_mean"] == pytest.approx(old["runoff_mm_mean"], abs=0.1)
    assert new["curve_number"] == pytest.approx(old["curve_number"], abs=0.01)
