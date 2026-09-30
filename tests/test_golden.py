"""Regression tests against the original run, saved in tests/golden/lagos/original_run.

The downstream stages (hazard, exposure, validation) are replayed on the saved golden
intermediates, so these tests check that the refactored code reproduces the saved
results. They need FLOOD_DATA_DIR (for WorldPop and admin units) and FLOOD_GOLDEN_DIR (the
folder of saved rasters) and are skipped otherwise.

Tolerances:
* hazard zones: exact, cell for cell;
* hazard index: 1e-6 absolute (float32 rounding of the weighted sum);
* LGA exposure: exact except 1 person per cell, from rounding in the rescaling step of
  the original script;
* statewide exposure, zone areas, hotspot results: exact;
* AHP and runoff sensitivity numbers: 1e-6 relative.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from golden_replay import GOLDEN_OUTPUTS, golden_dir, seed

from flood import exposure, hazard, preprocess, validation
from flood.config import load_region

GOLDEN_TABLES = Path(__file__).parent / "golden" / "lagos" / "original_run"

pytestmark = [pytest.mark.golden, pytest.mark.slow]


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    golden = golden_dir()
    region = load_region("lagos", output_dir=tmp_path_factory.mktemp("replay"))
    if golden is None or region.data_dir is None or not region.data_dir.exists():
        pytest.skip("needs FLOOD_DATA_DIR and FLOOD_GOLDEN_DIR")
    seed(region, golden)
    preprocess.prepare_boundaries(region)
    hazard.run(region)
    exposure.run(region)
    validation.run(region)
    return region, golden


def _read(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def _json(name):
    return json.loads((GOLDEN_TABLES / name).read_text(encoding="utf-8"))


def test_hazard_rasters(replay):
    region, golden = replay
    zones = _read(hazard.zones_path(region))
    assert np.array_equal(
        np.nan_to_num(zones), np.nan_to_num(_read(golden / GOLDEN_OUTPUTS["hazard_zones"]))
    )
    index = _read(hazard.index_path(region))
    old = _read(golden / GOLDEN_OUTPUTS["hazard_index"])
    assert np.array_equal(np.isnan(index), np.isnan(old))
    assert np.nanmax(np.abs(index - old)) < 1e-6


def test_zone_areas(replay):
    region, _ = replay
    new = pd.read_csv(hazard.zone_areas_path(region))
    pd.testing.assert_frame_equal(new, pd.read_csv(GOLDEN_TABLES / "hazard_zone_areas.csv"))


def test_ahp(replay):
    region, _ = replay
    new = json.loads(hazard.ahp_path(region).read_text(encoding="utf-8"))
    old = _json("ahp.json")
    assert new["weights"] == old["weights"]
    for key in ("lambda_max", "ci", "cr"):
        assert new[key] == pytest.approx(old[key], rel=1e-6)


def test_exposure_by_lga(replay):
    region, _ = replay
    new = pd.read_csv(exposure.by_lga_path(region))
    old = pd.read_csv(GOLDEN_TABLES / "exposure_by_lga.csv")
    assert new.lga_name.tolist() == old.lga_name.tolist()
    counts = [c for c in old.columns if c.startswith(("zone", "total"))]
    assert (new[counts] - old[counts]).abs().to_numpy().max() <= 1
    assert new.lga_id.tolist() == old.lga_id.tolist()
    assert new.pct_high_very_high.tolist() == old.pct_high_very_high.tolist()


def test_exposure_by_zone(replay):
    region, _ = replay
    pd.testing.assert_frame_equal(
        pd.read_csv(exposure.by_zone_path(region)),
        pd.read_csv(GOLDEN_TABLES / "exposure_by_zone.csv"),
    )


def test_hotspots(replay):
    region, _ = replay
    new = json.loads(validation.hotspot_summary_path(region).read_text(encoding="utf-8"))
    assert new == _json("hotspot_check.json")


def test_runoff_sensitivity(replay):
    region, _ = replay
    new = json.loads(validation.sensitivity_summary_path(region).read_text(encoding="utf-8"))
    old = _json("runoff_sensitivity.json")
    assert new["baseline_mean_runoff_mm"] == pytest.approx(old["baseline_mean_runoff_mm"], rel=1e-6)
    for key, span in old["range_by_change_pct"].items():
        assert new["range_by_change_pct"][key] == pytest.approx(span, rel=1e-6)
    assert new["range_amc"] == pytest.approx(old["range_amc"], rel=1e-6)
