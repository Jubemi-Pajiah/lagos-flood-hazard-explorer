import json

import numpy as np
import pandas as pd
import pytest

from flood.api import Explorer

# A 2 x 2 grid of 1 km cells in UTM 31N around Lagos Island.
TRANSFORM = [1000.0, 0.0, 541000.0, 0.0, -1000.0, 715000.0]


@pytest.fixture
def app_dir(tmp_path):
    meta = {
        "region": "Test",
        "crs": "EPSG:32631",
        "transform": TRANSFORM,
        "shape": [2, 2],
        "cell_size_m": 1000.0,
        "zones": {
            str(z): {"name": n, "color": "#000000"}
            for z, n in enumerate(["Very Low", "Low", "Moderate", "High", "Very High"], 1)
        },
        "high_zones": [4, 5],
        "weights": {"rainfall": 0.08, "runoff": 0.12},
        "admin_label": "LGA",
        "admin_name_field": "shapeName",
        "admin_name_map": {"A": "Alpha"},
        "baseline": "rain_x1.00",
        "scenarios": [
            {
                "id": "rain_x1.00",
                "label": "Baseline",
                "kind": "rainfall_scale",
                "rainfall_scale": 1.0,
            },
            {
                "id": "rain_x1.50",
                "label": "Rainfall x1.50",
                "kind": "rainfall_scale",
                "rainfall_scale": 1.5,
            },
        ],
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    np.savez_compressed(
        tmp_path / "zones_rain_x1.00.npz", zones=np.array([[3, 4], [0, 5]], "uint8")
    )
    np.savez_compressed(
        tmp_path / "zones_rain_x1.50.npz", zones=np.array([[4, 4], [0, 5]], "uint8")
    )
    np.savez_compressed(
        tmp_path / "hazard_index.npz", index=np.array([[0.5, 0.7], [np.nan, 0.9]], "float32")
    )
    rows = []
    for sc, pops in (("rain_x1.00", [0, 0, 60, 30, 10]), ("rain_x1.50", [0, 0, 40, 50, 10])):
        rows += [
            {"zone": z, "name": "", "population": p, "scenario": sc} for z, p in enumerate(pops, 1)
        ]
    pd.DataFrame(rows).to_csv(tmp_path / "exposure_by_zone.csv", index=False)
    pd.DataFrame(
        [{"lga_name": "Alpha", "pct_high_very_high": 40.0, "scenario": "rain_x1.00"}]
    ).to_csv(tmp_path / "exposure_by_lga.csv", index=False)
    square = [[3.0, 6.0], [4.0, 6.0], [4.0, 7.0], [3.0, 7.0], [3.0, 6.0]]
    (tmp_path / "admin_units.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"shapeName": "A"},
                        "geometry": {"type": "Polygon", "coordinates": [square]},
                    }
                ],
            }
        )
    )
    return tmp_path


def test_point_query(app_dir):
    ex = Explorer.load(app_dir)
    # centre of the top-right cell: x = 542500, y = 714500
    lon, lat = 3.3847, 6.4630
    result = ex.point_query(lat, lon)
    assert ex.cell(lat, lon) == (0, 1)
    assert result["hazard_zone_name"] == "High"
    assert result["hazard_index"] == pytest.approx(0.7)
    assert result["admin_unit"] == "Alpha"


def test_outside_grid(app_dir):
    result = Explorer.load(app_dir).point_query(6.0, 2.0)
    assert result["hazard_zone"] is None


def test_scenarios(app_dir):
    ex = Explorer.load(app_dir)
    assert ex.people_in_high_zones() == 40
    summary = ex.scenario_summary().set_index("scenario")
    assert summary.loc["rain_x1.50", "change"] == 20
    assert ex.zones("rain_x1.50")[0, 0] == 4
    with pytest.raises(KeyError):
        ex.zones("rain_x9.00")


def test_missing_data_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m flood run"):
        Explorer.load(tmp_path / "nothing")
