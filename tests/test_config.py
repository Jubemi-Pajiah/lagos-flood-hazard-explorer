import pytest

from flood.config import ConfigError, MissingDataError, load_region


def test_lagos_grid_matches_saved_outputs(region):
    grid = region.grid
    assert grid.shape == (1225, 6053)
    assert grid.resolution == 30
    assert grid.crs == "EPSG:32631"
    assert region.landcover_grid.shape == (3672, 18159)


def test_missing_data_names_path_and_key(region):
    with pytest.raises(MissingDataError) as err:
        region.data_path("population")
    message = str(err.value)
    assert "data.population" in message
    assert "NGA_population_v1_2_gridded.tif" in message


def test_unset_data_dir_is_explained(monkeypatch, tmp_path):
    monkeypatch.delenv("FLOOD_DATA_DIR", raising=False)
    region = load_region("lagos", output_dir=tmp_path)
    with pytest.raises(ConfigError, match="FLOOD_DATA_DIR"):
        region.data_path("population")


def test_unknown_region():
    with pytest.raises(ConfigError, match="Known regions"):
        load_region("atlantis")


def test_section_hash_changes_with_settings(region):
    before = region.section_hash("hazard")
    region.settings["hazard"]["factors"][0]["weight"] = 0.21
    assert region.section_hash("hazard") != before
