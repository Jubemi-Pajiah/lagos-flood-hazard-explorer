"""Query API over the precomputed app data written by the ``scenarios`` stage.

Needs only numpy, pandas, pyproj and shapely, so the app can be deployed without the
raster stack. Nothing is recomputed at runtime.

    from flood.api import Explorer
    ex = Explorer.load("app/data/lagos")
    ex.point_query(6.455, 3.40)
    ex.exposure_by_lga("rain_x1.30")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, shape

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "lagos"


@dataclass
class Explorer:
    """Read-only access to one region's precomputed hazard, exposure and scenarios."""

    data_dir: Path
    meta: dict[str, Any]

    @classmethod
    def load(cls, data_dir: str | Path = DEFAULT_DATA_DIR) -> Explorer:
        data_dir = Path(data_dir)
        meta_file = data_dir / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(
                f"No app data at {data_dir}. Run `python -m flood run --region <name>` first."
            )
        return cls(data_dir, json.loads(meta_file.read_text(encoding="utf-8")))

    # ------------------------------------------------------------ scenarios and zones
    @property
    def baseline(self) -> str:
        return self.meta["baseline"]

    @property
    def zone_names(self) -> dict[int, str]:
        return {int(z): v["name"] for z, v in self.meta["zones"].items()}

    @property
    def zone_colors(self) -> dict[int, str]:
        return {int(z): v["color"] for z, v in self.meta["zones"].items()}

    def scenarios(self) -> pd.DataFrame:
        return pd.DataFrame(self.meta["scenarios"])

    def _check(self, scenario: str) -> str:
        known = {s["id"] for s in self.meta["scenarios"]}
        if scenario not in known:
            raise KeyError(f"Unknown scenario '{scenario}'. Known: {', '.join(sorted(known))}")
        return scenario

    def zones(self, scenario: str | None = None) -> np.ndarray:
        """Downsampled hazard zone grid (uint8, 0 = outside the map) for a scenario."""
        scenario = self._check(scenario or self.baseline)
        with np.load(self.data_dir / f"zones_{scenario}.npz") as data:
            return data["zones"]

    @cached_property
    def hazard_index(self) -> np.ndarray:
        """Downsampled baseline hazard index (float32, NaN outside the map)."""
        with np.load(self.data_dir / "hazard_index.npz") as data:
            return data["index"]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Grid extent as (xmin, xmax, ymin, ymax) in the grid CRS."""
        a, _, c, _, e, f = self.meta["transform"]
        rows, cols = self.meta["shape"]
        return c, c + a * cols, f + e * rows, f

    # ------------------------------------------------------------ point query
    @cached_property
    def _to_grid(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", self.meta["crs"], always_xy=True)

    @cached_property
    def _admin_units(self) -> list[tuple[str, Any]]:
        field = self.meta["admin_name_field"]
        data = json.loads((self.data_dir / "admin_units.geojson").read_text(encoding="utf-8"))
        return [(str(f["properties"][field]), shape(f["geometry"])) for f in data["features"]]

    def admin_unit_shapes(self) -> list[tuple[str, Any]]:
        """(name, shapely geometry in WGS84) for each admin unit outline."""
        return list(self._admin_units)

    def cell(self, lat: float, lon: float) -> tuple[int, int] | None:
        """Row and column of the grid cell containing a WGS84 point, or None if outside."""
        x, y = self._to_grid.transform(lon, lat)
        a, _, c, _, e, f = self.meta["transform"]
        row, col = math.floor((y - f) / e), math.floor((x - c) / a)
        rows, cols = self.meta["shape"]
        return (row, col) if 0 <= row < rows and 0 <= col < cols else None

    def admin_unit(self, lat: float, lon: float) -> str | None:
        """Name of the admin unit containing the point, as used in the exposure tables."""
        point = Point(lon, lat)
        for name, geom in self._admin_units:
            if geom.contains(point):
                return self.meta["admin_name_map"].get(name, name)
        return None

    def point_query(self, lat: float, lon: float, scenario: str | None = None) -> dict[str, Any]:
        """Hazard zone, hazard index and admin unit at a WGS84 point.

        Values come from the downsampled grids (cell size in ``meta['cell_size_m']``).
        The index is the baseline index; the zone follows the chosen scenario.
        """
        scenario = self._check(scenario or self.baseline)
        result: dict[str, Any] = {
            "lat": lat,
            "lon": lon,
            "scenario": scenario,
            "hazard_zone": None,
            "hazard_zone_name": None,
            "hazard_index": None,
            "admin_unit": self.admin_unit(lat, lon),
        }
        cell = self.cell(lat, lon)
        if cell is None:
            return result
        zone = int(self.zones(scenario)[cell])
        index = float(self.hazard_index[cell])
        if zone > 0:
            result["hazard_zone"] = zone
            result["hazard_zone_name"] = self.zone_names[zone]
        if math.isfinite(index):
            result["hazard_index"] = round(index, 4)
        return result

    # ------------------------------------------------------------ exposure
    @cached_property
    def _by_zone(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "exposure_by_zone.csv")

    @cached_property
    def _by_lga(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / "exposure_by_lga.csv")

    def exposure_by_zone(self, scenario: str | None = None) -> pd.DataFrame:
        scenario = self._check(scenario or self.baseline)
        table = self._by_zone[self._by_zone.scenario == scenario]
        return table.drop(columns="scenario").reset_index(drop=True)

    def exposure_by_lga(self, scenario: str | None = None) -> pd.DataFrame:
        scenario = self._check(scenario or self.baseline)
        table = self._by_lga[self._by_lga.scenario == scenario]
        return table.drop(columns="scenario").reset_index(drop=True)

    def people_in_high_zones(self, scenario: str | None = None) -> int:
        table = self.exposure_by_zone(scenario)
        return int(table.loc[table.zone.isin(self.meta["high_zones"]), "population"].sum())

    def scenario_summary(self) -> pd.DataFrame:
        """People in high zones per scenario, with the change from the baseline."""
        base = self.people_in_high_zones(self.baseline)
        rows = []
        for sc in self.meta["scenarios"]:
            people = self.people_in_high_zones(sc["id"])
            rows.append(
                {
                    "scenario": sc["id"],
                    "label": sc["label"],
                    "rainfall_scale": sc["rainfall_scale"],
                    "people_high": people,
                    "change": people - base,
                    "change_pct": 100 * (people - base) / base if base else 0.0,
                }
            )
        return pd.DataFrame(rows)
