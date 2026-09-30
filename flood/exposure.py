"""Population exposure: WorldPop counts overlaid on the hazard zones.

WorldPop people-per-cell counts are resampled onto the hazard grid with a sum-preserving
method. Region totals count only cells inside the hazard map (inside the region
boundary); admin unit totals count the cells whose centre falls in each unit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flood import hazard, preprocess
from flood.config import Region
from flood.io import (
    Grid,
    inside_mask,
    read_classes,
    read_raster,
    read_vector,
    resample_sum,
    write_csv,
    write_raster,
)
from flood.stages import Stage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdminUnit:
    admin_id: int
    name: str
    mask: np.ndarray


def by_zone_path(region: Region) -> Path:
    return region.out("exposure", "exposure_by_zone.csv")


def by_lga_path(region: Region) -> Path:
    return region.out("exposure", "exposure_by_lga.csv")


def population_path(region: Region) -> Path:
    return region.out("exposure", "population.tif")


def zone_column(zone: int, name: str) -> str:
    return f"zone{zone}_{name.lower().replace(' ', '_')}"


def admin_units(region: Region, grid: Grid) -> list[AdminUnit]:
    """Admin units with their cell masks on ``grid`` (cell centre inside the polygon)."""
    units = read_vector(preprocess.admin_units_path(region), grid.crs)
    return [
        AdminUnit(int(row.admin_id), str(row.name), inside_mask([row.geometry], grid))
        for row in units.itertuples()
    ]


def zone_table(
    zones: np.ndarray, population: np.ndarray, names: dict[int, str], cell_km2: float
) -> pd.DataFrame:
    """People and area per hazard zone, over all cells that have a zone."""
    mapped = zones > 0
    total = float(population[mapped].sum())
    rows = []
    for z, name in names.items():
        in_zone = zones == z
        people = float(population[in_zone].sum())
        rows.append(
            {
                "zone": z,
                "name": name,
                "population": round(people),
                "pct_total": round(100 * people / total, 2),
                "area_km2": round(int(in_zone.sum()) * cell_km2, 1),
            }
        )
    return pd.DataFrame(rows)


def admin_table(
    zones: np.ndarray,
    population: np.ndarray,
    units: list[AdminUnit],
    names: dict[int, str],
    high_zones: list[int],
) -> pd.DataFrame:
    """People per zone for each admin unit, ranked by the share in the two highest zones."""
    rows = []
    for unit in units:
        total = float(population[unit.mask].sum())
        per_zone = {z: float(population[unit.mask & (zones == z)].sum()) for z in names}
        high = sum(per_zone[z] for z in high_zones)
        row = {"lga_id": unit.admin_id, "lga_name": unit.name, "total_pop": round(total)}
        row.update({zone_column(z, n): round(per_zone[z]) for z, n in names.items()})
        row["pct_high_very_high"] = round(100 * high / total, 1) if total > 0 else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("pct_high_very_high", ascending=False, kind="stable")


def run(region: Region) -> None:
    zones, grid = read_classes(hazard.zones_path(region))
    population = resample_sum(region.data_path("population"), grid)
    write_raster(population_path(region), population, grid, "float64")
    names = hazard.zone_names(region)

    by_zone = zone_table(zones, population, names, grid.cell_area_m2 / 1e6)
    write_csv(by_zone, by_zone_path(region))
    high_zones = region["hazard"]["zones"]["high"]
    by_lga = admin_table(zones, population, admin_units(region, grid), names, high_zones)
    write_csv(by_lga, by_lga_path(region))

    high = int(by_zone.loc[by_zone.zone.isin(high_zones), "population"].sum())
    log.info(
        "%s people mapped, %s in High or Very High", f"{by_zone.population.sum():,}", f"{high:,}"
    )


def load_population(region: Region) -> tuple[np.ndarray, Grid]:
    """The population grid written by this stage (float64 people per cell)."""
    arr, grid = read_raster(population_path(region), "float64")
    return np.nan_to_num(arr), grid


STAGE = Stage(
    name="exposure",
    run=run,
    inputs=lambda r: [
        hazard.zones_path(r),
        preprocess.admin_units_path(r),
        r.data_path("population"),
    ],
    outputs=lambda r: [by_zone_path(r), by_lga_path(r), population_path(r)],
    config=("hazard", "data"),
    description="population by hazard zone and by admin unit",
)
