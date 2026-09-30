"""Precomputed rainfall scenarios for the app.

For each scenario the rainfall-driven factors are recomputed and the hazard index, zones
and exposure tables are rebuilt; every other factor stays at its baseline value. Factor
normalisation uses the baseline bounds (``hazard/normalisation.json``), so a wetter
scenario scores higher instead of being rescaled back to the same range.

Scenarios:

* ``scenarios.rainfall_scales``: multiply rainfall by a constant (1.0 is the baseline);
* ``scenarios.return_periods``: scale rainfall to the station's Gumbel design annual
  rainfall for each return period, relative to the station mean. Needs the station
  statistics (``climate.station.enabled``).

Outputs go to ``scenarios.app_data_dir`` and are all the deployed app reads: downsampled
zone grids (uint8, compressed), a downsampled baseline index, exposure tables and a
metadata file.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flood import climate, exposure, factors, hazard, runoff
from flood.climate import gumbel_return_level
from flood.config import Region
from flood.io import Grid, align, downsample, read_raster, warp_array, write_csv, write_json
from flood.stages import Stage

log = logging.getLogger(__name__)


def app_dir(region: Region) -> Path:
    return region.repo_path(region["scenarios"]["app_data_dir"])


def meta_path(region: Region) -> Path:
    return app_dir(region) / "meta.json"


def scenario_list(region: Region) -> list[dict[str, Any]]:
    """Scenario definitions: id, label, kind and rainfall scale."""
    cfg = region["scenarios"]
    out = [
        {
            "id": f"rain_x{s:.2f}",
            "label": "Baseline" if s == 1.0 else f"Rainfall x{s:.2f}",
            "kind": "rainfall_scale",
            "rainfall_scale": float(s),
        }
        for s in cfg["rainfall_scales"]
    ]
    periods = cfg.get("return_periods") or []
    if periods:
        if not climate.station_enabled(region):
            raise ValueError("scenarios.return_periods needs climate.station.enabled: true")
        summary_file = climate.station_dir(region) / "station_summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        mu, beta = summary["gumbel"]["mu"], summary["gumbel"]["beta"]
        mean = summary["annual_rain_mean_mm"]
        for t in periods:
            design = gumbel_return_level(mu, beta, t)
            out.append(
                {
                    "id": f"rp_{t}",
                    "label": f"{t}-year rainfall year",
                    "kind": "return_period",
                    "return_period_years": t,
                    "design_rainfall_mm": design,
                    "rainfall_scale": design / mean,
                }
            )
    return out


class ScenarioModel:
    """Baseline factors held in memory, recombined for any rainfall scale."""

    def __init__(self, region: Region) -> None:
        self.region = region
        cfg = region["scenarios"]
        self.grid = region.grid
        self.weights = hazard.weights(region)
        self.bounds = json.loads(hazard.normalisation_path(region).read_text(encoding="utf-8"))
        self.specs = {s["name"]: s for s in hazard.factor_specs(region)}
        self.scaled = list(cfg["rainfall_factors"])
        self.runoff_factor = cfg["runoff_factor"]

        normalised, _ = hazard.normalised_factors(region)
        self.static = {
            k: v for k, v in normalised.items() if k not in {*self.scaled, self.runoff_factor}
        }
        self.raw_rain = {name: factors.build(region, self.specs[name]) for name in self.scaled}
        self.cn, self.runoff_grid = read_raster(runoff.output_path(region, "curve_number"))
        self.rain_input = align(runoff.rainfall_input_path(region), self.runoff_grid, "bilinear")
        self.ia = float(region["runoff"]["initial_abstraction_ratio"])

    def _normalise(self, name: str, raw: np.ndarray) -> np.ndarray:
        b = self.bounds[name]
        return hazard.apply_bounds(raw, b["lower"], b["upper"], b["direction"])

    def zones(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        """Hazard index and zones with rainfall multiplied by ``scale``."""
        q = runoff.scs_runoff(self.rain_input * np.float32(scale), self.cn, self.ia)
        q = warp_array(q, self.runoff_grid, self.grid, "bilinear")
        layers = dict(self.static)
        layers[self.runoff_factor] = self._normalise(self.runoff_factor, q)
        for name, raw in self.raw_rain.items():
            layers[name] = self._normalise(name, raw * np.float32(scale))
        ordered = {name: layers[name] for name in self.specs}
        return hazard.classify_with(self.region, ordered, self.weights)


def run(region: Region) -> None:
    cfg = region["scenarios"]
    out = app_dir(region)
    out.mkdir(parents=True, exist_ok=True)
    factor = int(cfg["downsample"])
    names = hazard.zone_names(region)
    high = region["hazard"]["zones"]["high"]

    model = ScenarioModel(region)
    population, pop_grid = exposure.load_population(region)
    units = exposure.admin_units(region, pop_grid)
    zone_rows, lga_rows, scenarios = [], [], scenario_list(region)
    coarse_grid: Grid | None = None

    for sc in scenarios:
        index, zones = model.zones(sc["rainfall_scale"])
        small, coarse_grid = downsample(zones, region.grid, factor, "nearest")
        np.savez_compressed(out / f"zones_{sc['id']}.npz", zones=small)
        if sc["id"] == "rain_x1.00":
            small_index, _ = downsample(index, region.grid, factor, "nearest")
            np.savez_compressed(out / "hazard_index.npz", index=small_index.astype("float32"))

        cell_km2 = region.grid.cell_area_m2 / 1e6
        zt = exposure.zone_table(zones, population, names, cell_km2)
        lt = exposure.admin_table(zones, population, units, names, high)
        zone_rows.append(zt.assign(scenario=sc["id"]))
        lga_rows.append(lt.assign(scenario=sc["id"]))
        people = int(zt.loc[zt.zone.isin(high), "population"].sum())
        log.info("%-14s people in high zones: %s", sc["id"], f"{people:,}")

    write_csv(pd.concat(zone_rows), out / "exposure_by_zone.csv")
    write_csv(pd.concat(lga_rows), out / "exposure_by_lga.csv")

    admin_src = region.repo_path(region["app"]["admin_units"])
    shutil.copyfile(admin_src, out / "admin_units.geojson")
    assert coarse_grid is not None
    meta = {
        "region": region["title"],
        "crs": coarse_grid.crs,
        "transform": list(coarse_grid.transform)[:6],
        "shape": list(coarse_grid.shape),
        "downsample_factor": factor,
        "cell_size_m": coarse_grid.resolution,
        "zones": {
            str(z): {"name": n, "color": c}
            for (z, n), c in zip(names.items(), region["app"]["zone_colors"], strict=True)
        },
        "high_zones": high,
        "weights": hazard.weights(region),
        "admin_label": region["data"]["admin_units"].get("label", "Admin unit"),
        "admin_name_field": region["app"]["admin_name_field"],
        "admin_name_map": region["app"].get("admin_name_map", {}),
        "baseline": "rain_x1.00",
        "scenarios": scenarios,
    }

    write_json(meta_path(region), meta)


def _outputs(region: Region) -> list[Path]:
    out = app_dir(region)
    ids = [f"rain_x{s:.2f}" for s in region["scenarios"]["rainfall_scales"]]
    ids += [f"rp_{t}" for t in region["scenarios"].get("return_periods") or []]
    return [
        meta_path(region),
        out / "hazard_index.npz",
        out / "exposure_by_zone.csv",
        out / "exposure_by_lga.csv",
        *(out / f"zones_{i}.npz" for i in ids),
    ]


STAGE = Stage(
    name="scenarios",
    run=run,
    inputs=lambda r: [
        hazard.index_path(r),
        hazard.normalisation_path(r),
        exposure.population_path(r),
        runoff.output_path(r, "curve_number"),
        r.repo_path(r["app"]["admin_units"]),
    ],
    outputs=_outputs,
    config=("hazard", "scenarios", "app"),
    description="precompute rainfall scenarios and export the app data",
)
