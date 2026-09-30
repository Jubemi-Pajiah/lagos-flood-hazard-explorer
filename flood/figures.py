"""Maps and charts of the model outputs, written to ``<output_dir>/<region>/figures/``."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from flood import climate, exposure, hazard, landcover, scenarios  # noqa: E402
from flood.config import Region  # noqa: E402
from flood.io import Grid, read_classes, read_raster, read_vector  # noqa: E402
from flood.stages import Stage  # noqa: E402

log = logging.getLogger(__name__)

DPI = 150
LANDCOVER_COLORS = ["#d7301f", "#41ab5d", "#2b8cbe", "#88419d", "#e6ab02"]
FACTOR_CMAPS = {"cost": "viridis_r", "benefit": "viridis"}


def figure_path(region: Region, name: str) -> Path:
    return region.out("figures", f"{name}.png")


def _figure_names(region: Region) -> list[str]:
    names = [
        "hazard_zones",
        "hazard_index",
        "hazard_factors",
        "landcover",
        "exposure_by_lga",
        "scenario_exposure",
    ]
    if climate.station_enabled(region):
        names += ["station_climatology", "station_annual_rainfall", "station_return_levels"]
    return names


def _extent_km(grid: Grid) -> list[float]:
    xmin, ymin, xmax, ymax = grid.bounds
    return [xmin / 1000, xmax / 1000, ymin / 1000, ymax / 1000]


def _thin(arr: np.ndarray, step: int) -> np.ndarray:
    return arr[::step, ::step]


def _map_axes(title: str, grid: Grid):
    aspect = grid.height / grid.width
    fig, ax = plt.subplots(figsize=(11, max(3.0, 11 * aspect + 1.2)))
    ax.set_title(title)
    ax.set_xlabel("Easting (km, UTM 31N)")
    ax.set_ylabel("Northing (km)")
    return fig, ax


def _scaled_outlines(region: Region, ax, grid: Grid) -> None:
    """Draw admin unit outlines (in km) from the redistributable boundary file of the app."""
    units = read_vector(region.repo_path(region["app"]["admin_units"]), grid.crs)
    units = units.set_geometry(units.geometry.scale(0.001, 0.001, origin=(0, 0)))
    units.boundary.plot(ax=ax, color="#333333", linewidth=0.4)


def hazard_zone_map(region: Region) -> None:
    zones, grid = read_classes(hazard.zones_path(region))
    names = hazard.zone_names(region)
    colors = region["app"]["zone_colors"]
    cmap = ListedColormap(["#ffffff00", *colors])
    fig, ax = _map_axes(f"Flood hazard zones, {region['title']}", grid)
    ax.imshow(
        _thin(zones, 2),
        cmap=cmap,
        norm=BoundaryNorm(np.arange(-0.5, len(names) + 1), cmap.N),
        extent=_extent_km(grid),
        interpolation="nearest",
    )
    _scaled_outlines(region, ax, grid)
    ax.legend(
        handles=[Patch(color=c, label=n) for n, c in zip(names.values(), colors, strict=True)],
        loc="lower left",
        fontsize=8,
        title="Hazard zone",
    )
    fig.tight_layout()
    fig.savefig(figure_path(region, "hazard_zones"), dpi=DPI)
    plt.close(fig)


def hazard_index_map(region: Region) -> None:
    index, grid = read_raster(hazard.index_path(region))
    fig, ax = _map_axes(f"Flood hazard index, {region['title']}", grid)
    im = ax.imshow(_thin(index, 2), cmap="YlOrRd", vmin=0, vmax=1, extent=_extent_km(grid))
    _scaled_outlines(region, ax, grid)
    fig.colorbar(im, ax=ax, shrink=0.6, label="Hazard index (0 to 1)")
    fig.tight_layout()
    fig.savefig(figure_path(region, "hazard_index"), dpi=DPI)
    plt.close(fig)


def factor_panels(region: Region) -> None:
    normalised, _ = hazard.normalised_factors(region)
    specs = {s["name"]: s for s in hazard.factor_specs(region)}
    grid = region.grid
    n = len(normalised)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 1.55), layout="constrained")
    for ax, (name, arr) in zip(axes.ravel(), normalised.items(), strict=False):
        im = ax.imshow(_thin(arr, 3), cmap="YlOrRd", vmin=0, vmax=1, extent=_extent_km(grid))
        label = specs[name].get("label", name.replace("_", " "))
        ax.set_title(f"{label} (weight {specs[name]['weight']:.2f})", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.5, label="Normalised hazard contribution (0 to 1)")
    fig.suptitle("Hazard factors after normalisation")
    fig.savefig(figure_path(region, "hazard_factors"), dpi=DPI)
    plt.close(fig)


def landcover_map(region: Region) -> None:
    lulc, grid = read_classes(landcover.landcover_path(region))
    classes = {int(k): v for k, v in region["landcover"]["classes"].items()}
    cmap = ListedColormap(["#ffffff00", *LANDCOVER_COLORS[: len(classes)]])
    fig, ax = _map_axes(f"Land cover (Random Forest on Sentinel-2), {region['title']}", grid)
    ax.imshow(
        _thin(lulc, 6),
        cmap=cmap,
        extent=_extent_km(grid),
        interpolation="nearest",
        norm=BoundaryNorm(np.arange(-0.5, len(classes) + 1), cmap.N),
    )
    ax.legend(
        handles=[
            Patch(color=c, label=n)
            for n, c in zip(classes.values(), LANDCOVER_COLORS, strict=False)
        ],
        loc="lower left",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(figure_path(region, "landcover"), dpi=DPI)
    plt.close(fig)


def exposure_chart(region: Region) -> None:
    table = pd.read_csv(exposure.by_lga_path(region)).sort_values("pct_high_very_high")
    label = region["data"]["admin_units"].get("label", "Admin unit")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(table) + 1.5))
    ax.barh(table.lga_name, table.pct_high_very_high, color="#d73027")
    ax.set_xlabel("Share of residents in High or Very High zones (%)")
    ax.set_title(f"Population in High or Very High hazard zones by {label}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path(region, "exposure_by_lga"), dpi=DPI)
    plt.close(fig)


def scenario_chart(region: Region) -> None:
    table = pd.read_csv(scenarios.app_dir(region) / "exposure_by_zone.csv")
    meta = json.loads(scenarios.meta_path(region).read_text(encoding="utf-8"))
    scales = {
        s["id"]: s["rainfall_scale"] for s in meta["scenarios"] if s["kind"] == "rainfall_scale"
    }
    high = table[table.zone.isin(meta["high_zones"]) & table.scenario.isin(scales)]
    people = high.groupby("scenario").population.sum()
    x = [scales[s] for s in people.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, people.to_numpy() / 1e6, "o-", color="#d73027")
    ax.set_xlabel("Rainfall multiplier (1.0 = 2020-2021 baseline)")
    ax.set_ylabel("People in High or Very High zones (millions)")
    ax.set_title("Exposure under rainfall scenarios")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path(region, "scenario_exposure"), dpi=DPI)
    plt.close(fig)


def station_charts(region: Region) -> None:
    folder = climate.station_dir(region)
    summary = json.loads((folder / "station_summary.json").read_text(encoding="utf-8"))
    station = summary["station"]
    clim = pd.read_csv(folder / "station_climatology.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(clim.month, clim.rain_mm, color="#2b8cbe")
    ax.set_ylabel("Mean monthly rainfall (mm)")
    ax2 = ax.twinx()
    ax2.plot(clim.month, clim.temp_c, "o-", color="#d73027", label="Temperature (C)")
    ax2.plot(clim.month, clim.rh_pct, "s--", color="#41ab5d", label="Relative humidity (%)")
    ax2.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Mean monthly climate, {station} station")
    fig.tight_layout()
    fig.savefig(figure_path(region, "station_climatology"), dpi=DPI)
    plt.close(fig)

    annual = pd.read_csv(folder / "station_annual.csv").dropna(subset=["rain_mm"])
    t = summary["trend_rain"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(annual.year, annual.rain_mm, "o-", color="#2b8cbe", label="Annual rainfall")
    ax.plot(
        annual.year,
        t["intercept"] + t["slope_per_year"] * annual.year,
        "k--",
        label=f"Theil-Sen {t['slope_per_decade']:+.0f} mm per decade (p = {t['p']:.2f})",
    )
    ax.set_ylabel("Annual rainfall (mm)")
    ax.set_title(f"Annual rainfall, {station} station")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(region, "station_annual_rainfall"), dpi=DPI)
    plt.close(fig)

    g = summary["gumbel"]
    periods = np.array([1.1, 1.5, 2, 3, 5, 10, 25, 50, 100, 200])
    levels = [climate.gumbel_return_level(g["mu"], g["beta"], p) for p in periods]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(periods, levels, "-", color="#2b8cbe", label="Gumbel fit")
    observed = np.sort(annual.rain_mm.to_numpy())[::-1]
    ax.plot(
        (len(observed) + 1) / np.arange(1, len(observed) + 1),
        observed,
        "k+",
        label="Observed (Weibull plotting position)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Annual rainfall (mm)")
    ax.set_title(f"Return levels of annual rainfall, {station} station")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path(region, "station_return_levels"), dpi=DPI)
    plt.close(fig)


def run(region: Region) -> None:
    region.out("figures").mkdir(parents=True, exist_ok=True)
    hazard_zone_map(region)
    hazard_index_map(region)
    factor_panels(region)
    landcover_map(region)
    exposure_chart(region)
    scenario_chart(region)
    if climate.station_enabled(region):
        station_charts(region)
    log.info("figures written to %s", region.out("figures"))


STAGE = Stage(
    name="figures",
    run=run,
    inputs=lambda r: [
        hazard.zones_path(r),
        hazard.index_path(r),
        landcover.landcover_path(r),
        exposure.by_lga_path(r),
        scenarios.meta_path(r),
    ],
    outputs=lambda r: [figure_path(r, n) for n in _figure_names(r)],
    config=("hazard", "app", "landcover"),
    description="maps and charts of the outputs",
)
