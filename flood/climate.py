"""Rainfall products from monthly CHIRPS, plus optional rain gauge station statistics.

Gridded outputs (under ``climate/``): seasonal totals per year, and the mean, maximum and
an upper percentile of the monthly values per cell.

When ``climate.station.enabled`` is true, the station workbook is also summarised:
monthly climatology, Mann-Kendall trend with Theil-Sen slope, Gumbel (EV-I) return
levels of annual rainfall, and a monthly comparison with CHIRPS.
"""

from __future__ import annotations

import calendar
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from flood import preprocess
from flood.config import Region
from flood.io import read_raster, write_csv, write_json, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)

MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
EULER_GAMMA = 0.5772


# ---------------------------------------------------------------- output paths
def seasonal_total_path(region: Region, year: int) -> Path:
    return region.out("climate", f"rainfall_total_{year}.tif")


def product_path(region: Region, name: str) -> Path:
    """Path of a named climate product such as ``rainfall_p95_monthly``."""
    return region.out("climate", f"{name}.tif")


def percentile_product(region: Region) -> str:
    return f"rainfall_p{int(region['climate']['extreme_percentile'])}_monthly"


def station_dir(region: Region) -> Path:
    return region.out("climate", "station")


def station_enabled(region: Region) -> bool:
    return bool(region["climate"].get("station", {}).get("enabled", False))


def _outputs(region: Region) -> list[Path]:
    years = region["data"]["rainfall_monthly"]["years"]
    paths = [seasonal_total_path(region, y) for y in years]
    paths += [
        product_path(region, n)
        for n in ("rainfall_mean_monthly", "rainfall_max_monthly", percentile_product(region))
    ]
    if station_enabled(region):
        paths.append(station_dir(region) / "station_summary.json")
    return paths


def _inputs(region: Region) -> list[Path]:
    paths = list(preprocess.monthly_rainfall_paths(region).values())
    if station_enabled(region):
        paths.append(region.data_path("station"))
    return paths


# ---------------------------------------------------------------- gridded rainfall
def rainfall_products(
    stack: np.ndarray, years: list[int], months_by_year: dict[int, list[int]], percentile: float
) -> dict[str, np.ndarray]:
    """Per-cell rainfall statistics from a (month, row, col) stack in chronological order.

    ``months_by_year`` gives, for each year, the indices of its layers in ``stack``.
    Cells with no data in every month stay NaN.
    """
    empty = np.all(np.isnan(stack), axis=0)

    def keep(arr: np.ndarray) -> np.ndarray:
        return np.where(empty, np.nan, arr).astype("float32")

    filled = np.where(empty, 0, stack)
    # np.nanpercentile is very slow on large stacks; use the plain version when it can.
    percentile_fn = np.nanpercentile if np.isnan(filled).any() else np.percentile
    out = {
        "rainfall_mean_monthly": keep(np.nanmean(filled, axis=0)),
        "rainfall_max_monthly": keep(np.nanmax(filled, axis=0)),
        f"rainfall_p{int(percentile)}_monthly": keep(percentile_fn(filled, percentile, axis=0)),
    }
    for year in years:
        layers = stack[months_by_year[year]]
        out[f"rainfall_total_{year}"] = keep(np.nansum(layers, axis=0))
    return out


def gridded_rainfall(region: Region) -> None:
    monthly = preprocess.monthly_rainfall_paths(region)
    keys = sorted(monthly)
    layers = []
    for key in keys:
        arr, _ = read_raster(monthly[key])
        layers.append(arr)
    stack = np.stack(layers)
    years = sorted({y for y, _ in keys})
    months_by_year = {y: [i for i, (yy, _) in enumerate(keys) if yy == y] for y in years}
    percentile = float(region["climate"]["extreme_percentile"])
    for name, arr in rainfall_products(stack, years, months_by_year, percentile).items():
        write_raster(product_path(region, name), arr, region.grid)
    log.info("rainfall products written from %d monthly rasters", len(keys))


# ---------------------------------------------------------------- station statistics
def read_station_workbook(path: Path, sheet: str) -> pd.DataFrame:
    """Read one sheet laid out as rows = months (JAN..DEC), columns = years.

    Returns a DataFrame indexed by month number (1-12) with one column per year.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    header = rows[1]
    years = [int(c) for c in header[1:] if c is not None]
    data: dict[int, list[float]] = {}
    for row in rows[2:]:
        label = str(row[0]).strip().upper()[:3] if row[0] is not None else ""
        if label not in MONTH_ABBR:
            continue
        values = [np.nan if c in (None, "") else float(c) for c in row[1 : 1 + len(years)]]
        data[MONTH_ABBR.index(label) + 1] = values
    frame = pd.DataFrame(data, index=years).T.sort_index()
    frame.columns = years
    return frame


def complete_years(frame: pd.DataFrame) -> list[int]:
    return [y for y in frame.columns if frame[y].notna().sum() == 12]


def rate_to_monthly_total(rate: pd.DataFrame) -> pd.DataFrame:
    """Convert mean daily rain rate (mm/day) per month to monthly totals (mm)."""
    total = rate.copy()
    for year in rate.columns:
        for month in rate.index:
            total.loc[month, year] = rate.loc[month, year] * calendar.monthrange(year, month)[1]
    return total


def mann_kendall(values: np.ndarray) -> dict[str, float]:
    """Mann-Kendall trend test with tie correction. Returns S, Z and two-sided p."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    s = sum(np.sum(np.sign(x[k + 1 :] - x[k])) for k in range(n - 1))
    _, counts = np.unique(x, return_counts=True)
    ties = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
    z = (s - np.sign(s)) / math.sqrt(var_s) if s != 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"S": float(s), "Z": float(z), "p": float(p), "n": n}


def trend(years: list[int], values: np.ndarray) -> dict[str, Any]:
    """Mann-Kendall test and Theil-Sen slope (per year and per decade)."""
    mk = mann_kendall(values)
    slope, intercept, lo, hi = stats.theilslopes(values, np.asarray(years, dtype=float))
    return {
        **mk,
        "slope_per_year": float(slope),
        "slope_per_decade": float(slope * 10),
        "intercept": float(intercept),
        "ci_low_per_decade": float(lo * 10),
        "ci_high_per_decade": float(hi * 10),
    }


def gumbel_fit(annual: np.ndarray) -> tuple[float, float]:
    """Gumbel location and scale by the method of moments."""
    beta = float(np.std(annual, ddof=1) * math.sqrt(6) / math.pi)
    mu = float(np.mean(annual) - EULER_GAMMA * beta)
    return mu, beta


def gumbel_return_level(mu: float, beta: float, period_years: float) -> float:
    """Annual value with the given return period under a Gumbel distribution."""
    return mu - beta * math.log(-math.log(1 - 1.0 / period_years))


def station_statistics(region: Region) -> None:
    """Summarise the station workbook and compare it with CHIRPS."""
    spec = region["climate"]["station"]
    path = region.data_path("station")
    sheets = spec["sheets"]
    monthly_rain = rate_to_monthly_total(read_station_workbook(path, sheets["rain_rate"]))
    temp = read_station_workbook(path, sheets["temperature"])
    humidity = read_station_workbook(path, sheets["relative_humidity"])

    rain_years = complete_years(monthly_rain)
    annual_rain = monthly_rain[rain_years].sum(axis=0)
    temp_years, rh_years = complete_years(temp), complete_years(humidity)
    annual_temp = temp[temp_years].mean(axis=0)
    annual_rh = humidity[rh_years].mean(axis=0)

    out = station_dir(region)
    out.mkdir(parents=True, exist_ok=True)
    climatology = pd.DataFrame(
        {
            "month": MONTH_ABBR,
            "rain_mm": monthly_rain[rain_years].mean(axis=1).to_numpy(),
            "temp_c": temp[temp_years].mean(axis=1).to_numpy(),
            "rh_pct": humidity[rh_years].mean(axis=1).to_numpy(),
        }
    )
    write_csv(climatology, out / "station_climatology.csv")
    annual = (
        pd.DataFrame({"year": rain_years, "rain_mm": annual_rain.to_numpy()})
        .merge(
            pd.DataFrame({"year": temp_years, "temp_c": annual_temp.to_numpy()}),
            on="year",
            how="outer",
        )
        .merge(
            pd.DataFrame({"year": rh_years, "rh_pct": annual_rh.to_numpy()}), on="year", how="outer"
        )
        .sort_values("year")
    )
    write_csv(annual, out / "station_annual.csv")

    mu, beta = gumbel_fit(annual_rain.to_numpy())
    periods = spec["return_periods"]
    summary: dict[str, Any] = {
        "station": spec["name"],
        "rain_years": [rain_years[0], rain_years[-1]],
        "annual_rain_mean_mm": float(annual_rain.mean()),
        "annual_rain_min_mm": float(annual_rain.min()),
        "annual_rain_max_mm": float(annual_rain.max()),
        "wettest_year": int(annual_rain.idxmax()),
        "trend_rain": trend(rain_years, annual_rain.to_numpy()),
        "trend_temp": trend(temp_years, annual_temp.to_numpy()),
        "trend_rh": trend(rh_years, annual_rh.to_numpy()),
        "gumbel": {
            "mu": mu,
            "beta": beta,
            "return_levels": {str(t): gumbel_return_level(mu, beta, t) for t in periods},
        },
    }
    comparison = compare_with_gridded(region, monthly_rain)
    if comparison is not None:
        write_csv(comparison, out / "station_vs_gridded.csv")
        diff = comparison["gridded_mm"] - comparison["station_mm"]
        r, p = stats.pearsonr(comparison["station_mm"], comparison["gridded_mm"])
        summary["gridded_comparison"] = {
            "n": len(comparison),
            "r": float(r),
            "p": float(p),
            "bias_mm": float(diff.mean()),
            "rmse_mm": float(np.sqrt((diff**2).mean())),
        }
    write_json((out / "station_summary.json"), summary)
    log.info("station statistics written (%d complete rain years)", len(rain_years))


def compare_with_gridded(region: Region, monthly_rain: pd.DataFrame) -> pd.DataFrame | None:
    """Station monthly totals against the region-mean CHIRPS value for the same months."""
    rows = []
    for (year, month), path in preprocess.monthly_rainfall_paths(region).items():
        if year not in monthly_rain.columns or not path.exists():
            continue
        arr, _ = read_raster(path)
        if np.isfinite(arr).any():
            rows.append(
                {
                    "year": year,
                    "month": month,
                    "station_mm": float(monthly_rain.loc[month, year]),
                    "gridded_mm": float(np.nanmean(arr)),
                }
            )
    return pd.DataFrame(rows) if len(rows) >= 3 else None


def run(region: Region) -> None:
    gridded_rainfall(region)
    if station_enabled(region):
        station_statistics(region)


STAGE = Stage(
    name="climate",
    run=run,
    inputs=_inputs,
    outputs=_outputs,
    config=("grid", "climate"),
    description="CHIRPS rainfall products and optional station statistics",
)
