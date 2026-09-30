"""Surface runoff depth with the USDA SCS curve number method.

Hydrologic soil group (A-D) comes from topsoil clay content, the curve number from a
land cover by soil group table, and runoff depth from

    S  = 25400 / CN - 254          (potential maximum retention, mm)
    Ia = lambda * S                (initial abstraction, lambda = 0.2 by default)
    Q  = (P - Ia)^2 / (P - Ia + S) for P > Ia, otherwise 0

The rainfall input P is the climate product named in ``runoff.rainfall``. With a seasonal
total this gives a seasonal runoff depth, which is used as a relative index in the hazard
model, not as a design value.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from flood import climate, landcover, preprocess
from flood.config import Region
from flood.io import Grid, align, read_raster, write_json, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)


def output_path(region: Region, name: str) -> Path:
    return region.out("runoff", f"{name}.tif")


def summary_path(region: Region) -> Path:
    return region.out("runoff", "runoff_summary.json")


def rainfall_input_path(region: Region) -> Path:
    return climate.product_path(region, region["runoff"]["rainfall"])


def hydrologic_soil_group(
    clay_pct: np.ndarray, breaks: list[float], missing_group: int = 0
) -> np.ndarray:
    """Soil group 1-4 (A-D) from clay percent; ``breaks`` are the A/B, B/C and C/D limits.

    Cells with no clay value get ``missing_group`` (0 leaves them without a group).
    """
    hsg = np.full(clay_pct.shape, missing_group, dtype="uint8")
    finite = np.isfinite(clay_pct)
    groups = np.digitize(clay_pct[finite], breaks) + 1
    hsg[finite] = groups.astype("uint8")
    return hsg


def curve_numbers(
    lulc: np.ndarray, hsg: np.ndarray, table: dict[int, list[float]], default: float
) -> np.ndarray:
    """Curve number per cell from land cover class and soil group; NaN where undefined.

    Cells with a land cover class and a soil group that the table does not cover get
    ``default``.
    """
    cn = np.zeros(lulc.shape, dtype="float32")
    for cls, by_group in table.items():
        for group, value in enumerate(by_group, start=1):
            cn[(lulc == cls) & (hsg == group)] = value
    cn[(cn == 0) & (lulc > 0) & (hsg > 0)] = default
    cn[cn == 0] = np.nan
    return cn


def scs_runoff(rain: np.ndarray, cn: np.ndarray, ia_ratio: float = 0.2) -> np.ndarray:
    """SCS-CN runoff depth (mm). NaN where the curve number is undefined."""
    has_cn = np.isfinite(cn) & (cn > 0)
    safe_cn = np.where(has_cn, cn, 1.0)
    s = 25400.0 / safe_cn - 254.0
    ia = ia_ratio * s
    p = np.where(np.isfinite(rain), np.maximum(rain, 0.0), 0.0)
    wet = p > ia
    q = np.where(wet, (p - ia) ** 2 / np.where(wet, p - ia + s, 1.0), 0.0)
    return np.where(has_cn, q, np.nan).astype("float32")


def run(region: Region) -> None:
    cfg = region["runoff"]
    clay, grid = read_raster(preprocess.clay_path(region))
    clay_pct = clay * float(cfg["clay_to_percent"])
    hsg = hydrologic_soil_group(
        clay_pct, cfg["hsg_clay_breaks"], int(cfg.get("hsg_where_missing", 0))
    )

    lulc = align(landcover.landcover_path(region), grid, "nearest")
    lulc = np.nan_to_num(lulc, nan=0).astype("uint8")
    table = {int(k): v for k, v in cfg["curve_numbers"].items()}
    cn = curve_numbers(lulc, hsg, table, float(cfg["default_curve_number"]))

    rain = align(rainfall_input_path(region), grid, "bilinear")
    q = scs_runoff(rain, cn, float(cfg["initial_abstraction_ratio"]))

    write_raster(output_path(region, "hydrologic_soil_group"), hsg, grid, "uint8")
    write_raster(output_path(region, "curve_number"), cn, grid)
    write_raster(output_path(region, "runoff_depth"), q, grid)
    write_summary(region, rain, cn, q, grid)


def write_summary(
    region: Region, rain: np.ndarray, cn: np.ndarray, q: np.ndarray, grid: Grid
) -> None:
    valid = np.isfinite(cn)
    wet = valid & (q > 0)
    p = np.where(np.isfinite(rain), rain, 0.0)
    summary = {
        "rainfall_input": region["runoff"]["rainfall"],
        "cells_with_curve_number": int(valid.sum()),
        "curve_number": {
            "mean": float(np.nanmean(cn)),
            "min": float(np.nanmin(cn)),
            "max": float(np.nanmax(cn)),
        },
        "rainfall_mm_mean": float(p[valid].mean()),
        "runoff_mm_mean": float(q[valid].mean()),
        "runoff_mm_mean_where_positive": float(q[wet].mean()) if wet.any() else 0.0,
        "runoff_coefficient_where_positive": float(q[wet].mean() / p[wet].mean())
        if wet.any()
        else 0.0,
        "grid_resolution_m": grid.resolution,
    }
    write_json(summary_path(region), summary)
    log.info(
        "mean runoff %.0f mm, mean CN %.1f",
        summary["runoff_mm_mean"],
        summary["curve_number"]["mean"],
    )


STAGE = Stage(
    name="runoff",
    run=run,
    inputs=lambda r: [preprocess.clay_path(r), landcover.landcover_path(r), rainfall_input_path(r)],
    outputs=lambda r: (
        [output_path(r, n) for n in ("hydrologic_soil_group", "curve_number", "runoff_depth")]
        + [summary_path(r)]
    ),
    config=("runoff",),
    description="hydrologic soil group, curve number and SCS-CN runoff depth",
)
