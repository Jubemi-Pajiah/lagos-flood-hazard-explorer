"""Flood hazard index: normalised factors combined with AHP weights, then zoned.

Each continuous factor is rescaled to [0, 1] between its lower and upper percentile
(2nd and 98th by default) and clipped; "cost" factors are reversed so that 1 always means
more hazard. The hazard index is the weighted sum of the factors, defined where every
factor has a value, and zones come from fixed breaks on the index.

The weights in the region file are the model weights. The AHP pairwise matrix is kept
alongside them and checked on every run: its principal eigenvector, lambda_max and
consistency ratio are reported together with how far the eigenvector is from the weights
and how many cells would change zone if the eigenvector were used instead.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flood import factors, landcover
from flood.config import ConfigError, Region
from flood.io import align, write_csv, write_json, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)

# Saaty's random consistency index for matrices of order n.
RANDOM_INDEX = {
    1: 0.0,
    2: 0.0,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
    11: 1.51,
    12: 1.48,
    13: 1.56,
    14: 1.57,
    15: 1.59,
}
CONSISTENCY_LIMIT = 0.10


# ---------------------------------------------------------------- paths
def index_path(region: Region) -> Path:
    return region.out("hazard", "hazard_index.tif")


def zones_path(region: Region) -> Path:
    return region.out("hazard", "hazard_zones.tif")


def ahp_path(region: Region) -> Path:
    return region.out("hazard", "ahp.json")


def zone_areas_path(region: Region) -> Path:
    return region.out("hazard", "hazard_zone_areas.csv")


def normalisation_path(region: Region) -> Path:
    return region.out("hazard", "normalisation.json")


# ---------------------------------------------------------------- settings
def factor_specs(region: Region) -> list[dict[str, Any]]:
    specs = region["hazard"]["factors"]
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ConfigError("hazard.factors has duplicate names")
    for spec in specs:
        if spec.get("direction") not in {"benefit", "cost"}:
            raise ConfigError(f"factor '{spec['name']}' needs direction: benefit or cost")
        if "weight" not in spec:
            raise ConfigError(f"factor '{spec['name']}' needs a weight")
    return specs


def weights(region: Region) -> dict[str, float]:
    """Model weights by factor, checked to sum to 1."""
    w = {s["name"]: float(s["weight"]) for s in factor_specs(region)}
    if abs(sum(w.values()) - 1.0) > 1e-9:
        raise ConfigError(f"hazard factor weights sum to {sum(w.values()):.6f}, not 1")
    return w


def zone_names(region: Region) -> dict[int, str]:
    return {i + 1: name for i, name in enumerate(region["hazard"]["zones"]["names"])}


# ---------------------------------------------------------------- AHP
def parse_judgement(value: Any) -> float:
    """A Saaty judgement written as a number or a fraction such as '1/3'."""
    return float(Fraction(str(value)))


def pairwise_matrix(rows: list[list[Any]]) -> np.ndarray:
    """Build the full reciprocal matrix from its upper triangle.

    Row i lists the judgements a_ij for j >= i (the first entry is the diagonal, 1).
    """
    n = len(rows)
    a = np.ones((n, n))
    for i, row in enumerate(rows):
        if len(row) != n - i:
            raise ConfigError(f"pairwise_matrix row {i + 1} needs {n - i} entries, has {len(row)}")
        for offset, value in enumerate(row):
            j = i + offset
            a[i, j] = parse_judgement(value)
            a[j, i] = 1.0 / a[i, j]
    if not np.allclose(np.diag(a), 1.0):
        raise ConfigError("pairwise_matrix diagonal entries must be 1")
    return a


def ahp_weights(matrix: np.ndarray) -> dict[str, Any]:
    """Principal eigenvector weights, lambda_max, CI, RI and CR of a pairwise matrix."""
    n = matrix.shape[0]
    values, vectors = np.linalg.eig(matrix)
    k = int(np.argmax(values.real))
    w = np.abs(vectors[:, k].real)
    w = w / w.sum()
    lam = float(values[k].real)
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = RANDOM_INDEX.get(n)
    if ri is None:
        raise ConfigError(f"No random index for a {n}x{n} matrix")
    cr = ci / ri if ri > 0 else 0.0
    return {"weights": w, "lambda_max": lam, "ci": ci, "ri": ri, "cr": cr}


# ---------------------------------------------------------------- index
def normalise(
    arr: np.ndarray, lower_pct: float, upper_pct: float, direction: str
) -> tuple[np.ndarray, tuple[float, float]]:
    """Percentile-clipped rescaling to [0, 1]. Returns the result and the (lo, hi) bounds."""
    lo, hi = np.nanpercentile(arr, lower_pct), np.nanpercentile(arr, upper_pct)
    return apply_bounds(arr, float(lo), float(hi), direction), (float(lo), float(hi))


def apply_bounds(arr: np.ndarray, lo: float, hi: float, direction: str) -> np.ndarray:
    """Rescale with fixed bounds; ``cost`` factors are reversed."""
    if hi == lo:
        return np.zeros_like(arr)
    lo_t, hi_t = arr.dtype.type(lo), arr.dtype.type(hi)
    out = np.clip((arr - lo_t) / (hi_t - lo_t), 0, 1)
    return 1.0 - out if direction == "cost" else out


def hazard_index(
    normalised: dict[str, np.ndarray], factor_weights: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sum of normalised factors, NaN where any factor is missing.

    Returns (index as float32, valid mask).
    """
    shape = next(iter(normalised.values())).shape
    total = np.zeros(shape, dtype="float64")
    valid = np.ones(shape, dtype=bool)
    for name, norm in normalised.items():
        finite = np.isfinite(norm)
        valid &= finite
        total += np.where(finite, factor_weights[name] * norm, 0.0)
    return np.where(valid, total, np.nan).astype("float32"), valid


def classify(index: np.ndarray, breaks: list[float]) -> np.ndarray:
    """Zones 1..n from ascending breaks (lower bound inclusive); 0 where the index is NaN."""
    zones = np.zeros(index.shape, dtype="uint8")
    valid = np.isfinite(index)
    for z in range(1, len(breaks)):
        zones[(index >= breaks[z - 1]) & (index < breaks[z]) & valid] = z
    return zones


def normalised_factors(region: Region) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build and normalise every factor. Returns arrays and the bounds used per factor."""
    cfg = region["hazard"]["normalisation"]
    arrays, bounds = {}, {}
    for spec in factor_specs(region):
        raw = factors.build(region, spec)
        if spec.get("normalise", True):
            norm, (lo, hi) = normalise(
                raw, cfg["lower_percentile"], cfg["upper_percentile"], spec["direction"]
            )
            bounds[spec["name"]] = {"lower": lo, "upper": hi, "direction": spec["direction"]}
        else:
            norm = raw if spec["direction"] == "benefit" else 1.0 - raw
        arrays[spec["name"]] = norm
        log.info("factor %-20s mean %.4f", spec["name"], float(np.nanmean(norm)))
    return arrays, bounds


# ---------------------------------------------------------------- reports
def ahp_report(region: Region, normalised: dict[str, np.ndarray], zones: np.ndarray) -> dict:
    specs = factor_specs(region)
    names = [s["name"] for s in specs]
    matrix = pairwise_matrix(region["hazard"]["pairwise_matrix"])
    if matrix.shape[0] != len(names):
        raise ConfigError(
            f"pairwise_matrix is {matrix.shape[0]}x{matrix.shape[0]} but there "
            f"are {len(names)} factors"
        )
    ahp = ahp_weights(matrix)
    used = weights(region)
    eigen = dict(zip(names, (float(v) for v in ahp["weights"]), strict=True))
    _, eigen_zones = classify_with(region, normalised, eigen)
    changed = int(((eigen_zones != zones) & (zones > 0)).sum())
    report = {
        "factors": names,
        "weights": used,
        "matrix": matrix.round(6).tolist(),
        "eigenvector": eigen,
        "lambda_max": ahp["lambda_max"],
        "ci": ahp["ci"],
        "ri": ahp["ri"],
        "cr": ahp["cr"],
        "consistent": ahp["cr"] < CONSISTENCY_LIMIT,
        "max_abs_difference": max(abs(eigen[n] - used[n]) for n in names),
        "cells_changing_zone_with_eigenvector": changed,
        "share_changing_zone_with_eigenvector": changed / int((zones > 0).sum()),
    }
    if not report["consistent"]:
        log.warning("AHP consistency ratio %.3f is above %.2f", ahp["cr"], CONSISTENCY_LIMIT)
    return report


def classify_with(
    region: Region, normalised: dict[str, np.ndarray], factor_weights: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    index, _ = hazard_index(normalised, factor_weights)
    return index, classify(index, region["hazard"]["zones"]["breaks"])


def zone_area_table(region: Region, zones: np.ndarray) -> pd.DataFrame:
    """Area per zone, split into water and land using the land cover water class."""
    grid = region.grid
    lulc = align(landcover.landcover_path(region), grid, "nearest")
    water = lulc == int(region["landcover"]["water_class"])
    km2 = grid.cell_area_m2 / 1e6
    mapped = int((zones > 0).sum())
    land_total = int(((zones > 0) & ~water).sum())
    rows = []
    for z, name in zone_names(region).items():
        in_zone = zones == z
        cells = int(in_zone.sum())
        water_cells = int((in_zone & water).sum())
        rows.append(
            {
                "zone": z,
                "name": name,
                "area_km2": round(cells * km2, 1),
                "pct_of_mapped": round(100 * cells / mapped, 2),
                "water_km2": round(water_cells * km2, 1),
                "land_km2": round((cells - water_cells) * km2, 1),
                "pct_of_land": round(100 * (cells - water_cells) / land_total, 2),
            }
        )
    return pd.DataFrame(rows)


def run(region: Region) -> None:
    normalised, bounds = normalised_factors(region)
    index, zones = classify_with(region, normalised, weights(region))
    write_raster(index_path(region), index, region.grid)
    write_raster(zones_path(region), zones, region.grid, "uint8")

    report = ahp_report(region, normalised, zones)
    write_json(ahp_path(region), report)
    write_json(normalisation_path(region), bounds)
    write_csv(zone_area_table(region, zones), zone_areas_path(region))
    log.info("hazard index written; AHP CR %.3f", report["cr"])


def _inputs(region: Region) -> list[Path]:
    paths = [landcover.landcover_path(region)]
    for spec in factor_specs(region):
        paths += factors.inputs(region, spec)
    return paths


STAGE = Stage(
    name="hazard",
    run=run,
    inputs=_inputs,
    outputs=lambda r: [
        index_path(r),
        zones_path(r),
        ahp_path(r),
        zone_areas_path(r),
        normalisation_path(r),
    ],
    config=("grid", "hazard", "landcover"),
    description="normalise factors, check AHP consistency, build hazard index and zones",
)
