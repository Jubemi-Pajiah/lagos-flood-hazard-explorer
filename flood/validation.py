"""Checks on the model outputs: documented flood hotspots and runoff sensitivity.

Hotspot check: the hazard zone is sampled at locations repeatedly reported as flooded.
The list is a convenience sample with approximate coordinates and no matched non-flooded
controls, so the hit rate is descriptive, not a skill score.

Runoff sensitivity: runoff depth is recomputed with the curve numbers scaled up and down
(capped to a physical range) and converted to dry (AMC I) and wet (AMC III) antecedent
conditions with the Chow et al. (1988) relations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from flood import hazard, runoff
from flood.config import Region
from flood.io import align, read_classes, read_raster, write_csv, write_json
from flood.stages import Stage

log = logging.getLogger(__name__)


def hotspot_table_path(region: Region) -> Path:
    return region.out("validation", "hotspot_check.csv")


def hotspot_summary_path(region: Region) -> Path:
    return region.out("validation", "hotspot_check.json")


def sensitivity_table_path(region: Region) -> Path:
    return region.out("validation", "runoff_sensitivity.csv")


def sensitivity_summary_path(region: Region) -> Path:
    return region.out("validation", "runoff_sensitivity.json")


def hotspot_file(region: Region) -> Path:
    return region.repo_path(region["validation"]["hotspots"])


# ---------------------------------------------------------------- hotspots
def sample_zone(zones: np.ndarray, row: int, col: int) -> int:
    """Zone at a cell; if it is no-data, the most common zone in the 3x3 neighbourhood."""
    if not (0 <= row < zones.shape[0] and 0 <= col < zones.shape[1]):
        return 0
    zone = int(zones[row, col])
    if zone == 0:
        window = zones[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
        window = window[window > 0]
        zone = int(np.bincount(window).argmax()) if window.size else 0
    return zone


def hotspot_check(region: Region) -> tuple[pd.DataFrame, dict]:
    zones, grid = read_classes(hazard.zones_path(region))
    names = {0: "No data", **hazard.zone_names(region)}
    spots = pd.read_csv(hotspot_file(region))
    to_grid = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    xs, ys = to_grid.transform(spots.lon.to_numpy(), spots.lat.to_numpy())
    inverse = ~grid.transform
    found = []
    for x, y in zip(xs, ys, strict=True):
        col, row = inverse * (x, y)
        found.append(sample_zone(zones, int(np.floor(row)), int(np.floor(col))))
    spots["zone"] = found
    spots["zone_name"] = [names[z] for z in found]

    high = region["hazard"]["zones"]["high"]
    moderate = [min(high) - 1]
    sampled = int((spots.zone > 0).sum())
    n_high = int(spots.zone.isin(high).sum())
    n_mod = int(spots.zone.isin(moderate).sum())
    summary = {
        "hotspots": sampled,
        "in_high_or_very_high": n_high,
        "in_moderate": n_mod,
        "in_moderate_or_above": n_high + n_mod,
        "share_high_or_very_high": n_high / sampled,
        "share_moderate_or_above": (n_high + n_mod) / sampled,
    }
    return spots, summary


# ---------------------------------------------------------------- runoff sensitivity
def amc_dry(cn: np.ndarray) -> np.ndarray:
    """Convert AMC II curve numbers to AMC I (dry)."""
    return 4.2 * cn / (10.0 - 0.058 * cn)


def amc_wet(cn: np.ndarray) -> np.ndarray:
    """Convert AMC II curve numbers to AMC III (wet)."""
    return 23.0 * cn / (10.0 + 0.13 * cn)


def runoff_sensitivity(region: Region) -> tuple[pd.DataFrame, dict]:
    cfg = region["runoff"]
    sens = cfg["sensitivity"]
    cn0, grid = read_raster(runoff.output_path(region, "curve_number"))
    rain = align(runoff.rainfall_input_path(region), grid, "bilinear")
    rain = np.where(np.isfinite(rain), np.maximum(rain, 0.0), 0.0).astype("float32")
    valid = np.isfinite(cn0) & (cn0 > 0)
    ia = float(cfg["initial_abstraction_ratio"])
    lo, hi = float(sens["cn_min"]), float(sens["cn_max"])

    def scenario(cn: np.ndarray) -> dict[str, float]:
        cn = np.where(valid, np.clip(cn, lo, hi), np.nan).astype("float32")
        q = runoff.scs_runoff(rain, cn, ia)[valid]
        wet = q[q > 0]
        return {
            "mean_cn": float(cn[valid].mean()),
            "mean_runoff_mm": float(q.mean()),
            "mean_runoff_where_positive_mm": float(wet.mean()),
            "max_runoff_mm": float(wet.max()),
        }

    rows = []
    for pct in sens["cn_changes_pct"]:
        label = "baseline (AMC II)" if pct == 0 else f"CN {pct:+d}%"
        rows.append({"scenario": label, **scenario(cn0 * np.float32(1 + pct / 100.0))})
    rows.append({"scenario": "AMC I (dry)", **scenario(amc_dry(cn0))})
    rows.append({"scenario": "AMC III (wet)", **scenario(amc_wet(cn0))})
    table = pd.DataFrame(rows)
    base = table.loc[table.scenario == "baseline (AMC II)"].iloc[0]
    table["change_pct"] = 100 * (
        table.mean_runoff_where_positive_mm / base.mean_runoff_where_positive_mm - 1
    )

    def span(labels: list[str]) -> list[float]:
        sel = table[table.scenario.isin(labels)].mean_runoff_mm
        return [float(sel.min()), float(sel.max())]

    changes = [p for p in sens["cn_changes_pct"] if p != 0]
    summary = {
        "rainfall_mean_mm": float(rain[valid].mean()),
        "baseline_mean_runoff_mm": float(base.mean_runoff_mm),
        "range_by_change_pct": {
            str(abs(p)): span([f"CN {-abs(p):+d}%", f"CN {abs(p):+d}%"])
            for p in sorted({abs(p) for p in changes})
        },
        "range_amc": span(["AMC I (dry)", "AMC III (wet)"]),
    }
    return table, summary


def run(region: Region) -> None:
    spots, spot_summary = hotspot_check(region)
    hotspot_table_path(region).parent.mkdir(parents=True, exist_ok=True)
    write_csv(spots, hotspot_table_path(region))
    write_json(hotspot_summary_path(region), spot_summary)
    log.info(
        "hotspots in High or Very High: %d of %d",
        spot_summary["in_high_or_very_high"],
        spot_summary["hotspots"],
    )

    table, summary = runoff_sensitivity(region)
    write_csv(table, sensitivity_table_path(region))
    write_json(sensitivity_summary_path(region), summary)


STAGE = Stage(
    name="validation",
    run=run,
    inputs=lambda r: [
        hazard.zones_path(r),
        hotspot_file(r),
        runoff.output_path(r, "curve_number"),
        runoff.rainfall_input_path(r),
    ],
    outputs=lambda r: [
        hotspot_table_path(r),
        hotspot_summary_path(r),
        sensitivity_table_path(r),
        sensitivity_summary_path(r),
    ],
    config=("validation", "runoff", "hazard"),
    description="documented hotspot check and runoff sensitivity",
)
