"""Seed an output folder with the saved golden intermediates so downstream stages can be
replayed and compared with the golden results.

FLOOD_GOLDEN_DIR points at the folder holding the golden rasters (read only; files are
copied, never modified).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

GOLDEN_DIR_ENV = "FLOOD_GOLDEN_DIR"

# new output path -> golden path (relative to FLOOD_GOLDEN_DIR)
INTERMEDIATES = {
    "preprocess/dem.tif": "dem/dem_lagos_utm31n_30m.tif",
    "terrain/twi.tif": "terrain/twi.tif",
    "terrain/slope.tif": "terrain/slope.tif",
    "terrain/flow_accumulation.tif": "terrain/flow_accumulation.tif",
    "terrain/distance_to_rivers.tif": "terrain/distance_to_rivers.tif",
    "runoff/runoff_depth.tif": "runoff/runoff_depth_mm.tif",
    "runoff/curve_number.tif": "runoff/curve_number_lagos.tif",
    "climate/rainfall_p95_monthly.tif": "rainfall/rainfall_95pct.tif",
    "climate/rainfall_total_2021.tif": "rainfall/rainfall_total_2021.tif",
    "landcover/landcover.tif": "lulc/lulc_lagos_10m.tif",
}
GOLDEN_OUTPUTS = {
    "hazard_index": "hazard/flood_hazard_index.tif",
    "hazard_zones": "hazard/flood_hazard_zones.tif",
}


def golden_dir() -> Path | None:
    value = os.environ.get(GOLDEN_DIR_ENV)
    return Path(value) if value and Path(value).exists() else None


def seed(region, golden: Path) -> None:
    """Copy golden intermediates into the region's output folder under their new names."""
    for new, old in INTERMEDIATES.items():
        dst = region.out(new)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(golden / old, dst)
