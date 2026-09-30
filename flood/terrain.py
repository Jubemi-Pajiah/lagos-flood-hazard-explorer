"""Terrain factors: filled DEM, slope, flow accumulation, TWI and distance to rivers.

Slope, flow accumulation and the topographic wetness index come from GRASS GIS
(``r.fill.dir``, ``r.slope.aspect`` and ``r.watershed``), run through QGIS's
``qgis_process`` command line tool. Set ``FLOOD_QGIS_PROCESS`` to its path if it is not
found automatically. Distance to rivers is computed here with a Euclidean distance
transform on the analysis grid.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from flood import preprocess
from flood.config import Region
from flood.io import inside_mask, read_raster, read_vector, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)

QGIS_PROCESS_ENV = "FLOOD_QGIS_PROCESS"
_QGIS_PROCESS_GLOBS = [
    "C:/Program Files/QGIS */bin/qgis_process-qgis*.bat",
    "C:/OSGeo4W/bin/qgis_process-qgis*.bat",
    "/Applications/QGIS*.app/Contents/MacOS/bin/qgis_process",
]
_GRASS_COMMON = {
    "GRASS_REGION_CELLSIZE_PARAMETER": "0",
    "GRASS_RASTER_FORMAT_OPT": "",
    "GRASS_RASTER_FORMAT_META": "",
}

OUTPUT_NAMES = ["dem_filled", "slope", "flow_accumulation", "twi", "distance_to_rivers"]


def output_path(region: Region, name: str) -> Path:
    return region.out("terrain", f"{name}.tif")


def find_qgis_process() -> str:
    """Locate the ``qgis_process`` executable or raise with instructions."""
    configured = os.environ.get(QGIS_PROCESS_ENV)
    if configured:
        if not Path(configured).exists():
            raise FileNotFoundError(f"{QGIS_PROCESS_ENV} points at a missing file: {configured}")
        return configured
    on_path = shutil.which("qgis_process") or shutil.which("qgis_process-qgis")
    if on_path:
        return on_path
    for pattern in _QGIS_PROCESS_GLOBS:
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "The terrain stage needs GRASS GIS through QGIS's qgis_process tool, which was not "
        f"found. Install QGIS (with GRASS) or set {QGIS_PROCESS_ENV} to the qgis_process path."
    )


def run_grass(algorithm: str, params: dict[str, str]) -> None:
    """Run one GRASS algorithm through qgis_process."""
    exe = find_qgis_process()
    args = [exe, "run", f"grass:{algorithm}", "--"]
    args += [f"{k}={v}" for k, v in {**params, **_GRASS_COMMON}.items()]
    log.info("running grass:%s", algorithm)
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"grass:{algorithm} failed:\n{result.stdout}\n{result.stderr}")


def grass_terrain(dem: Path, out_dir: Path, accumulation_threshold: int) -> dict[str, Path]:
    """Run the GRASS terrain chain on ``dem`` and return the raw output paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: out_dir / f"{name}.tif"
        for name in ["dem_filled", "slope", "flow_accumulation", "twi"]
    }
    tmp = "TEMPORARY_OUTPUT"
    run_grass(
        "r.fill.dir",
        {"input": str(dem), "output": str(paths["dem_filled"]), "direction": tmp, "areas": tmp},
    )
    run_grass(
        "r.slope.aspect",
        {
            "elevation": str(dem),
            "format": "0",
            "precision": "0",
            "zscale": "1.0",
            "min_slope": "0.0",
            "slope": str(paths["slope"]),
            "aspect": tmp,
            "pcurvature": tmp,
            "tcurvature": tmp,
            "dx": tmp,
            "dy": tmp,
            "dxx": tmp,
            "dyy": tmp,
            "dxy": tmp,
        },
    )
    run_grass(
        "r.watershed",
        {
            "elevation": str(paths["dem_filled"]),
            "threshold": str(accumulation_threshold),
            "accumulation": str(paths["flow_accumulation"]),
            "tci": str(paths["twi"]),
            "drainage": tmp,
            "basin": tmp,
            "stream": tmp,
            "half_basin": tmp,
            "length_slope": tmp,
            "slope_steepness": tmp,
            "-s": "false",
            "-4": "false",
            "-a": "true",
            "-b": "false",
            "-m": "false",
        },
    )
    return paths


def distance_to_rivers(rivers, grid) -> np.ndarray:
    """Distance in metres from each cell to the nearest cell touched by a river line."""
    shapes = ((geom, 1) for geom in rivers.geometry if geom is not None and not geom.is_empty)
    river_cells = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    )
    if not river_cells.any():
        raise ValueError("No river cells on the grid; check data.rivers and the grid extent")
    return distance_transform_edt(river_cells == 0).astype("float32") * grid.resolution


def run(region: Region) -> None:
    grid = region.grid
    boundary = read_vector(preprocess.boundary_path(region))
    inside = inside_mask(boundary.geometry, grid)
    threshold = int(region["terrain"]["accumulation_threshold"])

    with tempfile.TemporaryDirectory(prefix="flood_terrain_") as tmp:
        raw = grass_terrain(preprocess.dem_path(region), Path(tmp), threshold)
        for name, path in raw.items():
            arr, _ = read_raster(path)
            write_raster(output_path(region, name), np.where(inside, arr, np.nan), grid)

    rivers = read_vector(preprocess.rivers_path(region))
    dist = distance_to_rivers(rivers, grid)
    write_raster(output_path(region, "distance_to_rivers"), np.where(inside, dist, np.nan), grid)
    log.info("terrain factors written")


STAGE = Stage(
    name="terrain",
    run=run,
    inputs=lambda r: [
        preprocess.dem_path(r),
        preprocess.rivers_path(r),
        preprocess.boundary_path(r),
    ],
    outputs=lambda r: [output_path(r, n) for n in OUTPUT_NAMES],
    config=("grid", "terrain"),
    description="slope, flow accumulation, TWI (GRASS) and distance to rivers",
)
