"""Clip, reproject and align every raw input to the region grids.

Outputs (under ``<output_dir>/<region>/preprocess/``):

* ``boundary.gpkg`` and ``admin_units.gpkg`` in the region CRS;
* ``rivers.gpkg`` clipped to the analysis grid extent;
* ``dem.tif``, ``clay.tif`` and ``rainfall/<year>-<month>.tif`` on the analysis grid;
* ``sentinel2_<band>.tif`` and ``worldcover.tif`` on the land cover grid.

Cells outside the region boundary are set to no-data in every raster.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds

from flood.config import Region
from flood.io import RESAMPLING, Grid, inside_mask, read_vector, write_raster
from flood.stages import Stage

log = logging.getLogger(__name__)

MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


# ---------------------------------------------------------------- output paths
def boundary_path(region: Region) -> Path:
    return region.out("preprocess", "boundary.gpkg")


def admin_units_path(region: Region) -> Path:
    return region.out("preprocess", "admin_units.gpkg")


def rivers_path(region: Region) -> Path:
    return region.out("preprocess", "rivers.gpkg")


def dem_path(region: Region) -> Path:
    return region.out("preprocess", "dem.tif")


def clay_path(region: Region) -> Path:
    return region.out("preprocess", "clay.tif")


def monthly_rainfall_paths(region: Region) -> dict[tuple[int, int], Path]:
    """Map (year, month) to the aligned monthly rainfall raster."""
    spec = region["data"]["rainfall_monthly"]
    return {
        (year, month): region.out("preprocess", "rainfall", f"{year}-{month:02d}.tif")
        for year in spec["years"]
        for month in spec["months"]
    }


def sentinel2_path(region: Region, band: str) -> Path:
    return region.out("preprocess", f"sentinel2_{band}.tif")


def worldcover_path(region: Region) -> Path:
    return region.out("preprocess", "worldcover.tif")


# ---------------------------------------------------------------- raw inputs
def _raw_monthly_rainfall(region: Region) -> dict[tuple[int, int], Path]:
    spec = region["data"]["rainfall_monthly"]
    out = {}
    for year in spec["years"]:
        for month in spec["months"]:
            rel = spec["pattern"].format(year=year, month=month, month_name=MONTH_NAMES[month - 1])
            out[(year, month)] = region.resolve_data(rel, "data.rainfall_monthly.pattern")
    return out


def _raw_inputs(region: Region) -> list[Path]:
    data = region["data"]
    paths = [
        region.data_path("boundary", "path"),
        region.data_path("admin_units", "path"),
        region.data_path("rivers"),
        region.data_path("soil_clay", "path"),
        *region.data_paths("dem_tiles"),
        *region.data_paths("worldcover_tiles"),
        *_raw_monthly_rainfall(region).values(),
    ]
    for band, tiles in data["sentinel2"].items():
        paths += [region.resolve_data(t, f"data.sentinel2.{band}") for t in tiles]
    return paths


def _outputs(region: Region) -> list[Path]:
    bands = list(region["data"]["sentinel2"])
    return [
        boundary_path(region),
        admin_units_path(region),
        rivers_path(region),
        dem_path(region),
        clay_path(region),
        worldcover_path(region),
        *monthly_rainfall_paths(region).values(),
        *(sentinel2_path(region, b) for b in bands),
    ]


# ---------------------------------------------------------------- helpers
def _mosaic_onto(
    paths: list[Path], grid: Grid, resampling: str, src_nodata: float | None, dtype: str
) -> np.ndarray:
    """Mosaic source tiles in their own CRS, then warp the mosaic onto ``grid``."""
    with rasterio.open(paths[0]) as first:
        src_crs = first.crs
        nodata = first.nodata if src_nodata is None else src_nodata
    bounds = transform_bounds(grid.crs, src_crs, *grid.bounds, densify_pts=21)
    mosaic, transform = merge([str(p) for p in paths], bounds=bounds, nodata=nodata)
    fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    dst = np.full(grid.shape, fill, dtype=dtype)
    reproject(
        mosaic[0],
        dst,
        src_transform=transform,
        src_crs=src_crs,
        src_nodata=nodata,
        dst_transform=grid.transform,
        dst_crs=grid.crs,
        dst_nodata=fill,
        resampling=RESAMPLING[resampling],
    )
    return dst


def _warp_single(path: Path, grid: Grid, resampling: str, src_nodata: float | None) -> np.ndarray:
    return _mosaic_onto([path], grid, resampling, src_nodata, "float32")


def fill_nearest(array: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Fill NaN cells inside the boundary with the value of the nearest valid cell."""
    from scipy.ndimage import distance_transform_edt

    missing = np.isnan(array)
    if not (missing & inside).any() or missing.all():
        return array
    _, (rows, cols) = distance_transform_edt(missing, return_indices=True)
    return np.where(missing & inside, array[rows, cols], array)


def _mask_outside(array: np.ndarray, inside: np.ndarray) -> np.ndarray:
    fill = np.nan if np.issubdtype(array.dtype, np.floating) else 0
    return np.where(inside, array, fill).astype(array.dtype)


# ---------------------------------------------------------------- steps
def prepare_boundaries(region: Region) -> None:
    """Write the dissolved region boundary and the numbered admin units."""
    data = region["data"]
    b = data["boundary"]
    boundary = read_vector(region.data_path("boundary", "path"), region.crs, b.get("where"))
    boundary = boundary.dissolve()[["geometry"]]
    boundary["name"] = region["title"]
    boundary_path(region).parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(boundary_path(region), driver="GPKG")

    a = data["admin_units"]
    units = read_vector(region.data_path("admin_units", "path"), region.crs, a.get("where"))
    names = units[a["name_field"]].replace(a.get("rename") or {})
    units = units.assign(admin_id=np.arange(1, len(units) + 1), name=names)
    units[["admin_id", "name", "geometry"]].to_file(admin_units_path(region), driver="GPKG")
    log.info("boundary and %d admin units written", len(units))


def prepare_rivers(region: Region) -> None:
    """Clip the river network to the analysis grid extent (not the boundary).

    Rivers just outside the boundary still set distances for cells near the edge.
    """
    import geopandas as gpd
    from shapely.geometry import box

    grid = region.grid
    clip_box = box(*grid.bounds)
    src = region.data_path("rivers")
    with rasterio.Env():
        rivers = gpd.read_file(src, bbox=transform_bounds(grid.crs, "EPSG:4326", *grid.bounds))
    rivers = rivers.to_crs(grid.crs)
    rivers = gpd.clip(rivers, clip_box)
    rivers[["geometry"]].to_file(rivers_path(region), driver="GPKG")
    log.info("%d river segments inside the grid", len(rivers))


def prepare_rasters(region: Region) -> None:
    """Warp DEM, clay, monthly rainfall, Sentinel-2 bands and WorldCover onto the grids."""
    data = region["data"]
    grid, lc_grid = region.grid, region.landcover_grid
    boundary = read_vector(boundary_path(region))
    inside = inside_mask(boundary.geometry, grid)
    lc_inside = inside_mask(boundary.geometry, lc_grid)

    dem = _mosaic_onto(region.data_paths("dem_tiles"), grid, "bilinear", None, "float32")
    write_raster(dem_path(region), _mask_outside(dem, inside), grid)
    log.info("dem written")

    soil = data["soil_clay"]
    clay = _warp_single(region.data_path("soil_clay", "path"), grid, "bilinear", soil.get("nodata"))
    write_raster(clay_path(region), _mask_outside(clay, inside), grid)
    log.info("clay written")

    nodata = data["rainfall_monthly"].get("nodata")
    fill = data["rainfall_monthly"].get("fill_gaps", False)
    for key, src in _raw_monthly_rainfall(region).items():
        month = _warp_single(src, grid, "bilinear", nodata)
        if fill:
            month = fill_nearest(month, inside)
        write_raster(monthly_rainfall_paths(region)[key], _mask_outside(month, inside), grid)
    log.info("monthly rainfall written")

    for band, tiles in data["sentinel2"].items():
        paths = [region.resolve_data(t, f"data.sentinel2.{band}") for t in tiles]
        arr = _mosaic_onto(paths, lc_grid, "nearest", 0, "uint16")
        write_raster(sentinel2_path(region, band), _mask_outside(arr, lc_inside), lc_grid, "uint16")
        log.info("sentinel-2 %s written", band)

    wc = _mosaic_onto(region.data_paths("worldcover_tiles"), lc_grid, "nearest", 0, "uint8")
    write_raster(worldcover_path(region), _mask_outside(wc, lc_inside), lc_grid, "uint8")
    log.info("worldcover written")


def run(region: Region) -> None:
    prepare_boundaries(region)
    prepare_rivers(region)
    prepare_rasters(region)


STAGE = Stage(
    name="preprocess",
    run=run,
    inputs=_raw_inputs,
    outputs=_outputs,
    config=("crs", "grid", "landcover_grid", "data"),
    description="clip, reproject and align raw inputs to the region grids",
)
