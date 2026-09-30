"""Raster and vector helpers shared by every stage: grids, alignment, reading and writing.

Conventions used across the package:

* every analysis raster sits on the region grid defined in the region file;
* float rasters use NaN in memory and ``FLOAT_NODATA`` on disk;
* class and zone rasters are uint8 with 0 meaning "no data".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

log = logging.getLogger(__name__)

FLOAT_NODATA = -9999.0
CLASS_NODATA = 0
GTIFF_OPTIONS = {"driver": "GTiff", "compress": "deflate", "tiled": True}

RESAMPLING = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "sum": Resampling.sum,
}


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid: CRS, affine transform and shape."""

    crs: str
    transform: Affine
    width: int
    height: int

    @classmethod
    def from_bounds(cls, bounds: Sequence[float], resolution: float, crs: str) -> Grid:
        """Grid covering ``bounds`` (xmin, ymin, xmax, ymax), snapped outwards to ``resolution``."""
        xmin, ymin, xmax, ymax = (float(b) for b in bounds)
        xmin = np.floor(xmin / resolution) * resolution
        ymin = np.floor(ymin / resolution) * resolution
        xmax = np.ceil(xmax / resolution) * resolution
        ymax = np.ceil(ymax / resolution) * resolution
        width = int(round((xmax - xmin) / resolution))
        height = int(round((ymax - ymin) / resolution))
        transform = Affine(resolution, 0.0, xmin, 0.0, -resolution, ymax)
        return cls(crs=crs, transform=transform, width=width, height=height)

    @classmethod
    def of(cls, path: str | Path) -> Grid:
        """Grid of an existing raster file."""
        with rasterio.open(path) as src:
            return cls(
                crs=src.crs.to_string(), transform=src.transform, width=src.width, height=src.height
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def resolution(self) -> float:
        return abs(self.transform.a)

    @property
    def cell_area_m2(self) -> float:
        return abs(self.transform.a * self.transform.e)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xmin, ymax = self.transform.c, self.transform.f
        return (
            xmin,
            ymax + self.transform.e * self.height,
            xmin + self.transform.a * self.width,
            ymax,
        )

    def scaled(self, factor: int) -> Grid:
        """A coarser grid with cells ``factor`` times larger, covering the same area."""
        width = int(np.ceil(self.width / factor))
        height = int(np.ceil(self.height / factor))
        return Grid(self.crs, self.transform * Affine.scale(factor), width, height)

    def profile(self, dtype: str, nodata: float | int | None) -> dict[str, Any]:
        return {
            **GTIFF_OPTIONS,
            "dtype": dtype,
            "nodata": nodata,
            "count": 1,
            "crs": self.crs,
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
        }


def align(
    path: str | Path,
    grid: Grid,
    resampling: str = "bilinear",
    src_nodata: float | None = None,
) -> np.ndarray:
    """Read band 1 of ``path`` warped onto ``grid`` as float32, with no-data as NaN.

    ``src_nodata`` overrides the no-data value stored in the file, for sources such as
    CHIRPS that use -9999 without declaring it.
    """
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        nodata = src.nodata if src_nodata is None else src_nodata
        kwargs = {"src_nodata": src_nodata} if src_nodata is not None else {}
        reproject(
            rasterio.band(src, 1),
            dst,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=RESAMPLING[resampling],
            **kwargs,
        )
    if nodata is not None and not np.isnan(nodata):
        dst[dst == nodata] = np.nan
    return dst


def align_classes(path: str | Path, grid: Grid) -> np.ndarray:
    """Read band 1 of a class raster onto ``grid`` with nearest-neighbour resampling.

    Cells outside the source extent are 0, the class no-data value.
    """
    with rasterio.open(path) as src:
        dst = np.zeros(grid.shape, dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1),
            dst,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.nearest,
        )
    return dst


def read_raster(path: str | Path, dtype: str = "float32") -> tuple[np.ndarray, Grid]:
    """Read band 1 of a raster as floats with no-data as NaN, plus its grid."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(dtype)
        if src.nodata is not None and not np.isnan(src.nodata):
            arr[arr == src.nodata] = np.nan
        grid = Grid(src.crs.to_string(), src.transform, src.width, src.height)
    return arr, grid


def read_classes(path: str | Path) -> tuple[np.ndarray, Grid]:
    """Read a class or zone raster without type conversion."""
    with rasterio.open(path) as src:
        grid = Grid(src.crs.to_string(), src.transform, src.width, src.height)
        return src.read(1), grid


def write_raster(path: str | Path, array: np.ndarray, grid: Grid, dtype: str = "float32") -> Path:
    """Write a single-band GeoTIFF using the package no-data conventions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(np.dtype(dtype), np.floating):
        nodata: float | int = FLOAT_NODATA
        data = np.where(np.isfinite(array), array, FLOAT_NODATA).astype(dtype)
    else:
        nodata = CLASS_NODATA
        data = array.astype(dtype)
    with rasterio.open(path, "w", **grid.profile(dtype, nodata)) as dst:
        dst.write(data, 1)
    log.debug("wrote %s", path)
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    """Write indented JSON as UTF-8 with LF line endings."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=2) + "\n")
    return path


def write_csv(table, path: str | Path) -> Path:
    """Write a DataFrame as UTF-8 CSV with LF line endings and no index."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, lineterminator="\n", encoding="utf-8")
    return path


def read_vector(path: str | Path, crs: str | None = None, where: dict[str, Any] | None = None):
    """Read a vector file, optionally filtering on attribute equality and reprojecting."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    for column, value in (where or {}).items():
        if column not in gdf.columns:
            raise KeyError(f"Column '{column}' not found in {path}; columns: {list(gdf.columns)}")
        gdf = gdf[gdf[column] == value]
    if gdf.empty:
        raise ValueError(f"No features left in {path} after filtering on {where}")
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf.reset_index(drop=True)


def inside_mask(geometries, grid: Grid, all_touched: bool = False) -> np.ndarray:
    """Boolean array that is True for cells whose centre falls inside the geometries."""
    return ~geometry_mask(
        list(geometries), out_shape=grid.shape, transform=grid.transform, all_touched=all_touched
    )


def resample_sum(path: str | Path, grid: Grid) -> np.ndarray:
    """Warp a count raster (such as people per cell) onto ``grid``, preserving totals.

    No-data, non-finite and negative cells are treated as zero.
    """
    dst = np.zeros(grid.shape, dtype="float64")
    with rasterio.open(path) as src:
        data = src.read(1).astype("float64")
        if src.nodata is not None:
            data[data == src.nodata] = 0.0
        data[~np.isfinite(data)] = 0.0
        data[data < 0] = 0.0
        reproject(
            data,
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=Resampling.sum,
        )
    return dst


def warp_array(array: np.ndarray, src: Grid, dst: Grid, resampling: str = "bilinear") -> np.ndarray:
    """Warp an in-memory float array (NaN as no-data) from one grid onto another."""
    if src == dst:
        return array.astype("float32")
    out = np.full(dst.shape, np.nan, dtype="float32")
    reproject(
        array.astype("float32"),
        out,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=np.nan,
        dst_transform=dst.transform,
        dst_crs=dst.crs,
        dst_nodata=np.nan,
        resampling=RESAMPLING[resampling],
    )
    return out


def downsample(
    array: np.ndarray, grid: Grid, factor: int, resampling: str
) -> tuple[np.ndarray, Grid]:
    """Resample an in-memory array to a grid ``factor`` times coarser."""
    coarse = grid.scaled(factor)
    categorical = resampling in {"mode", "nearest"}
    src = array if categorical else np.where(np.isfinite(array), array, np.nan)
    dst = (
        np.zeros(coarse.shape, dtype=array.dtype)
        if categorical
        else np.full(coarse.shape, np.nan, dtype="float32")
    )
    reproject(
        src,
        dst,
        src_transform=grid.transform,
        src_crs=grid.crs,
        dst_transform=coarse.transform,
        dst_crs=coarse.crs,
        src_nodata=CLASS_NODATA if categorical else np.nan,
        dst_nodata=CLASS_NODATA if categorical else np.nan,
        resampling=RESAMPLING[resampling],
    )
    return dst, coarse
