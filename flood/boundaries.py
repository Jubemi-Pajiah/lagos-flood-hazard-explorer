"""Fetch redistributable admin unit polygons for the app from geoBoundaries (CC BY 4.0).

The model itself reads admin units from ``data.admin_units`` under FLOOD_DATA_DIR. Those
files may not allow redistribution, so the app ships a separate copy of the unit outlines
from geoBoundaries, selected by overlap with the region boundary.

    python -m flood fetch-boundaries --region lagos
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

from flood import preprocess
from flood.config import Region
from flood.io import read_vector

log = logging.getLogger(__name__)

API = "https://www.geoboundaries.org/api/current/gbOpen/{iso}/{level}/"


def fetch(region: Region, min_overlap: float = 0.5) -> Path:
    """Download the geoBoundaries layer named in ``app.admin_units_source`` and keep the
    units that lie mostly (``min_overlap`` of their area) inside the region boundary."""
    import geopandas as gpd

    src = region["app"]["admin_units_source"]
    with urllib.request.urlopen(API.format(iso=src["iso"], level=src["level"])) as resp:
        meta = json.load(resp)
    units = gpd.read_file(meta["simplifiedGeometryGeoJSON"])
    boundary = read_vector(preprocess.boundary_path(region)).to_crs(region.crs)
    projected = units.to_crs(region.crs)
    inside = projected.intersection(boundary.union_all()).area / projected.area
    selected = units[inside.to_numpy() >= min_overlap][["shapeName", "shapeID", "geometry"]]

    out = region.repo_path(region["app"]["admin_units"])
    out.parent.mkdir(parents=True, exist_ok=True)
    selected.reset_index(drop=True).to_file(out, driver="GeoJSON")
    log.info(
        "%d units from geoBoundaries %s %s (%s, %s) written to %s",
        len(selected),
        src["iso"],
        src["level"],
        meta.get("boundarySource"),
        meta.get("boundaryLicense"),
        out,
    )
    return out
