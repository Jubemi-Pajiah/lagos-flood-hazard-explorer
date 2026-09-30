"""Hazard factor builders.

Each factor listed under ``hazard.factors`` in the region file names a builder with its
``builder`` key (default ``raster``). A builder takes the region and the factor's settings
and returns an array on the region grid, with NaN where the factor has no value.

To add a new kind of factor, write a function and register it:

    @register("distance_to_coast", inputs=lambda region, spec: [...])
    def distance_to_coast(region, spec):
        ...
        return array_on_region_grid

then list it in the region file (see docs/adding-a-factor.md).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flood.config import ConfigError, Region
from flood.io import align

Builder = Callable[[Region, dict[str, Any]], np.ndarray]
InputsFn = Callable[[Region, dict[str, Any]], list[Path]]


@dataclass(frozen=True)
class RegisteredFactor:
    build: Builder
    inputs: InputsFn


_REGISTRY: dict[str, RegisteredFactor] = {}


def source_path(region: Region, spec: dict[str, Any]) -> Path:
    """The generated file a factor reads, given relative to the region output folder."""
    if "source" not in spec:
        raise ConfigError(f"hazard factor '{spec.get('name')}' needs a 'source' path")
    return region.out(spec["source"])


def register(name: str, inputs: InputsFn | None = None) -> Callable[[Builder], Builder]:
    """Register a factor builder under ``name``.

    ``inputs`` returns the files the builder reads, so the hazard stage knows when to rerun;
    it defaults to the factor's ``source`` file.
    """

    def wrap(fn: Builder) -> Builder:
        _REGISTRY[name] = RegisteredFactor(fn, inputs or (lambda r, s: [source_path(r, s)]))
        return fn

    return wrap


def get(name: str) -> RegisteredFactor:
    if name not in _REGISTRY:
        raise ConfigError(f"Unknown factor builder '{name}'. Registered: {', '.join(_REGISTRY)}")
    return _REGISTRY[name]


def build(region: Region, spec: dict[str, Any]) -> np.ndarray:
    return get(spec.get("builder", "raster")).build(region, spec)


def inputs(region: Region, spec: dict[str, Any]) -> list[Path]:
    return get(spec.get("builder", "raster")).inputs(region, spec)


@register("raster")
def raster(region: Region, spec: dict[str, Any]) -> np.ndarray:
    """A continuous raster warped onto the region grid, with an optional transform.

    ``transform: log1p`` applies log(1 + x) to positive values and drops the rest, which
    suits heavily skewed layers such as flow accumulation.
    """
    arr = align(source_path(region, spec), region.grid, spec.get("resampling", "bilinear"))
    transform = spec.get("transform")
    if transform == "log1p":
        arr = np.where(arr > 0, np.log1p(arr), np.nan)
    elif transform is not None:
        raise ConfigError(f"Unknown transform '{transform}' for factor '{spec['name']}'")
    return arr


@register("class_scores")
def class_scores(region: Region, spec: dict[str, Any]) -> np.ndarray:
    """A class raster mapped to hazard scores in [0, 1] via ``scores: {class: score}``.

    Classes without a score, and no-data cells, become NaN.
    """
    raw = align(source_path(region, spec), region.grid, "nearest")
    out = np.full_like(raw, np.nan)
    for cls, score in spec["scores"].items():
        out[raw == int(cls)] = score
    return out
