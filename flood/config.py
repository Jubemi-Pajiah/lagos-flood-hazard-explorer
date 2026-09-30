"""Region configuration: loads ``regions/<name>.yaml`` and resolves data and output paths."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from flood.io import Grid

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIONS_DIR = REPO_ROOT / "regions"
DATA_DIR_ENV = "FLOOD_DATA_DIR"
OUTPUT_DIR_ENV = "FLOOD_OUTPUT_DIR"


class ConfigError(Exception):
    """Raised when a region file is malformed or a required setting is missing."""


class MissingDataError(FileNotFoundError):
    """Raised when an input file named in the region file does not exist."""


@dataclass
class Region:
    """A loaded region: its settings plus the directories it reads from and writes to."""

    name: str
    settings: dict[str, Any]
    config_path: Path
    data_dir: Path | None
    output_dir: Path
    _grid: Grid | None = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Any:
        try:
            return self.settings[key]
        except KeyError as exc:
            raise ConfigError(f"'{key}' is missing from {self.config_path}") from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    @property
    def crs(self) -> str:
        return self["crs"]

    @property
    def grid(self) -> Grid:
        """The analysis grid shared by every factor raster."""
        if self._grid is None:
            self._grid = self._make_grid("grid")
        return self._grid

    @property
    def landcover_grid(self) -> Grid:
        """The finer grid used for Sentinel-2 features and the land cover map."""
        return self._make_grid("landcover_grid")

    def _make_grid(self, key: str) -> Grid:
        spec = self[key]
        bounds = spec.get("bounds")
        if bounds is None:
            from flood.io import read_vector

            boundary = read_vector(self.out("preprocess", "boundary.gpkg"))
            bounds = boundary.total_bounds
        return Grid.from_bounds(bounds, spec["resolution"], self.crs)

    def data_path(self, *keys: str | int, required: bool = True) -> Path:
        """Resolve a path under ``data:`` in the region file against FLOOD_DATA_DIR.

        ``keys`` walks into nested entries, for example ``data_path("boundary", "path")``.
        """
        node: Any = self["data"]
        for key in keys:
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError) as exc:
                raise ConfigError(
                    f"data.{'.'.join(map(str, keys))} is missing from {self.config_path}"
                ) from exc
        if not isinstance(node, str):
            raise ConfigError(f"data.{'.'.join(map(str, keys))} must be a path string")
        return self.resolve_data(node, "data." + ".".join(map(str, keys)), required=required)

    def resolve_data(self, relative: str, yaml_key: str, required: bool = True) -> Path:
        """Resolve one data path and fail with a clear message when it does not exist."""
        if self.data_dir is None:
            raise ConfigError(
                f"{DATA_DIR_ENV} is not set; it must point at the folder holding the raw data "
                f"(needed for {yaml_key} in {self.config_path.name})"
            )
        path = self.data_dir / relative
        if required and not path.exists():
            raise MissingDataError(
                f"Input file not found for {yaml_key} (set in {self.config_path.name}).\n"
                f"  expected: {path}\n"
                f"  {DATA_DIR_ENV} = {self.data_dir}"
            )
        return path

    def data_paths(self, key: str) -> list[Path]:
        """Resolve a list of paths under ``data.<key>``."""
        entries = self["data"].get(key)
        if not isinstance(entries, list):
            raise ConfigError(f"data.{key} must be a list of paths in {self.config_path}")
        return [self.resolve_data(p, f"data.{key}[{i}]") for i, p in enumerate(entries)]

    def out(self, *parts: str) -> Path:
        """Path of a generated file under ``<output_dir>/<region>/``."""
        return self.output_dir.joinpath(self.name, *parts)

    def repo_path(self, relative: str) -> Path:
        """Resolve a path stored in the region file relative to the repository root."""
        return REPO_ROOT / relative

    def section_hash(self, *sections: str) -> str:
        """Stable hash of the named top-level sections, used to detect config changes."""
        subset = {s: self.settings.get(s) for s in sections}
        blob = json.dumps(subset, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def region_file(name_or_path: str | Path) -> Path:
    """Find the yaml file for a region name, or accept a direct path to one."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"}:
        if not candidate.exists():
            raise ConfigError(f"Region file not found: {candidate}")
        return candidate
    path = REGIONS_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        known = sorted(p.stem for p in REGIONS_DIR.glob("*.yaml"))
        raise ConfigError(f"Unknown region '{name_or_path}'. Known regions: {', '.join(known)}")
    return path


def load_region(
    name_or_path: str | Path,
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Region:
    """Load a region file.

    ``data_dir`` defaults to the FLOOD_DATA_DIR environment variable and ``output_dir`` to
    FLOOD_OUTPUT_DIR, falling back to ``outputs/`` in the repository.
    """
    path = region_file(name_or_path)
    with path.open(encoding="utf-8") as fh:
        settings = yaml.safe_load(fh)
    if not isinstance(settings, dict) or "name" not in settings:
        raise ConfigError(f"{path} must be a mapping with at least a 'name' key")

    data = data_dir or os.environ.get(DATA_DIR_ENV)
    out = output_dir or os.environ.get(OUTPUT_DIR_ENV) or REPO_ROOT / "outputs"
    return Region(
        name=settings["name"],
        settings=settings,
        config_path=path,
        data_dir=Path(data) if data else None,
        output_dir=Path(out),
    )
