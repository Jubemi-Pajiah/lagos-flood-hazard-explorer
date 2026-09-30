"""Stage registry and the up-to-date check that lets the pipeline skip finished work.

A stage is up to date when every declared output exists, no output is older than any
declared input, and the config sections the stage depends on are unchanged since it last
ran (tracked in ``<output_dir>/<region>/.stamps/<stage>.json``).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from flood.config import Region
from flood.io import write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    """One pipeline step.

    ``inputs`` and ``outputs`` return the files the stage reads and writes; ``config`` names
    the top-level region-file sections whose values the stage depends on.
    """

    name: str
    run: Callable[[Region], None]
    inputs: Callable[[Region], list[Path]]
    outputs: Callable[[Region], list[Path]]
    config: tuple[str, ...]
    description: str = ""

    def stamp_path(self, region: Region) -> Path:
        return region.out(".stamps", f"{self.name}.json")

    def is_up_to_date(self, region: Region) -> tuple[bool, str]:
        """Return (up_to_date, reason)."""
        outputs = self.outputs(region)
        missing = [p for p in outputs if not p.exists()]
        if missing:
            return False, f"missing output {missing[0]}"
        stamp = self.stamp_path(region)
        if not stamp.exists():
            return False, "no record of a previous run"
        recorded = json.loads(stamp.read_text(encoding="utf-8")).get("config_hash")
        if recorded != region.section_hash(*self.config):
            return False, f"settings changed in {', '.join(self.config)}"
        oldest_output = min(p.stat().st_mtime for p in outputs)
        for path in self.inputs(region):
            if path.exists() and path.stat().st_mtime > oldest_output:
                return False, f"input newer than outputs: {path}"
        return True, "up to date"

    def execute(self, region: Region, force: bool = False) -> bool:
        """Run the stage unless it is up to date. Returns True if it ran."""
        if not force:
            fresh, reason = self.is_up_to_date(region)
            if fresh:
                log.info("[%s] skipped (up to date)", self.name)
                return False
            log.info("[%s] running: %s", self.name, reason)
        else:
            log.info("[%s] running (forced)", self.name)
        started = time.perf_counter()
        self.run(region)
        stamp = self.stamp_path(region)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        write_json(stamp, {"config_hash": region.section_hash(*self.config)})
        log.info("[%s] done in %.1f s", self.name, time.perf_counter() - started)
        return True


def all_stages() -> list[Stage]:
    """Every stage in run order."""
    from flood import (
        climate,
        exposure,
        figures,
        hazard,
        landcover,
        preprocess,
        runoff,
        scenarios,
        terrain,
        validation,
    )

    modules = [
        preprocess,
        terrain,
        climate,
        landcover,
        runoff,
        hazard,
        exposure,
        validation,
        scenarios,
        figures,
    ]
    return [module.STAGE for module in modules]


def get_stage(name: str) -> Stage:
    stages = {stage.name: stage for stage in all_stages()}
    if name not in stages:
        raise KeyError(f"Unknown stage '{name}'. Stages: {', '.join(stages)}")
    return stages[name]


def run_pipeline(region: Region, names: list[str] | None = None, force: bool = False) -> None:
    """Run the named stages (default: all) in pipeline order."""
    stages = all_stages()
    if names:
        unknown = set(names) - {s.name for s in stages}
        if unknown:
            raise KeyError(f"Unknown stage(s): {', '.join(sorted(unknown))}")
        stages = [s for s in stages if s.name in names]
    for stage in stages:
        stage.execute(region, force=force)
