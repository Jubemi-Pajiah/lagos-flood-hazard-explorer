# Architecture

```
regions/lagos.yaml ──> flood.config.Region ──> stages (in order) ──> outputs/lagos/...
                                                                  └─> app/data/lagos/ ──> flood.api ──> app/app.py
```

## Pieces

| Module | Role |
|---|---|
| `flood/config.py` | Loads `regions/<name>.yaml`, resolves raw data paths against `FLOOD_DATA_DIR` and output paths under `outputs/<name>/` (or `FLOOD_OUTPUT_DIR`). Missing files raise an error naming the expected path and the yaml key. |
| `flood/io.py` | The grid definition, raster alignment, reading and writing, vector reading, masks, sum-preserving resampling, downsampling. Every stage uses these helpers. |
| `flood/stages.py` | The `Stage` record (inputs, outputs, config sections, run function), the up-to-date check and the pipeline runner. |
| `flood/cli.py` | `python -m flood run / status / stages / fetch-boundaries`. |
| `flood/<stage>.py` | One module per stage, each exposing `STAGE`. |
| `flood/factors.py` | Registry of hazard factor builders (`raster`, `class_scores`, and any you add). |
| `flood/api.py` | Read-only query API over the precomputed app data. Needs only numpy, pandas, pyproj and shapely. |
| `app/app.py` | Streamlit UI over `flood.api`. |

## Stages

| Stage | Reads | Writes (under `outputs/<region>/`) |
|---|---|---|
| `preprocess` | raw boundary, admin units, SRTM tiles, HydroRIVERS, CHIRPS monthly, SoilGrids clay, Sentinel-2 bands, WorldCover tiles | `preprocess/`: boundary and admin units (GeoPackage), rivers clipped to the grid, DEM, clay, monthly rainfall on the 30 m grid; Sentinel-2 bands and WorldCover on the 10 m grid |
| `terrain` | DEM, rivers | `terrain/`: filled DEM, slope, flow accumulation, TWI (GRASS through `qgis_process`), distance to rivers |
| `climate` | monthly rainfall, optional station workbook | `climate/`: seasonal totals, monthly mean, max and 95th percentile; `climate/station/` when enabled |
| `landcover` | Sentinel-2 bands, WorldCover | `landcover/landcover.tif`, `landcover/accuracy.json` |
| `runoff` | clay, land cover, rainfall input | `runoff/`: soil group, curve number, runoff depth, summary |
| `hazard` | every factor source | `hazard/`: index, zones, AHP check, zone areas, normalisation bounds |
| `exposure` | zones, WorldPop, admin units | `exposure/`: population grid, tables by zone and by LGA |
| `validation` | zones, hotspots, curve number, rainfall | `validation/`: hotspot check, runoff sensitivity |
| `scenarios` | hazard outputs, population, curve number | `app/data/<region>/` (committed): downsampled zones per scenario, baseline index, exposure tables, metadata |
| `figures` | outputs above | `figures/*.png` |

A stage is skipped when all its outputs exist, none is older than an input, and the yaml
sections it depends on have not changed since it last ran (recorded in
`outputs/<region>/.stamps/`). `--force` reruns regardless.

## Conventions

* One CRS per region, set once as `crs` in the yaml (EPSG:32631 for Lagos).
* One analysis grid (`grid`, 30 m) for every factor; one finer grid (`landcover_grid`,
  10 m) for the land cover classifier.
* Float rasters: NaN in memory, -9999 on disk. Class and zone rasters: uint8 with 0 as
  no-data. Cells outside the region boundary are no-data.
* Logging through `logging`; no prints in library code.
* Files are UTF-8 with LF line endings.
