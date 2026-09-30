# Adding a region

A region is one yaml file plus its raw data. No code changes are needed. This walks
through adding a Nigerian state; for other countries only the data sources change.

## 1. Download the data

Put everything under one folder and point `FLOOD_DATA_DIR` at it. For a Nigerian state:

| Input | yaml key | Source | Notes |
|---|---|---|---|
| State boundary and LGAs | `data.boundary`, `data.admin_units` | any polygon layer with a state name and an LGA name column (GADM Nigeria levels 1 and 2 were used for Lagos) | check the licence; GADM does not allow redistribution, so keep it out of the repo |
| DEM | `data.dem_tiles` | SRTM 1 arc-second tiles covering the state (USGS EarthExplorer) | list every tile |
| Rivers | `data.rivers` | HydroRIVERS v1.0, Africa | the whole continental file is fine; it is clipped to the grid |
| Monthly rainfall | `data.rainfall_monthly` | CHIRPS v2.0 monthly, Africa, GeoTIFF | named by `pattern`, for example `{month_name}-{year}.tif` |
| Topsoil clay | `data.soil_clay` | SoilGrids 250 m, `clay_0-5cm_mean` (g/kg) | cropped to the state is fine |
| Sentinel-2 | `data.sentinel2` | L2A 10 m bands B02, B03, B04, B08 for every tile over the state (Copernicus Data Space) | one list of tiles per band, all tiles in the same UTM zone |
| Land cover labels | `data.worldcover_tiles` | ESA WorldCover 2021 v200 3x3 degree tiles | |
| Population | `data.population` | WorldPop Nigeria gridded population (people per cell) | |

## 2. Write the yaml

Copy `regions/lagos.yaml` to `regions/<name>.yaml` and change:

1. `name` and `title`.
2. `crs`: the UTM zone of the state (EPSG:32631 for 0-6 E, EPSG:32632 for 6-12 E,
   EPSG:32633 for 12-18 E). Sentinel-2 tiles must be in the same zone.
3. `grid.bounds` and `landcover_grid.bounds`: delete both `bounds` lines to fit the grids
   to the boundary automatically.
4. Every path under `data:`, and the `where` filters (for example `{NAME_1: Kano}`).
5. `data.admin_units.rename`: delete it, or map source names to the names you want in the
   outputs.
6. `data.rainfall_monthly.years` and `months`.
7. `runoff.rainfall`: must name one of the climate products, `rainfall_total_<year>` for a
   year you listed.
8. `validation.hotspots`: a CSV like `data/lagos/hotspots.csv` (columns `name`,
   `admin_unit`, `lon`, `lat`, `source`) for the new region.
9. `scenarios.app_data_dir` and `app.admin_units`: new paths such as
   `app/data/<name>` and `data/<name>/lga_boundaries.geojson`.
10. `climate.station.enabled: false` unless you have a station workbook in the same
    layout.

The hazard settings (factors, weights, pairwise matrix, zones), land cover classes and
curve number table can stay as they are. Review them for the new setting: for a dry,
elevated state the land cover scores and weights chosen for coastal Lagos may not fit.

## 3. Run

```
export FLOOD_DATA_DIR=/path/to/raw/data      # PowerShell: $env:FLOOD_DATA_DIR = "..."
python -m flood fetch-boundaries --region <name>
python -m flood run --region <name>
python -m flood status --region <name>
```

The terrain stage needs QGIS with GRASS (see the README). The land cover stage is the
slowest, about an hour for Lagos on a 4-core PC.

Outputs appear in `outputs/<name>/`, figures in `outputs/<name>/figures/`, and the app
files in `app/data/<name>/`. Point `DATA_DIR` in `app/app.py` at the new folder to view it.
