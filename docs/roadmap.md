# Roadmap

## Multi-state comparison

Run the same pipeline for seven Nigerian states that span a gradient of flood exposure,
using one consistent method so the results can be compared.

| State | Setting | Role in the comparison |
|---|---|---|
| Lagos | coastal, urban | done |
| Delta | Niger Delta, coastal surge | flood-prone |
| Kogi | Niger and Benue confluence, riverine | flood-prone (large floods in 2012 and 2022) |
| Benue | Benue River agricultural floodplain | flood-prone |
| Bayelsa | rural Niger Delta | flood-prone |
| Plateau | high (about 1,200 m), rocky, well drained | low-hazard contrast (topography) |
| Kano | semi-arid, about 800 mm of rain a year | low-hazard contrast (climate) |

Most inputs already cover the whole country (SRTM, CHIRPS, HydroRIVERS, SoilGrids,
WorldPop, WorldCover). Sentinel-2 bands B02, B03, B04 and B08 must be downloaded per
state. Each state is a new region file; see `adding-a-region.md`.

Runoff stays with the SCS curve number method in every state: it needs no calibration
data, so it can be applied the same way everywhere.

## Data that would strengthen the model

| Priority | Dataset | Why |
|---|---|---|
| High | Observed flood extents (Copernicus EMS, UNOSAT; 2012, 2017, 2020, 2022 events) | a real validation set with flooded and non-flooded locations, allowing ROC/AUC |
| High | Sentinel-1 SAR | flood extent under cloud |
| High | FABDEM (buildings and forest removed) | better elevation in dense urban areas than SRTM |
| High | Longer CHIRPS record (2010 onwards) | rainfall statistics from more than two seasons |
| High | JRC Global Surface Water | mask permanent water out of the hazard map |
| Medium | GHSL built-up, OpenStreetMap | exposure of buildings and infrastructure |
| Medium | ERA5 reanalysis | consistent climate inputs across states |
| Medium | Gauged discharge (Niger at Lokoja, Benue at Makurdi) | check runoff patterns on the rivers |
| Later | CMIP6 / CORDEX Africa projections | future rainfall scenarios |
| Later | EM-DAT losses by state | compare hazard rankings with recorded losses |

## Model improvements

* Mask permanent water before zoning and report land-only statistics by default.
* Drive runoff with design storms instead of a seasonal total.
* Replace the documented hotspot list with an inventory that includes non-flooded
  controls, so skill can be measured.
* Use constrained WorldPop, or building footprints, to place population more precisely.
