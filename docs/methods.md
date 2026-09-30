# Methods

All settings quoted here live in `regions/lagos.yaml`.

## Grids and preprocessing

Every hazard factor sits on one 30 m grid in UTM zone 31N (EPSG:32631), covering the
bounding box of Lagos State. The land cover classifier uses a matching 10 m grid.
Continuous inputs are resampled bilinearly, class inputs by nearest neighbour, and cells
outside the state boundary are set to no-data.

## Hazard factors

| Factor | Source | Direction | Weight |
|---|---|---|---|
| Elevation | SRTM 1 arc-second DEM | lower is more hazardous | 0.20 |
| Topographic wetness index | GRASS `r.watershed` on the sink-filled DEM | higher | 0.18 |
| Distance to rivers | Euclidean distance to HydroRIVERS lines rasterised on the grid | nearer | 0.15 |
| Flow accumulation | GRASS `r.watershed`, log(1 + x) transformed | higher | 0.12 |
| Runoff depth | SCS curve number method (below) | higher | 0.12 |
| Slope | GRASS `r.slope.aspect`, degrees | flatter | 0.10 |
| Rainfall | 95th percentile of the 18 monthly CHIRPS values (March to November, 2020 and 2021) per cell | higher | 0.08 |
| Land cover | Random Forest classes scored urban 0.90, bare soil 0.75, wetland 0.70, vegetation 0.30, water 0.10 | higher score | 0.05 |

Continuous factors are rescaled to 0-1 between their 2nd and 98th percentiles and clipped,
then reversed where lower values mean more hazard. The land cover scores are used as they
are.

## Weights and the AHP check

The eight weights are the model weights. They were set from published AHP flood studies
in comparable coastal and urban settings and adjusted for the flat Lagos terrain. A Saaty
pairwise comparison matrix expressing the same ranking is stored with them and checked
on every run: its principal eigenvector, lambda_max, consistency index and consistency
ratio (CR) are written to `outputs/lagos/hazard/ahp.json`, together with the largest
difference between the eigenvector and the weights and the number of cells that would
change zone if the eigenvector were used instead. The numbers are in the README.

## Hazard index and zones

The hazard index is the weighted sum of the eight normalised factors, defined where all
eight have a value. It is split into five zones at fixed breaks: Very Low (below 0.2),
Low (0.2-0.4), Moderate (0.4-0.6), High (0.6-0.8) and Very High (0.8 and above).

## Land cover

A Random Forest (500 trees, balanced class weights, minimum 5 samples per leaf) is
trained on Sentinel-2 L2A bands B02, B03, B04 and B08 (19 April 2026) and four indices:
NDVI, NDWI and two green/red ratios labelled MNDWI and NDBI. No SWIR band was used, so
those two are exact negatives of each other. Training labels come from ESA WorldCover 2021
mapped to five classes. For each class 10,000 pixels are sampled (random seed 42) and
split 70/30 into 7,000 training and 3,000 holdout pixels; the holdout gives the accuracy
figures. No smoothing is applied to the classified map.

## Runoff

Hydrologic soil groups come from SoilGrids topsoil clay (0-5 cm): A below 10%, B 10-20%,
C 20-40%, D 40% and above. Curve numbers follow a USDA TR-55 based table by land cover and
soil group (urban 77/85/90/92, vegetation 55/70/77/83, water 98, wetland 75/82/87/89,
bare soil 72/81/88/91). Runoff depth is

    S = 25400 / CN - 254,  Ia = 0.2 S,  Q = (P - Ia)^2 / (P - Ia + S) for P > Ia

with P the 2021 March to November CHIRPS total. This is a seasonal total, not a storm, so
Q is used only as a relative index (see limitations).

## Population exposure

WorldPop bottom-up gridded population estimates for Nigeria, version 1.2 (WorldPop and
GRID3, released January 2020), give people per 100 m cell. People are placed only in cells
mapped as settled (LandScanHD settlement layer). These counts are resampled onto the 30 m grid with a sum-preserving method and overlaid on the zones.
State totals count cells inside the hazard map; LGA totals count cells whose centre lies
in each LGA.

## Checks

* **Hotspots.** The zone at 20 locations repeatedly reported as flooded
  (`data/lagos/hotspots.csv`), with a 3x3 neighbourhood fallback where a point lands on a
  no-data cell.
* **Runoff sensitivity.** Curve numbers scaled by -20%, -10%, +10% and +20% (kept between
  30 and 98) and converted to dry (AMC I) and wet (AMC III) conditions.

## Scenarios

Rainfall is multiplied by 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 and 1.5. The rainfall factor is
scaled directly and runoff is recomputed from the scaled rainfall; the other six factors
are unchanged. Normalisation uses the baseline percentile bounds so that wetter scenarios
score higher. For each scenario the zones and exposure tables are recomputed at 30 m and
the zones and the baseline index are stored at 120 m for the app, both taken from the
centre cell of each 4 x 4 block so a point query's zone always matches its index.

## Station climate (optional)

With `climate.station.enabled`, a station workbook of monthly mean daily rain rate
(mm/day), temperature and humidity is summarised: monthly totals (rate x days in month),
climatology, Mann-Kendall trend with Theil-Sen slope, a Gumbel (method of moments) fit to
annual totals with return levels, and a monthly comparison with the state-mean CHIRPS
value. It is off by default.
