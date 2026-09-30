# Limitations

Numbers below come from `outputs/lagos/` after `python -m flood run --region lagos`.

## What the hazard map is and is not

* **An index, not a hydraulic model.** The hazard index is an AHP weighted sum of eight
  normalised factors. It ranks places by how many flood-favouring conditions they
  combine. It does not simulate water, predict depth or timing, or give a probability.
  It has not been calibrated against observed flooding.
* **The weights are judgements.** They come from published AHP studies and were adjusted
  for flat, coastal Lagos. The stored pairwise matrix is consistent (CR 0.029, below 0.10),
  but its eigenvector differs from the weights by up to 0.0069; using the eigenvector
  instead would move 89,133 cells (2.1% of the map) to another zone.
* **Zones use fixed breaks** (0.2, 0.4, 0.6, 0.8), so small shifts in the index near a
  break move cells between zones. This is why rainfall scenarios move many people (below).

## Checks

* **Hotspots are not a skill score.** Of 20 locations repeatedly reported as flooded, 6
  (30%) fall in High or Very High zones and 19 (95%) in Moderate or above; one (Ikotun) is
  in Low. The list is a convenience sample with approximate coordinates and no non-flooded
  control sites, so it cannot measure false alarms and no ROC or AUC can be computed.
* **Land cover accuracy is modest.** The Random Forest reaches 51.8% overall accuracy
  (kappa 0.397) on 15,000 holdout pixels, with per-class F1 between 0.47 (urban) and 0.62
  (bare soil). Accuracy is measured against WorldCover labels, which have their own
  errors. Land cover enters the index directly (weight 0.05) and through the curve numbers.

## Inputs

* **Water is not masked.** Lagoons and sea surfaces inside the boundary are scored like
  land. Of the 363.3 km2 Very High zone, 159.3 km2 (43.8%) is water in the land cover map.
  Of the 3,765.7 km2 mapped, 702.2 km2 is water. Land-only figures: High plus Very High
  covers 746.1 km2, 24.4% of the 3,063.5 km2 of land (31.1% of all mapped area).
* **Runoff is not a design value.** SCS-CN is an event method, but it is driven here by
  the 2021 March to November rainfall total (mean 1,573 mm). The result, a mean runoff
  depth of 1,492 mm and a runoff coefficient of 0.95, is not physically meaningful as
  runoff. It is used only to rank places relative to each other. Varying curve numbers by
  10% changes mean runoff to 1,453-1,520 mm, by 20% to 1,406-1,542 mm, and dry to wet
  antecedent conditions give 1,394-1,536 mm.
* **Soil gaps are assumed permeable.** SoilGrids has no clay value for 46% of Lagos
  cells, much of it built-up land and water. Those cells are put in soil group A, the
  most permeable, as in the earlier analysis (`runoff.hsg_where_missing`). This lowers
  curve numbers in built-up areas and understates their runoff.
* **Rainfall is coarse and short.** CHIRPS is about 5 km, covers two rainy seasons
  (2020 and 2021, monthly), and has no values over open water; boundary cells without a
  value take the nearest CHIRPS value.
* **Terrain.** SRTM is a surface model (it includes buildings and trees), 30 m, with a few
  metres of vertical error, which matters on terrain this flat. Flow accumulation and TWI
  show straight-line artefacts across flat lagoon surfaces. About 16 km2 of edge cells
  have no slope or TWI and are left unmapped.
* **Boundaries.** The model uses GADM boundaries (read from `FLOOD_DATA_DIR`, not
  redistributed). The app draws geoBoundaries outlines, which extend further over the
  Lekki Lagoon, so a small area inside those outlines is blank on the map.

## Population

* **Source.** WorldPop / GRID3 bottom-up estimates for Nigeria, version 1.2 (people per
  100 m cell, released January 2020). People are placed only in cells mapped as settled,
  and the estimates do not account for growth since then.
* **Totals.** The 20 LGAs hold 10,338,873 people; 10,315,258 of them live in cells with a
  hazard zone. The difference (23,615) is people in edge cells without slope or TWI.
* **Exposure is residence, not impact.** Counts are people living in a zone, not people
  expected to be flooded.

## Scenarios

Rainfall scenarios scale the rainfall factor and recompute runoff; the other six factors,
carrying 0.80 of the weight, stay fixed. The two rainfall-driven factors carry only 0.20
of the weight, yet exposure responds strongly: people in High or Very High zones go from
1,214,171 at baseline to 554,819 at 0.8 times rainfall and 3,091,064 at 1.3 times. Many
densely populated cells sit just below the 0.6 break between Moderate and High, so a
small rise in the index moves them. Read scenarios as a sensitivity test of the index,
not as a forecast of what wetter years would bring. Return-period scenarios are not
included.

## Corrections to the earlier analysis

This code replaces an earlier set of scripts. Run on the same intermediate rasters, it
reproduces their hazard map cell for cell (tests in `tests/test_golden.py`). Running from
raw data exposed problems in how those intermediates were made, which are fixed here and
change the published numbers:

| Issue in the earlier run | Effect | Now |
|---|---|---|
| Percentile bounds for normalisation were taken over the whole bounding box, including zero-filled cells outside Lagos | inflated distance-to-river, runoff and rainfall scores across the state | bounds use cells inside the boundary only |
| Three CHIRPS months (April 2020, April 2021, August 2020) were clipped as all zeros | 2021 total about 220 mm too low | all 18 months used |
| Statewide exposure spread the bounding-box population (about 2.1 million people outside Lagos) over the zones | statewide counts about 20% too high | counts cells inside Lagos |
| Distance to rivers used a partial river clip | fewer river cells | every HydroRIVERS segment in the grid |

With these fixes, people in High or Very High zones fall from 3.10 million (earlier run,
30.0%) to 1.21 million (11.8%), and hotspots in High or Very High from 13 of 20 to 6 of 20.
