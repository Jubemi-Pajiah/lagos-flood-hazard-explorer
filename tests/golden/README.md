# Golden outputs

## `lagos/original_run/`

Results of the analysis scripts this package replaced, used by `tests/test_golden.py` to
check that the refactored hazard, exposure and validation code reproduces them when fed
the same intermediate rasters (set `FLOOD_GOLDEN_DIR` to the folder of saved rasters).

* `exposure_by_lga.csv`: as saved, with LGA names mapped to the spelling used here.
* `exposure_by_zone.csv`: the state total recomputed over cells inside Lagos. The saved
  table counted about 2.09 million people living outside the state boundary but inside
  its bounding box, then spread them over the zones in proportion.
* `landcover_accuracy.json`: from the saved accuracy report.
* `rasters.json`: shape and summary statistics of the saved hazard rasters.

## `lagos/baseline/`

The published results: a full run from raw data with the current code and
`regions/lagos.yaml`, checked by `tests/test_baseline.py` against `outputs/lagos`.
`docs/limitations.md` (section "Corrections to the earlier analysis") explains why these
differ from the original run.
