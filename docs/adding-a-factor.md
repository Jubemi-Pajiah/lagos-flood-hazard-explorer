# Adding a hazard factor

A factor is a raster on the region grid where 1 means more hazard after normalisation.
Adding one takes three steps: produce the raster, register it in the yaml, and extend the
weights and the AHP matrix.

## Worked example: distance to the coast

Low-lying land near the sea floods from storm surge and high tides, which none of the
current factors describe directly.

### 1. Write a builder

If the factor is already a raster produced by a stage, skip to step 2 and use the
built-in `raster` builder. Otherwise add a function to `flood/factors.py`. It receives
the region and the factor's yaml settings and returns a float32 array on `region.grid`,
NaN where there is no value:

```python
from scipy.ndimage import distance_transform_edt

from flood import preprocess
from flood.io import inside_mask, read_vector


@register(
    "distance_to_coast",
    inputs=lambda region, spec: [region.resolve_data(spec["coastline"], "coastline")],
)
def distance_to_coast(region, spec):
    """Distance in metres from each cell to the nearest coastline cell."""
    grid = region.grid
    coast = read_vector(region.resolve_data(spec["coastline"], "coastline"), grid.crs)
    on_coast = inside_mask(coast.buffer(grid.resolution).geometry, grid)
    dist = distance_transform_edt(~on_coast).astype("float32") * grid.resolution
    boundary = read_vector(preprocess.boundary_path(region))
    return np.where(inside_mask(boundary.geometry, grid), dist, np.nan)
```

`inputs` lists the files the builder reads so the hazard stage reruns when they change.
Without it the factor's `source` file is used.

### 2. Register it in the yaml

Add an entry to `hazard.factors` in `regions/lagos.yaml`:

```yaml
    - name: distance_to_coast
      label: Distance to the coast
      builder: distance_to_coast
      coastline: coastlines/nigeria_coast.shp   # relative to FLOOD_DATA_DIR
      direction: cost                           # closer to the coast = more hazard
      weight: 0.06
```

* `direction: benefit` means a higher raw value is more hazardous; `cost` means a lower
  value is. Continuous factors are rescaled between the 2nd and 98th percentile.
* For a class raster, use `builder: class_scores`, `normalise: false`, and a `scores`
  mapping, as the land cover factor does.
* For a raster another stage already writes, use `builder: raster` (the default) and
  `source: <path under outputs/<region>/>`; add `transform: log1p` for skewed layers.

### 3. Update the weights and the matrix

The weights must still sum to 1: lower others to make room (here 0.06 taken from the
existing factors). Then add a row and a column for the new factor to
`hazard.pairwise_matrix`. The matrix is written as its upper triangle, in the same order
as `hazard.factors`, so a ninth factor adds one entry to the end of every existing row
(its judgement against that row's factor) and a final row `[1]`:

```yaml
  pairwise_matrix:
    - [1, 1, 1, 3, 2, 2, 3, 2, 3]    # elevation vs ... vs distance_to_coast
    - [1, 1, 1, 2, 2, 2, 5, 3]
    ...
    - [1]
```

### 4. Run and check

```
python -m flood run --region lagos --stage hazard
```

The hazard stage reruns because the `hazard` section changed. Read
`outputs/lagos/hazard/ahp.json`: `cr` must stay below 0.10, and
`max_abs_difference` shows how far the matrix's eigenvector is from your weights. Every
later stage (exposure, validation, scenarios, figures) reruns on the next full `run`.

If the new factor depends on rainfall, add its name to `scenarios.rainfall_factors` so
scenarios scale it.
