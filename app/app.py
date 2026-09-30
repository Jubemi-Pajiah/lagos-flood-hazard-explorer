"""Streamlit app: hazard map, point query, exposure tables and rainfall scenarios.

Run from the repository root:  streamlit run app/app.py
Reads only the precomputed files in app/data/<region>/ (see flood.api).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flood.api import Explorer  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data" / "lagos"
DISCLAIMER = (
    "This is a hazard and scenario tool, not a dated flood forecast. The hazard map is a "
    "multi-criteria (AHP) index built from terrain, rainfall, runoff and land cover. It is "
    "not a calibrated hydraulic model and does not predict flood depth or timing."
)


@st.cache_resource
def explorer() -> Explorer:
    return Explorer.load(DATA_DIR)


@st.cache_data
def outlines() -> list[np.ndarray]:
    """Admin unit outlines in the grid CRS, as coordinate arrays for plotting."""
    ex = explorer()
    to_grid = Transformer.from_crs("EPSG:4326", ex.meta["crs"], always_xy=True)
    lines = []
    for _, geom in ex.admin_unit_shapes():
        polys = getattr(geom, "geoms", [geom])
        for poly in polys:
            x, y = to_grid.transform(*poly.exterior.xy)
            lines.append(np.column_stack([x, y]))
    return lines


def zone_map(zones: np.ndarray, title: str, marker: tuple[float, float] | None = None):
    ex = explorer()
    colors = ex.zone_colors
    cmap = ListedColormap(["#ffffff00", *[colors[z] for z in sorted(colors)]])
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 1), cmap.N)
    xmin, xmax, ymin, ymax = ex.extent
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.imshow(zones, cmap=cmap, norm=norm, extent=(xmin, xmax, ymin, ymax), interpolation="nearest")
    for line in outlines():
        ax.plot(line[:, 0], line[:, 1], color="#333333", linewidth=0.4)
    if marker is not None:
        ax.plot(*marker, marker="*", color="black", markersize=14)
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(
        handles=[Patch(color=colors[z], label=n) for z, n in ex.zone_names.items()],
        loc="lower left",
        fontsize=7,
        ncol=5,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Lagos Flood Hazard Explorer", layout="wide")
    ex = explorer()
    st.title("Lagos Flood Hazard Explorer")
    st.caption(DISCLAIMER)
    label = ex.meta["admin_label"]
    tab = st.tabs(["Hazard map", "Point query", "Exposure", "Rainfall scenarios"])

    with tab[0]:
        st.pyplot(zone_map(ex.zones(), f"Flood hazard zones, {ex.meta['region']}"))
        st.caption(
            f"Map cells are {ex.meta['cell_size_m']:.0f} m, downsampled from the 30 m "
            "model grid. Water bodies are not masked, so lagoon and sea surfaces are "
            "included in the zones."
        )

    with tab[1]:
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=6.4550, format="%.4f")
        lon = c2.number_input("Longitude", value=3.3900, format="%.4f")
        result = ex.point_query(lat, lon)
        if result["hazard_zone"] is None:
            st.warning("This point is outside the mapped area.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Hazard zone", result["hazard_zone_name"])
            m2.metric("Hazard index (0 to 1)", f"{result['hazard_index']:.3f}")
            m3.metric(label, result["admin_unit"] or "n/a")
            x, y = Transformer.from_crs("EPSG:4326", ex.meta["crs"], always_xy=True).transform(
                lon, lat
            )
            st.pyplot(zone_map(ex.zones(), "Location", marker=(x, y)))
        st.caption(
            "Values are read from the downsampled grid. Individual factor values "
            "(elevation, slope and so on) are not included in the deployed app."
        )

    with tab[2]:
        zones = ex.exposure_by_zone()
        high = ex.people_in_high_zones()
        total = int(zones.population.sum())
        st.metric(
            "People in High or Very High zones",
            f"{high:,}",
            f"{100 * high / total:.1f}% of {total:,} mapped residents",
            delta_color="off",
        )
        st.dataframe(zones, hide_index=True)
        table = ex.exposure_by_lga()
        st.subheader(f"By {label}")
        st.bar_chart(
            table.set_index("lga_name")["pct_high_very_high"],
            y_label="% of residents in High or Very High zones",
        )
        st.dataframe(table, hide_index=True)
        st.caption(
            "Population: WorldPop / GRID3 bottom-up gridded estimates for Nigeria, v1.2 "
            "(people per 100 m cell, settled areas only)."
        )

    with tab[3]:
        scenarios = ex.scenarios()
        choice = st.select_slider(
            "Scenario",
            options=list(scenarios.id),
            value=ex.baseline,
            format_func=lambda s: scenarios.set_index("id").label[s],
        )
        summary = ex.scenario_summary().set_index("scenario")
        row = summary.loc[choice]
        st.metric(
            "People in High or Very High zones",
            f"{int(row.people_high):,}",
            f"{int(row.change):+,} ({row.change_pct:+.1f}%) vs baseline",
        )
        st.pyplot(zone_map(ex.zones(choice), f"Hazard zones, {row.label}"))
        st.dataframe(
            summary.reset_index()[
                ["label", "rainfall_scale", "people_high", "change", "change_pct"]
            ],
            hide_index=True,
        )
        weight = sum(ex.meta["weights"].get(k, 0) for k in ("rainfall", "runoff"))
        st.caption(
            f"Rainfall changes two factors carrying {weight:.2f} of the index weight. "
            "Exposure still moves a lot because many populated cells sit just below the "
            "Moderate/High break. Treat scenarios as a sensitivity test, not a forecast."
        )


main()
