import numpy as np

from flood.exposure import AdminUnit, admin_table, zone_table

NAMES = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}


def test_zone_table_counts_only_mapped_cells():
    zones = np.array([[1, 4], [5, 0]], dtype="uint8")
    pop = np.array([[10.0, 20.0], [30.0, 1000.0]])
    table = zone_table(zones, pop, NAMES, cell_km2=0.0009)
    assert table.population.sum() == 60
    assert table.set_index("zone").loc[5, "pct_total"] == 50.0


def test_admin_table_ranks_by_high_share():
    zones = np.array([[1, 4], [5, 3]], dtype="uint8")
    pop = np.array([[10.0, 30.0], [10.0, 50.0]])
    a = AdminUnit(1, "A", np.array([[True, True], [False, False]]))
    b = AdminUnit(2, "B", np.array([[False, False], [True, True]]))
    table = admin_table(zones, pop, [a, b], NAMES, [4, 5])
    assert table.lga_name.tolist() == ["A", "B"]
    assert table.iloc[0].pct_high_very_high == 75.0
    assert table.iloc[1].zone3_moderate == 50
