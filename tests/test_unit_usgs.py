"""Unit tests: C/D dedup and ECCC gage keys."""
from __future__ import annotations

import pandas as pd

from interlake_qc.io_usgs_release import norm_gage_id
from interlake_qc.validation_yahara import d_preferred_else_c, dedup_prefer_D


def test_eccc_gage_keys_distinct():
    a1, g1 = norm_gage_id("ECCC", "06AF002")
    a2, g2 = norm_gage_id("ECCC", "06AC002")
    assert (a1, g1) != (a2, g2)
    assert g1 == "06AF002"
    assert g2 == "06AC002"


def test_dedup_prefer_d():
    rows = []
    for ver, wse in (("C", 1.0), ("D", 2.0), ("C", 1.1)):
        rows.append(
            {
                "lake_id": "1",
                "cycle_id": "10",
                "pass_id": "100",
                "SWOT_time": pd.Timestamp("2024-01-01T12:00:00Z"),
                "SWOT_data_version": ver,
                "lake": "test",
                "wse": wse,
            }
        )
    # two C at same key + one D
    g = pd.DataFrame(rows)
    # make first two same key, third different second
    g.loc[2, "SWOT_time"] = pd.Timestamp("2024-01-02T12:00:00Z")
    ded, _ = dedup_prefer_D(g)
    # for the shared key, D should win
    day1 = ded[ded["SWOT_time"] == pd.Timestamp("2024-01-01T12:00:00Z")]
    assert len(day1) == 1
    assert str(day1.iloc[0]["SWOT_data_version"]) == "D"
    pref = d_preferred_else_c(ded)
    assert len(pref) >= 1
