"""Unit tests: triangulation points_to."""
from __future__ import annotations

import pandas as pd

from interlake_qc.attribution import attribute_triangulation, is_named_source


def test_points_to_single_hit():
    cfg = {
        "triangulation": {
            "k_tri": 2.5,
            "pairs": [
                {"name": "a_minus_tri", "points_to": "semayang"},
                {"name": "b_minus_tri", "points_to": "melintang"},
            ],
        }
    }
    t0 = pd.Timestamp("2024-01-01T00:00:00Z")
    pairs = pd.DataFrame({"time": [t0]})
    res_a = pd.DataFrame({"time": [t0], "z": [5.0], "points_to": ["semayang"]})
    res_b = pd.DataFrame({"time": [t0], "z": [0.1], "points_to": ["melintang"]})
    tri_maps = {"a_minus_tri": res_a, "b_minus_tri": res_b}
    out = attribute_triangulation(0, pairs, pd.DataFrame(), tri_maps, cfg, k=3.5, temporal_sumber="")
    assert out == "semayang"


def test_is_named_source():
    assert is_named_source("semayang")
    assert not is_named_source("ambigu")
    assert not is_named_source("")
