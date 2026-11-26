"""Full S1/S2 synthetic regression (slow)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from helpers import assert_frame_close

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "synthetic"
REF = REPO / "reference"


@pytest.mark.slow
def test_synth_full_s1_s2():
    need = ["r4b_synth_S1.csv", "r4b_S1_baseline_flag_rates.csv", "r4b_synth_S2.csv"]
    if not all((OUT / f).exists() for f in need):
        subprocess.run([sys.executable, "scripts/31_synthetic_full.py"], cwd=REPO, check=True)
    assert_frame_close(
        pd.read_csv(OUT / "r4b_synth_S1.csv"),
        pd.read_csv(REF / "r4b_synth_S1.csv"),
        label="r4b_synth_S1",
    )
    assert_frame_close(
        pd.read_csv(OUT / "r4b_S1_baseline_flag_rates.csv"),
        pd.read_csv(REF / "r4b_S1_baseline_flag_rates.csv"),
        label="r4b_S1_baseline",
    )
    assert_frame_close(
        pd.read_csv(OUT / "r4b_synth_S2.csv"),
        pd.read_csv(REF / "r4b_synth_S2.csv"),
        label="r4b_synth_S2",
    )
