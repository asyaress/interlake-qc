"""Regression: Mahakam outputs vs reference/ golden files."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from helpers import assert_frame_close

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "mahakam"
REF = REPO / "reference"
NUM_COLS = ["z", "f_H", "sigma_H", "dWSE", "H"]


def _ensure_outputs():
    need = ["r5_d2_final.csv", "r5_flagged_passes.csv", "r5_reference_bins.csv"]
    if all((OUT / f).exists() for f in need):
        return
    subprocess.run([sys.executable, "scripts/20_run_mahakam.py"], cwd=REPO, check=True)


@pytest.fixture(scope="module", autouse=True)
def mahakam_outputs():
    _ensure_outputs()


def test_d2_final_numeric_and_flags():
    got = pd.read_csv(OUT / "r5_d2_final.csv")
    ref = pd.read_csv(REF / "r5_d2_final.csv")
    got["time"] = pd.to_datetime(got["time"], utc=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    assert_frame_close(got, ref, key_cols=["time"], label="r5_d2_final")


def test_flagged_passes_identical():
    got = pd.read_csv(OUT / "r5_flagged_passes.csv")
    ref = pd.read_csv(REF / "r5_flagged_passes.csv")
    assert_frame_close(
        got,
        ref,
        key_cols=["date"] if "date" in got.columns else ["tanggal"],
        label="r5_flagged",
    )


def test_reference_bins():
    assert_frame_close(
        pd.read_csv(OUT / "r5_reference_bins.csv"),
        pd.read_csv(REF / "r5_reference_bins.csv"),
        key_cols=["center"],
        label="r5_reference_bins",
    )
