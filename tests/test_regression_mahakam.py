"""Regression: Mahakam outputs vs reference/ golden files."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "mahakam"
REF = REPO / "reference"
TOL = 1e-9
NUM_COLS = ["z", "f_H", "sigma_H", "dWSE", "H"]


def _ensure_outputs():
    need = ["r5_d2_final.csv", "r5_flagged_passes.csv", "r5_reference_bins.csv"]
    if all((OUT / f).exists() for f in need):
        return
    subprocess.run([sys.executable, "scripts/20_run_mahakam.py"], cwd=REPO, check=True)


@pytest.fixture(scope="module", autouse=True)
def mahakam_outputs():
    _ensure_outputs()


def _align_by_time(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = a.copy()
    b = b.copy()
    a["_t"] = pd.to_datetime(a["time"], utc=True)
    b["_t"] = pd.to_datetime(b["time"], utc=True)
    a = a.sort_values("_t").reset_index(drop=True)
    b = b.sort_values("_t").reset_index(drop=True)
    assert len(a) == len(b), f"row count {len(a)} vs {len(b)}"
    return a, b


def test_d2_final_numeric_and_flags():
    got, ref = _align_by_time(pd.read_csv(OUT / "r5_d2_final.csv"), pd.read_csv(REF / "r5_d2_final.csv"))
    for c in NUM_COLS:
        if c in got.columns and c in ref.columns:
            g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
            r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
            # NaN positions must match
            assert np.array_equal(np.isnan(g), np.isnan(r)), f"{c} NaN mask differs"
            m = np.isfinite(g) & np.isfinite(r)
            if m.any():
                maxdiff = float(np.max(np.abs(g[m] - r[m])))
                assert maxdiff <= TOL, f"{c} maxdiff={maxdiff}"
    for c in ("sumber_akhir", "cara_atribusi", "flag"):
        if c in got.columns and c in ref.columns:
            assert got[c].astype(str).tolist() == ref[c].astype(str).tolist(), c


def test_flagged_passes_identical():
    got = pd.read_csv(OUT / "r5_flagged_passes.csv")
    ref = pd.read_csv(REF / "r5_flagged_passes.csv")
    assert len(got) == len(ref)
    for c in ("tanggal", "sumber_akhir", "cara_atribusi"):
        assert got[c].astype(str).tolist() == ref[c].astype(str).tolist(), c
    for c in ("H", "dWSE", "z"):
        g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        assert float(np.nanmax(np.abs(g - r))) <= TOL


def test_reference_bins():
    got = pd.read_csv(OUT / "r5_reference_bins.csv")
    ref = pd.read_csv(REF / "r5_reference_bins.csv")
    assert len(got) == len(ref)
    for c in ("center", "f", "sigma", "n"):
        g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        assert float(np.nanmax(np.abs(g - r))) <= TOL, c
