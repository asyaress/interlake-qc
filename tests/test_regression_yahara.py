"""Regression: Yahara R7b-A1 outputs vs reference/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "yahara"
REF = REPO / "reference"
TOL = 1e-9
COUNT_COLS = ["n_pass", "n_bermasalah", "n_ditandai", "tp", "fp", "fn", "tn"]


def _ensure_outputs():
    need = ["r7b_metrics.csv", "r7b_flagged.csv", "r7b_d2_waubesa_minus_kegonsa.csv", "r7b_rmse_by_lake.csv"]
    if all((OUT / f).exists() for f in need):
        return
    subprocess.run([sys.executable, "scripts/21_run_yahara.py"], cwd=REPO, check=True)


@pytest.fixture(scope="module", autouse=True)
def yahara_outputs():
    _ensure_outputs()


def test_metrics_counts_identical():
    got = pd.read_csv(OUT / "r7b_metrics.csv")
    ref = pd.read_csv(REF / "r7b_metrics.csv")
    keys = ["pasangan", "peran", "k", "T", "himpunan", "versi", "metode"]
    got = got.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    assert len(got) == len(ref)
    for c in keys:
        assert got[c].astype(str).tolist() == ref[c].astype(str).tolist(), c
    for c in COUNT_COLS:
        assert got[c].tolist() == ref[c].tolist(), c


def test_flagged_identical():
    got = pd.read_csv(OUT / "r7b_flagged.csv")
    ref = pd.read_csv(REF / "r7b_flagged.csv")
    got = got.sort_values(["tanggal", "pair"]).reset_index(drop=True)
    ref = ref.sort_values(["tanggal", "pair"]).reset_index(drop=True)
    assert len(got) == len(ref)
    for c in ("tanggal", "sumber_akhir", "cara_atribusi"):
        assert got[c].astype(str).tolist() == ref[c].astype(str).tolist(), c
    for c in ("H", "dWSE", "z"):
        g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        assert float(np.nanmax(np.abs(g - r))) <= TOL, c


def test_d2_p2_numeric():
    got = pd.read_csv(OUT / "r7b_d2_waubesa_minus_kegonsa.csv")
    ref = pd.read_csv(REF / "r7b_d2_waubesa_minus_kegonsa.csv")
    got["_t"] = pd.to_datetime(got["time"], utc=True)
    ref["_t"] = pd.to_datetime(ref["time"], utc=True)
    got = got.sort_values("_t").reset_index(drop=True)
    ref = ref.sort_values("_t").reset_index(drop=True)
    assert len(got) == len(ref)
    for c in ("z", "f_H", "sigma_H", "dWSE", "H"):
        g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        assert np.array_equal(np.isnan(g), np.isnan(r)), c
        m = np.isfinite(g) & np.isfinite(r)
        if m.any():
            assert float(np.max(np.abs(g[m] - r[m]))) <= TOL, c
    for c in ("sumber_akhir", "cara_atribusi"):
        if c in got.columns and c in ref.columns:
            assert got[c].fillna("").astype(str).tolist() == ref[c].fillna("").astype(str).tolist(), c


def test_rmse_tol():
    got = pd.read_csv(OUT / "r7b_rmse_by_lake.csv")
    ref = pd.read_csv(REF / "r7b_rmse_by_lake.csv")
    keys = ["danau", "variant"]
    got = got.sort_values(keys).reset_index(drop=True)
    ref = ref.sort_values(keys).reset_index(drop=True)
    assert len(got) == len(ref)
    assert got["n"].tolist() == ref["n"].tolist()
    for c in ("rmse_e_c", "mad_e_c"):
        g = pd.to_numeric(got[c], errors="coerce").to_numpy(dtype=float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(g) & np.isfinite(r)
        if m.any():
            assert float(np.max(np.abs(g[m] - r[m]))) <= TOL, c


def test_attribution_has_himpunan():
    attr = pd.read_csv(OUT / "r7b_attribution.csv")
    assert "himpunan" in attr.columns
