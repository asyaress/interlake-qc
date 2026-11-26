"""Regression: Yahara R7b-A1 outputs vs reference/."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from helpers import assert_frame_close

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs" / "yahara"
REF = REPO / "reference"


def _ensure_outputs():
    need = ["r7b_metrics.csv", "r7b_flagged.csv", "r7b_d2_waubesa_minus_kegonsa.csv", "r7b_rmse_by_lake.csv"]
    if all((OUT / f).exists() for f in need):
        return
    subprocess.run([sys.executable, "scripts/21_run_yahara.py"], cwd=REPO, check=True)


@pytest.fixture(scope="module", autouse=True)
def yahara_outputs():
    _ensure_outputs()


def test_metrics_counts_identical():
    assert_frame_close(
        pd.read_csv(OUT / "r7b_metrics.csv"),
        pd.read_csv(REF / "r7b_metrics.csv"),
        key_cols=["pair", "role", "k", "T", "subset", "versi", "method"],
        label="r7b_metrics",
    )


def test_flagged_identical():
    assert_frame_close(
        pd.read_csv(OUT / "r7b_flagged.csv"),
        pd.read_csv(REF / "r7b_flagged.csv"),
        key_cols=["date", "pair"],
        label="r7b_flagged",
    )


def test_d2_p2_numeric():
    got = pd.read_csv(OUT / "r7b_d2_waubesa_minus_kegonsa.csv")
    ref = pd.read_csv(REF / "r7b_d2_waubesa_minus_kegonsa.csv")
    got["time"] = pd.to_datetime(got["time"], utc=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    assert_frame_close(
        got,
        ref,
        key_cols=["time"],
        label="r7b_d2_p2",
    )


def test_rmse_tol():
    assert_frame_close(
        pd.read_csv(OUT / "r7b_rmse_by_lake.csv"),
        pd.read_csv(REF / "r7b_rmse_by_lake.csv"),
        key_cols=["lake", "variant"],
        label="r7b_rmse",
    )


def test_attribution_has_subset():
    attr = pd.read_csv(OUT / "r7b_attribution.csv")
    assert "subset" in attr.columns
