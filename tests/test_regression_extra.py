"""Extra golden gates: ablation, btemp, jempang, Yahara bins, R6, low-water."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from helpers import assert_frame_close

REPO = Path(__file__).resolve().parents[1]
OUT_M = REPO / "outputs" / "mahakam"
OUT_Y = REPO / "outputs" / "yahara"
OUT_E = REPO / "outputs" / "evidence"
REF = REPO / "reference"


def _run(script: str):
    subprocess.run([sys.executable, script], cwd=REPO, check=True)


@pytest.fixture(scope="module", autouse=True)
def ensure_mahakam_yahara():
    need_m = ["r5_ablation.csv", "r5_btemp.csv", "r5_jempang.csv", "r5_jempang_d2.csv", "r5_jempang_bins.csv"]
    if not all((OUT_M / f).exists() for f in need_m):
        _run("scripts/20_run_mahakam.py")
    need_y = [
        "r7b_bins_waubesa_minus_kegonsa.csv",
        "r7b_bins_monona_minus_waubesa.csv",
        "r7b_bins_mendota_minus_monona.csv",
    ]
    if not all((OUT_Y / f).exists() for f in need_y):
        _run("scripts/21_run_yahara.py")


def test_r5_ablation():
    assert_frame_close(
        pd.read_csv(OUT_M / "r5_ablation.csv"),
        pd.read_csv(REF / "r5_ablation.csv"),
        key_cols=["variant", "lake"],
        label="r5_ablation",
    )


def test_r5_btemp():
    got = pd.read_csv(OUT_M / "r5_btemp.csv")
    ref = pd.read_csv(REF / "r5_btemp.csv")
    got["time"] = pd.to_datetime(got["time"], utc=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    assert_frame_close(got, ref, key_cols=["lake", "time"], label="r5_btemp")


def test_r5_jempang_summary():
    assert_frame_close(
        pd.read_csv(OUT_M / "r5_jempang.csv"),
        pd.read_csv(REF / "r5_jempang.csv"),
        key_cols=["metric"],
        label="r5_jempang",
    )


def test_r5_jempang_d2_and_bins():
    got = pd.read_csv(OUT_M / "r5_jempang_d2.csv")
    ref = pd.read_csv(REF / "r5_jempang_d2.csv")
    got["time"] = pd.to_datetime(got["time"], utc=True)
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    assert_frame_close(
        got,
        ref,
        key_cols=["time"],
        exclude={"pass"},  # formatting 284 vs 284.0
        label="r5_jempang_d2",
    )
    assert_frame_close(
        pd.read_csv(OUT_M / "r5_jempang_bins.csv"),
        pd.read_csv(REF / "r5_jempang_bins.csv"),
        key_cols=["center"],
        label="r5_jempang_bins",
    )


@pytest.mark.parametrize(
    "name",
    [
        "waubesa_minus_kegonsa",
        "monona_minus_waubesa",
        "mendota_minus_monona",
    ],
)
def test_r7b_bins(name):
    assert_frame_close(
        pd.read_csv(OUT_Y / f"r7b_bins_{name}.csv"),
        pd.read_csv(REF / f"r7b_bins_{name}.csv"),
        key_cols=["center"] if "center" in pd.read_csv(REF / f"r7b_bins_{name}.csv").columns else None,
        label=f"r7b_bins_{name}",
    )


@pytest.fixture(scope="module")
def r6_outputs():
    need = ["r6_oat.csv", "r6_loyo.csv", "r6_by_pass_curve.csv", "r6_flag_frequency.csv"]
    if not all((OUT_M / f).exists() for f in need):
        _run("scripts/22_sensitivity_mahakam.py")


def test_r6_gates(r6_outputs):
    for name, keys in (
        ("r6_oat.csv", ["config_id"]),
        ("r6_flag_frequency.csv", None),
        ("r6_by_pass_curve.csv", None),
        ("r6_loyo.csv", None),
    ):
        got = pd.read_csv(OUT_M / name)
        ref = pd.read_csv(REF / name)
        runtime_cols = {
            c
            for c in set(got.columns) | set(ref.columns)
            if "runtime" in c.lower() or c.endswith("_sec") or c in {"elapsed", "wall_time"}
        }
        assert_frame_close(got, ref, key_cols=keys, exclude=runtime_cols, label=name)


@pytest.fixture(scope="module")
def evidence_outputs():
    need = ["r4b_atl13_swot_window.csv", "r4b_spearman_dark.csv"]
    if not all((OUT_E / f).exists() for f in need) and not all((OUT_M / f).exists() for f in need):
        _run("scripts/23_evidence_tables.py")


def test_lowwater_gates(evidence_outputs):
    for name in ("r4b_atl13_swot_window.csv", "r4b_spearman_dark.csv"):
        got_path = OUT_E / name if (OUT_E / name).exists() else OUT_M / name
        assert_frame_close(
            pd.read_csv(got_path),
            pd.read_csv(REF / name),
            label=name,
        )
