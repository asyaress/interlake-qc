"""Pairing built from processed SWOT must match mahakam_pairs.csv reference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interlake_qc.core import load_d2_config, prepare_pair_table
from interlake_qc.validation_yahara import (
    build_pair_table,
    compute_bias_and_ec,
    d_preferred_else_c,
    dedup_prefer_D,
    load_r7_config,
    load_yahara_raw,
    main_set_mask,
)

REPO = Path(__file__).resolve().parents[1]
TOL = 1e-9


def test_mahakam_pairing_from_swot():
    cfg = load_d2_config(REPO, "configs/mahakam.yaml")
    swot = pd.read_csv(REPO / "data/processed/mahakam_swot_lakesp.csv")
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    phase = pd.read_csv(REPO / "data/processed/mahakam_dwse_phase.csv")
    built = prepare_pair_table(swot, cfg, phase_df=phase).sort_values("time").reset_index(drop=True)
    ref = pd.read_csv(REPO / "data/processed/mahakam_pairs.csv")
    ref["time"] = pd.to_datetime(ref["time"], utc=True)
    ref = ref.sort_values("time").reset_index(drop=True)
    assert len(built) == len(ref)
    assert (built["time"] == ref["time"]).all()
    for c in ("dWSE", "H", "wse_a", "wse_b", "q_a", "q_b"):
        g = pd.to_numeric(built[c], errors="coerce").to_numpy(float)
        r = pd.to_numeric(ref[c], errors="coerce").to_numpy(float)
        m = np.isfinite(g) & np.isfinite(r)
        assert float(np.max(np.abs(g[m] - r[m]))) <= TOL, c
    # pass: CSV may store 284.0 vs built "284"
    bp = pd.to_numeric(built["pass"], errors="coerce")
    rp = pd.to_numeric(ref["pass"], errors="coerce")
    m = bp.notna() & rp.notna()
    assert (bp[m].astype(int).to_numpy() == rp[m].astype(int).to_numpy()).all()
    assert bp.isna().sum() == rp.isna().sum()


def test_yahara_pairing_builds_from_subset():
    r7 = load_r7_config(REPO, "configs/yahara.yaml")
    gage0, _, _ = load_yahara_raw(REPO, r7)
    gage, _ = dedup_prefer_D(gage0)
    gage = d_preferred_else_c(gage)
    gage = compute_bias_and_ec(gage, float(r7["truth"]["max_gage_dt_hours"]), main_mask=main_set_mask(gage, True))
    spec = r7["amendment_1"]["pairs"]["P2"]
    pairs = build_pair_table(gage, spec["danau_a"], spec["danau_b"], 120.0, require_ice0=True)
    assert len(pairs) >= 64
    assert int((pairs["main"] & pairs["both_truth"]).sum()) == 64
