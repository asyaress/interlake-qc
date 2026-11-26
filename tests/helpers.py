"""Shared helpers for regression gates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from interlake_qc.schema import golden_to_english

TOL = 1e-9


def assert_frame_close(
    got: pd.DataFrame,
    ref: pd.DataFrame,
    *,
    key_cols: list[str] | None = None,
    num_tol: float = TOL,
    exclude: set[str] | None = None,
    label: str = "",
    ref_is_golden: bool = True,
) -> None:
    exclude = exclude or set()
    if ref_is_golden:
        ref = golden_to_english(ref)
    # also normalize got column names if any legacy slipped through
    got = golden_to_english(got)
    if key_cols:
        # map legacy key names
        from interlake_qc.schema import COLUMN_MAP
        key_cols = [COLUMN_MAP.get(c, c) for c in key_cols]
        for c in key_cols:
            if c not in got.columns or c not in ref.columns:
                raise AssertionError(f"{label} missing key {c} got={list(got.columns)[:20]} ref={list(ref.columns)[:20]}")
        got = got.sort_values(key_cols).reset_index(drop=True)
        ref = ref.sort_values(key_cols).reset_index(drop=True)
    else:
        got = got.reset_index(drop=True)
        ref = ref.reset_index(drop=True)
    assert len(got) == len(ref), f"{label} rows {len(got)} vs {len(ref)}"
    cols = [c for c in ref.columns if c in got.columns and c not in exclude]
    for c in cols:
        g = got[c]
        r = ref[c]
        gv = pd.to_numeric(g, errors="coerce")
        rv = pd.to_numeric(r, errors="coerce")
        if gv.notna().mean() > 0.5 or rv.notna().mean() > 0.5:
            gvv = gv.to_numpy(dtype=float)
            rvv = rv.to_numpy(dtype=float)
            assert np.array_equal(np.isnan(gvv), np.isnan(rvv)), f"{label} {c} NaN mask"
            m = np.isfinite(gvv) & np.isfinite(rvv)
            if m.any():
                md = float(np.max(np.abs(gvv[m] - rvv[m])))
                assert md <= num_tol, f"{label} {c} maxdiff={md}"
            continue
        assert g.astype("string").fillna("").tolist() == r.astype("string").fillna("").tolist(), f"{label} {c}"
