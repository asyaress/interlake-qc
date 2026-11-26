"""ICESat-2 ATL13 segment QC (Luo & Song V2 daily). Credentials only via user .netrc."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def nmad_filter(heights: np.ndarray, factor: float) -> np.ndarray:
    """Boolean mask: |h - median| <= factor * MAD. Single pass (non-iterative)."""
    h = np.asarray(heights, dtype=float)
    ok = np.isfinite(h)
    out = np.zeros(len(h), dtype=bool)
    if ok.sum() == 0:
        return out
    med = np.median(h[ok])
    mad_raw = np.median(np.abs(h[ok] - med))
    thr = factor * mad_raw
    out[ok] = np.abs(h[ok] - med) <= thr
    return out


def atl13_daily_qc(seg: pd.DataFrame, cfg: dict, factor: float) -> tuple[pd.DataFrame, dict]:
    """Daily QC per lake: NMAD, min segments, max std."""
    min_n = int(cfg["atl13"]["min_segments_after_qc"])
    max_std = float(cfg["atl13"]["max_std_m"])
    stats: dict[str, Any] = {}
    rows = []
    for danau, gdanau in seg.groupby("danau"):
        n_before = gdanau["tanggal"].nunique()
        kept_dates = 0
        for tanggal, g in gdanau.groupby("tanggal"):
            h = g["ht_ortho"].to_numpy(dtype=float)
            mask = nmad_filter(h, factor)
            h2 = h[mask]
            if h2.size < min_n:
                continue
            std = float(np.std(h2, ddof=1)) if h2.size > 1 else 0.0
            if std > max_std:
                continue
            kept_dates += 1
            rows.append(
                {
                    "danau": danau,
                    "tanggal": tanggal,
                    "n_seg": int(h2.size),
                    "mean": float(np.mean(h2)),
                    "std": std,
                    "n_seg_before_nmad": int(h.size),
                }
            )
        stats[danau] = {"n_dates_before": int(n_before), "n_dates_after": int(kept_dates)}
    return pd.DataFrame(rows), stats


def note_earthaccess() -> str:
    return (
        "ATL13 downloads (if any) must use earthaccess authenticated via the user's "
        ".netrc; this package never embeds credentials."
    )
