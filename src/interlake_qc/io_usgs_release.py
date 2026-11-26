"""USGS SWOT–gage release reader (doi:10.5066/P1FE9W9E). Prefer SWOT version D."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interlake_qc.r7_yahara import d_preferred_else_c, dedup_prefer_D


def norm_gage_id(agency: str, gage_id: str) -> tuple[str, str]:
    """Preserve original string IDs (e.g. ECCC 06AF002 ≠ 06AC002)."""
    return str(agency).strip(), str(gage_id).strip()


def load_usgs_gage_csv(path: Path | str, lake_ids: list[str] | None = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(
        path,
        dtype={"gage_id": str, "lake_id": str, "cycle_id": str, "pass_id": str},
        low_memory=False,
    )
    if lake_ids is not None:
        keep = {str(x) for x in lake_ids}
        df = df[df["lake_id"].astype(str).isin(keep)].copy()
    if "gage_agency" in df.columns and "gage_id" in df.columns:
        keys = [norm_gage_id(a, g) for a, g in zip(df["gage_agency"], df["gage_id"])]
        df["gage_key"] = [f"{a}|{g}" for a, g in keys]
    return df.reset_index(drop=True)


def dedup_versions_prefer_d(gage: pd.DataFrame) -> pd.DataFrame:
    """Dedup then keep D when both C and D exist for the same key."""
    g, _ = dedup_prefer_D(gage)
    return d_preferred_else_c(g)
