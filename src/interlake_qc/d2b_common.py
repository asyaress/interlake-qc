"""Utilitas bersama R4b."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_d2b_config(repo: Path, path: str | Path = "config/d2b.yaml") -> dict:
    with open(repo / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def truth_curve_table(cfg: dict) -> pd.DataFrame:
    return pd.DataFrame(cfg["truth_curve"]["bins"]).sort_values("center").reset_index(drop=True)


def interp_truth(h: float | np.ndarray, bins: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Interpolasi linear g(H)=f dan σ(H) dari truth_curve."""
    centers = bins["center"].to_numpy(dtype=float)
    fvals = bins["f"].to_numpy(dtype=float)
    svals = bins["sigma"].to_numpy(dtype=float)
    hq = np.atleast_1d(np.asarray(h, dtype=float))
    f = np.interp(hq, centers, fvals)
    s = np.interp(hq, centers, svals)
    return f, s


def band_of(h: float) -> str:
    return "H<2.5" if float(h) < 2.5 else "H>=2.5"


def flag_col(df: pd.DataFrame, k: float) -> str:
    name = f"flag_k{k}"
    if name in df.columns:
        return name
    for c in df.columns:
        if c.startswith("flag_k"):
            try:
                if abs(float(c.replace("flag_k", "")) - float(k)) < 1e-9:
                    return c
            except ValueError:
                continue
    return name


def d2_cfg_from_d2b(cfg: dict, danau_a: str | None = None, danau_b: str | None = None) -> dict:
    """Bangun dict kompatibel run_d2 / prepare_pair_table."""
    return {
        "pair": {
            "danau_a": danau_a or cfg["pair"]["danau_a"],
            "danau_b": danau_b or cfg["pair"]["danau_b"],
            "name": f"{danau_a or cfg['pair']['danau_a']}_minus_{danau_b or cfg['pair']['danau_b']}",
        },
        "quality": cfg["quality"],
        "reference_curve": cfg["reference_curve"],
        "detection": cfg["detection"],
        "attribution": cfg["attribution"],
        "baseline_temporal": cfg["baseline_temporal"],
    }


def is_named_source(sumber: str) -> bool:
    s = str(sumber).strip().lower()
    return s not in ("", "ambigu", "tak bisa diatribusi", "nan", "none")
