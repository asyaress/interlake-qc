"""Atribusi triangulasi memakai Jempang (D2-tri)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from interlake_qc.d2_interlake import prepare_pair_table, run_d2
from interlake_qc.d2b_common import d2_cfg_from_d2b, flag_col, is_named_source


def build_jempang_pair_results(swot: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Jalankan D2 pada Semayang−Jempang dan Melintang−Jempang. Kembalikan map name→hasil."""
    out = {}
    for spec in cfg["triangulation"]["pairs"]:
        d2cfg = d2_cfg_from_d2b(cfg, danau_a=spec["danau_a"], danau_b=spec["danau_b"])
        pairs = prepare_pair_table(swot, d2cfg)
        if pairs.empty:
            out[spec["name"]] = pd.DataFrame()
            continue
        lakes = {
            spec["danau_a"]: swot[swot["danau"] == spec["danau_a"]].copy(),
            spec["danau_b"]: swot[swot["danau"] == spec["danau_b"]].copy(),
        }
        # Jempang must be q∈{0,1} for core — prepare_pair_table already sets core from both q
        # additionally require jempang q in core for scoring curve
        res, _ = run_d2(pairs, lakes, d2cfg, do_attribution=False)
        res["time"] = pd.to_datetime(res["time"], utc=True)
        res["points_to"] = spec["points_to"]
        out[spec["name"]] = res
    return out


def _z_at_time(res: pd.DataFrame, t: pd.Timestamp) -> float:
    if res is None or len(res) == 0:
        return float("nan")
    dt = (res["time"] - t).abs()
    j = int(dt.idxmin())
    if dt.loc[j] > pd.Timedelta(seconds=120):
        return float("nan")
    return float(res.loc[j, "z"])


def attribute_triangulation(
    idx: int,
    pairs_sm: pd.DataFrame,
    d2_sm: pd.DataFrame,
    tri_maps: dict[str, pd.DataFrame],
    cfg: dict,
    k: float,
    temporal_sumber: str = "",
) -> str:
    """Aturan D2-tri untuk satu pass SM yang tertandai.

    Bila tepat satu pasangan-Jempang punya |z| > k_tri → sumber = danau di pasangan itu.
    Bila keduanya atau tidak satu pun → pakai atribusi temporal; bila tetap tak jelas → ambigu.
    """
    k_tri = float(cfg["triangulation"]["k_tri"])
    t = pd.Timestamp(pairs_sm.loc[idx, "time"])
    hits = []
    for spec in cfg["triangulation"]["pairs"]:
        name = spec["name"]
        res = tri_maps.get(name, pd.DataFrame())
        z = _z_at_time(res, t)
        if np.isfinite(z) and abs(z) > k_tri:
            hits.append(spec["points_to"])
    hits = list(dict.fromkeys(hits))  # unique preserve order
    if len(hits) == 1:
        return hits[0]
    # keduanya atau tidak satu pun → temporal
    if is_named_source(temporal_sumber):
        return str(temporal_sumber)
    return "ambigu"


def apply_tri_to_real_flags(
    d2_sm: pd.DataFrame,
    pairs_sm: pd.DataFrame,
    swot: pd.DataFrame,
    cfg: dict,
    k_values: list[float],
) -> pd.DataFrame:
    """Untuk data nyata: isi kolom sumber_tri per k pada pass tertandai."""
    from interlake_qc.d2_interlake import attribute_source

    tri_maps = build_jempang_pair_results(swot, cfg)
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]
    lake_series = {
        danau_a: swot[swot["danau"] == danau_a].copy(),
        danau_b: swot[swot["danau"] == danau_b].copy(),
    }
    attr_cfg = {**cfg["attribution"], "mad_scale": cfg["reference_curve"]["mad_scale"]}
    out = d2_sm.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    pairs = pairs_sm.copy().reset_index(drop=True)
    pairs["time"] = pd.to_datetime(pairs["time"], utc=True)

    for kk in k_values:
        fcol = flag_col(out, kk)
        col_tri = f"sumber_tri_k{kk}"
        col_temp = f"sumber_temp_k{kk}"
        sumber_tri = []
        sumber_temp = []
        for i in range(len(out)):
            if not bool(out.iloc[i][fcol]):
                sumber_tri.append("")
                sumber_temp.append("")
                continue
            t = out.iloc[i]["time"]
            # match idx in pairs
            dt = (pairs["time"] - t).abs()
            j = int(dt.idxmin())
            temp, _ = attribute_source(lake_series, t, attr_cfg)
            tri = attribute_triangulation(j, pairs, out, tri_maps, cfg, kk, temporal_sumber=temp)
            sumber_temp.append(temp)
            sumber_tri.append(tri)
        out[col_temp] = sumber_temp
        out[col_tri] = sumber_tri
    return out


def summarize_real_attribution(d2_with_tri: pd.DataFrame, k_values: list[float]) -> pd.DataFrame:
    rows = []
    for kk in k_values:
        fcol = flag_col(d2_with_tri, kk)
        fl = d2_with_tri[d2_with_tri[fcol]]
        n_flag = len(fl)
        for kind, col in (("temporal", f"sumber_temp_k{kk}"), ("triangulasi", f"sumber_tri_k{kk}")):
            if col not in fl.columns:
                continue
            named = fl[col].map(is_named_source)
            rows.append(
                {
                    "k": kk,
                    "method": kind,
                    "n_flagged": n_flag,
                    "n_named": int(named.sum()),
                    "coverage": float(named.mean()) if n_flag else float("nan"),
                    # precision on real data unknown — leave empty
                    "precision": "",
                }
            )
    return pd.DataFrame(rows)
