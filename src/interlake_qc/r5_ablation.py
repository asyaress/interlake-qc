"""Ablasi QC V0–V6 + metrik tanpa skor D2."""

from __future__ import annotations

import numpy as np
import pandas as pd

from interlake_qc.baseline_temporal import _temporal_residuals_series, _to_utc_naive
from interlake_qc.d2_interlake import mad


def wse_valid(df: pd.DataFrame, fill: float = -999.0) -> pd.Series:
    w = pd.to_numeric(df["wse"], errors="coerce")
    return w.notna() & (w > fill)


def mask_ketat(df: pd.DataFrame, cfg_ketat: dict, fill: float = -999.0) -> pd.Series:
    return (
        wse_valid(df, fill)
        & (pd.to_numeric(df["quality_f"], errors="coerce") == cfg_ketat["quality_f"])
        & (pd.to_numeric(df["partial_f"], errors="coerce") == cfg_ketat["partial_f"])
        & (pd.to_numeric(df["dark_frac"], errors="coerce") < cfg_ketat["dark_frac_max"])
    )


def mask_q01(df: pd.DataFrame, fill: float = -999.0) -> pd.Series:
    return wse_valid(df, fill) & pd.to_numeric(df["quality_f"], errors="coerce").isin([0, 1])


def _near(t, timeset, sec: float = 120.0) -> bool:
    t = pd.Timestamp(t)
    return any(abs((ft - t).total_seconds()) <= sec for ft in timeset)


def build_variant_masks(
    swot: pd.DataFrame,
    d2_final: pd.DataFrame,
    btemp: pd.DataFrame,
    cfg: dict,
) -> dict[str, pd.Series]:
    """Mask boolean indexed seperti swot untuk V0–V6."""
    fill = float(cfg["quality"]["fill_value"])
    ketat = cfg["ketat"]
    k_bt = float(cfg["baseline_temporal"]["k_report"])
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]

    sw = swot.copy()
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    d2 = d2_final.copy()
    d2["time"] = pd.to_datetime(d2["time"], utc=True)
    flagged = d2[d2["flag"].astype(bool)]
    flagged_times = set(pd.Timestamp(t) for t in flagged["time"])
    src_map = {pd.Timestamp(r["time"]): str(r.get("sumber_akhir", "ambigu")) for _, r in flagged.iterrows()}

    bt = btemp.copy()
    bt["time"] = pd.to_datetime(bt["time"], utc=True)
    fcol = f"flag_k{k_bt}"
    if fcol not in bt.columns:
        for c in bt.columns:
            if str(c).startswith("flag_k"):
                try:
                    if abs(float(str(c).replace("flag_k", "")) - k_bt) < 1e-9:
                        fcol = c
                        break
                except ValueError:
                    continue
    bt_flag = {
        name: set(
            pd.Timestamp(t)
            for t in bt.loc[(bt["danau"] == name) & bt[fcol].astype(bool), "time"]
        )
        for name in (danau_a, danau_b)
    }

    def sumber_at(t) -> str:
        t = pd.Timestamp(t)
        for ft, s in src_map.items():
            if abs((ft - t).total_seconds()) <= 120:
                return s
        return "ambigu"

    v0 = wse_valid(sw, fill)
    v1 = mask_q01(sw, fill)
    v2 = mask_ketat(sw, ketat, fill)
    v3 = v1.copy()
    v4 = v1.copy()
    v5 = v1.copy()
    v6 = v2.copy()

    for i in sw.index:
        t = sw.loc[i, "time"]
        danau = sw.loc[i, "danau"]
        if danau not in (danau_a, danau_b):
            continue
        if _near(t, flagged_times):
            v3.loc[i] = False
            v6.loc[i] = False
            src = sumber_at(t)
            if src == danau or src in ("ambigu", "tak bisa diatribusi", ""):
                v4.loc[i] = False
            # sumber = danau lain → pertahankan obs danau ini
        if _near(t, bt_flag.get(danau, set())):
            v5.loc[i] = False

    return {
        "V0_nofilter": v0,
        "V1_q01_SQRTL": v1,
        "V2_ketat": v2,
        "V3_V1_D2_both": v3,
        "V4_V1_D2_attr": v4,
        "V5_V1_Btemp": v5,
        "V6_V2_D2_both": v6,
    }


def atl13_swot_metrics(
    swot_sub: pd.DataFrame,
    atl: pd.DataFrame,
    danau: str,
    max_days: float = 3.0,
) -> dict:
    a = atl[atl["danau"] == danau].copy()
    if a.empty or swot_sub.empty:
        return {"n_pasangan": 0, "median_diff_m": "", "mad_diff_m": "", "rmse_m": ""}
    a["tanggal"] = pd.to_datetime(a["tanggal"], utc=True)
    s = swot_sub.copy()
    s["time"] = pd.to_datetime(s["time"], utc=True)
    diffs = []
    for _, ar in a.iterrows():
        dt = (s["time"] - ar["tanggal"]).abs()
        j = dt.idxmin()
        if dt.loc[j] <= pd.Timedelta(days=max_days):
            diffs.append(float(ar["mean"]) - float(s.loc[j, "wse"]))
    if not diffs:
        return {"n_pasangan": 0, "median_diff_m": "", "mad_diff_m": "", "rmse_m": ""}
    arr = np.asarray(diffs, dtype=float)
    return {
        "n_pasangan": int(len(arr)),
        "median_diff_m": float(np.median(arr)),
        "mad_diff_m": float(mad(arr)),
        "rmse_m": float(np.sqrt(np.mean(arr**2))),
    }


def interlake_atl13_metric(
    swot: pd.DataFrame,
    mask: pd.Series,
    atl_i: pd.DataFrame,
    danau_a: str,
    danau_b: str,
    max_days: float = 10.0,
    pair_name: str = "semayang_minus_melintang",
) -> dict:
    ai = atl_i.copy()
    if "pair" in ai.columns:
        ai = ai[ai["pair"] == pair_name]
    if ai.empty:
        return {"n_interlake": 0, "median_abs_dWSE_minus_dH": ""}
    if "tanggal" in ai.columns:
        ai["t"] = pd.to_datetime(ai["tanggal"], utc=True)
    else:
        ai["t"] = pd.to_datetime(ai.get("time"), utc=True)
    dH_col = "dH_ATL13" if "dH_ATL13" in ai.columns else "dH"

    sa = swot.loc[mask & (swot["danau"] == danau_a)].copy()
    sb = swot.loc[mask & (swot["danau"] == danau_b)].copy()
    sa["time"] = pd.to_datetime(sa["time"], utc=True)
    sb["time"] = pd.to_datetime(sb["time"], utc=True)
    diffs = []
    for _, ar in ai.iterrows():
        t_atl = pd.Timestamp(ar["t"])
        dH = float(ar[dH_col])
        best = None
        best_dt = None
        for _, ra in sa.iterrows():
            dt_a = abs((ra["time"] - t_atl).total_seconds()) / 86400.0
            if dt_a > max_days:
                continue
            if sb.empty:
                continue
            dtb = (sb["time"] - ra["time"]).abs()
            jb = dtb.idxmin()
            if dtb.loc[jb] > pd.Timedelta(seconds=120):
                continue
            dwse = float(ra["wse"]) - float(sb.loc[jb, "wse"])
            if best_dt is None or dt_a < best_dt:
                best_dt = dt_a
                best = abs(dwse - dH)
        if best is not None:
            diffs.append(best)
    if not diffs:
        return {"n_interlake": 0, "median_abs_dWSE_minus_dH": ""}
    return {
        "n_interlake": int(len(diffs)),
        "median_abs_dWSE_minus_dH": float(np.median(diffs)),
    }


def smoothness_metrics(
    swot_sub: pd.DataFrame,
    neighbor_max_days: float = 30.0,
    mad_scale: float = 1.4826,
) -> dict:
    if swot_sub.empty or len(swot_sub) < 3:
        return {"smooth_mad_scaled": "", "smooth_p95_abs_e": ""}
    d = swot_sub.sort_values("time")
    t_arr = _to_utc_naive(d["time"])
    w_arr = pd.to_numeric(d["wse"], errors="coerce").to_numpy(dtype=float)
    e = _temporal_residuals_series(t_arr, w_arr, neighbor_max_days * 86400.0)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"smooth_mad_scaled": "", "smooth_p95_abs_e": ""}
    return {
        "smooth_mad_scaled": float(mad_scale * mad(e)),
        "smooth_p95_abs_e": float(np.percentile(np.abs(e), 95)),
    }


def run_ablation_table(
    swot: pd.DataFrame,
    masks: dict[str, pd.Series],
    atl: pd.DataFrame,
    atl_i: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    lakes = [cfg["pair"]["danau_a"], cfg["pair"]["danau_b"]]
    rows = []
    for vname, mask in masks.items():
        inter = interlake_atl13_metric(
            swot,
            mask,
            atl_i,
            cfg["pair"]["danau_a"],
            cfg["pair"]["danau_b"],
            float(cfg["coincidence"]["atl13_interlake_days"]),
            cfg["pair"]["name"],
        )
        for danau in lakes:
            sub = swot.loc[mask & (swot["danau"] == danau)].copy()
            m_atl = atl13_swot_metrics(
                sub, atl, danau, float(cfg["coincidence"]["atl13_swot_days"])
            )
            sm = smoothness_metrics(
                sub,
                float(cfg["smoothness"]["neighbor_max_days"]),
                float(cfg["smoothness"]["mad_scale"]),
            )
            rows.append(
                {
                    "variant": vname,
                    "danau": danau,
                    "n_kept": int(len(sub)),
                    **m_atl,
                    **sm,
                    **inter,
                }
            )
    return pd.DataFrame(rows)
