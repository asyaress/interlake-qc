"""Cek air rendah: SWOT vs ATL13 + dark_frac + baris 2025-08-05."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interlake_qc.core import interp_f_sigma
from interlake_qc.common import band_of


def _spearmanr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ + p-value dua sisi (aproksimasi t), tanpa SciPy."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 0:
        return float("nan"), float("nan")
    rho = float((rx * ry).sum() / denom)
    # p via t approx
    if abs(rho) >= 1:
        return rho, 0.0
    t = rho * np.sqrt((n - 2) / (1 - rho * rho))
    # regularized incomplete beta approx for p — gunakan erfc untuk normal besar n
    # dua sisi dari normal: z≈t
    from math import erfc, sqrt

    p = float(erfc(abs(t) / sqrt(2.0)))
    return rho, p


def atl13_swot_table(
    atl_i: pd.DataFrame,
    pairs: pd.DataFrame,
    swot: pd.DataFrame,
    window_days: float = 10.0,
    pair_name: str = "semayang_minus_melintang",
) -> pd.DataFrame:
    """Untuk tiap tanggal ATL13 antar-danau, semua pass SWOT q∈{0,1} dalam ±window_days."""
    ai = atl_i.copy()
    if "pair" in ai.columns:
        ai = ai[ai["pair"] == pair_name]
    if "tanggal" in ai.columns:
        ai["t_atl"] = pd.to_datetime(ai["tanggal"], utc=True)
    else:
        ai["t_atl"] = pd.to_datetime(ai.get("time"), utc=True)
    dH_col = "dH_ATL13" if "dH_ATL13" in ai.columns else ("dH" if "dH" in ai.columns else None)
    H_col = "rerata_H" if "rerata_H" in ai.columns else ("H_ATL13" if "H_ATL13" in ai.columns else None)

    p = pairs[pairs["core"]].copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    sw = swot.copy()
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    sw["dark_frac"] = pd.to_numeric(sw.get("dark_frac"), errors="coerce")
    sw["area_total"] = pd.to_numeric(sw.get("area_total"), errors="coerce")

    rows = []
    for _, ar in ai.iterrows():
        t_atl = pd.Timestamp(ar["t_atl"])
        dH = float(ar[dH_col]) if dH_col else float("nan")
        H_atl = float(ar[H_col]) if H_col else float("nan")
        near = p[(p["time"] - t_atl).abs() <= pd.Timedelta(days=window_days)]
        if near.empty:
            rows.append(
                {
                    "tanggal_atl13": t_atl.strftime("%Y-%m-%d"),
                    "dH_ATL13": dH,
                    "H_ATL13": H_atl,
                    "swot_tanggal": "",
                    "dt_days": "",
                    "dWSE": "",
                    "H": "",
                    "dark_frac_semayang": "",
                    "dark_frac_melintang": "",
                    "pass": "",
                }
            )
            continue
        for _, pr in near.iterrows():
            t = pd.Timestamp(pr["time"])
            dt = abs((t - t_atl).total_seconds()) / 86400.0
            # dark_frac from swot per lake
            def _dark(danau: str) -> float:
                sub = sw[(sw["danau"] == danau) & ((sw["time"] - t).abs() <= pd.Timedelta(seconds=120))]
                return float(sub.iloc[0]["dark_frac"]) if len(sub) else float("nan")

            rows.append(
                {
                    "tanggal_atl13": t_atl.strftime("%Y-%m-%d"),
                    "dH_ATL13": dH,
                    "H_ATL13": H_atl,
                    "swot_tanggal": t.strftime("%Y-%m-%d"),
                    "dt_days": dt,
                    "dWSE": float(pr["dWSE"]),
                    "H": float(pr["H"]),
                    "dark_frac_semayang": _dark("semayang"),
                    "dark_frac_melintang": _dark("melintang"),
                    "pass": pr.get("pass", ""),
                }
            )
    return pd.DataFrame(rows)


def summarize_by_h_band(tbl: pd.DataFrame) -> pd.DataFrame:
    """Median (dWSE terdekat − dH_ATL13) per pita H."""
    if tbl.empty:
        return pd.DataFrame()
    t = tbl.copy()
    t["dWSE"] = pd.to_numeric(t["dWSE"], errors="coerce")
    t["dH_ATL13"] = pd.to_numeric(t["dH_ATL13"], errors="coerce")
    t["H"] = pd.to_numeric(t["H"], errors="coerce")
    t["dt_days"] = pd.to_numeric(t["dt_days"], errors="coerce")
    t = t.dropna(subset=["dWSE", "dH_ATL13", "H"])
    rows = []
    for band, sub0 in [
        ("H<2.5", t[t["H"] < 2.5]),
        ("H>=2.5", t[t["H"] >= 2.5]),
    ]:
        # nearest SWOT per ATL13 date
        diffs = []
        for tanggal, g in sub0.groupby("tanggal_atl13"):
            j = g["dt_days"].idxmin()
            diffs.append(float(g.loc[j, "dWSE"]) - float(g.loc[j, "dH_ATL13"]))
        rows.append(
            {
                "h_band": band,
                "n_atl13_dates": len(diffs),
                "median_dWSE_minus_dH": float(np.median(diffs)) if diffs else "",
                "mad": float(np.median(np.abs(np.asarray(diffs) - np.median(diffs)))) if diffs else "",
            }
        )
    return pd.DataFrame(rows)


def spearman_resid_dark(
    pairs: pd.DataFrame,
    d2: pd.DataFrame,
    swot: pd.DataFrame,
) -> pd.DataFrame:
    """Spearman residu D2 (dWSE−f(H)) vs dark_frac Sem/Mel/selisih, per pita H."""
    p = pairs[pairs["core"]].copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    d = d2.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True)
    m = p.merge(d[["time", "f_H", "z", "dWSE"]], on="time", how="inner", suffixes=("", "_d2"))
    m["resid"] = pd.to_numeric(m["dWSE"], errors="coerce") - pd.to_numeric(m["f_H"], errors="coerce")
    sw = swot.copy()
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    sw["dark_frac"] = pd.to_numeric(sw["dark_frac"], errors="coerce")

    def dark_at(t, danau):
        sub = sw[(sw["danau"] == danau) & ((sw["time"] - t).abs() <= pd.Timedelta(seconds=120))]
        return float(sub.iloc[0]["dark_frac"]) if len(sub) else float("nan")

    m["dark_sem"] = [dark_at(t, "semayang") for t in m["time"]]
    m["dark_mel"] = [dark_at(t, "melintang") for t in m["time"]]
    m["dark_diff"] = m["dark_sem"] - m["dark_mel"]
    m["h_band"] = m["H"].map(band_of)

    rows = []
    for band, g in m.groupby("h_band"):
        for name, col in (
            ("dark_frac_semayang", "dark_sem"),
            ("dark_frac_melintang", "dark_mel"),
            ("dark_frac_sem_minus_mel", "dark_diff"),
        ):
            x = pd.to_numeric(g["resid"], errors="coerce")
            y = pd.to_numeric(g[col], errors="coerce")
            ok = x.notna() & y.notna()
            n = int(ok.sum())
            if n >= 3:
                rho, pval = _spearmanr(x[ok].to_numpy(), y[ok].to_numpy())
            else:
                rho, pval = float("nan"), float("nan")
            rows.append({"h_band": band, "variable": name, "n": n, "spearman_rho": rho, "p_value": pval})
    return pd.DataFrame(rows)


def anomaly_20250805(
    pairs: pd.DataFrame,
    d2: pd.DataFrame,
    swot: pd.DataFrame,
    bins: pd.DataFrame,
    date: str = "2025-08-05",
) -> pd.DataFrame:
    p = pairs.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    day = pd.Timestamp(date, tz="UTC")
    sub = p[(p["time"] - day).abs() <= pd.Timedelta(hours=12)]
    if sub.empty:
        # try day match
        sub = p[p["time"].dt.strftime("%Y-%m-%d") == date]
    rows = []
    sw = swot.copy()
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    for _, r in sub.iterrows():
        f_h, sig = interp_f_sigma(float(r["H"]), bins)
        z = (float(r["dWSE"]) - f_h) / sig if sig and np.isfinite(sig) else float("nan")
        # prefer d2 z if available
        d = d2.copy()
        d["time"] = pd.to_datetime(d["time"], utc=True)
        dt = (d["time"] - r["time"]).abs()
        if len(d) and dt.min() <= pd.Timedelta(seconds=120):
            j = int(dt.idxmin())
            f_h = float(d.loc[j, "f_H"])
            sig = float(d.loc[j, "sigma_H"])
            z = float(d.loc[j, "z"])

        def grab(danau):
            s = sw[(sw["danau"] == danau) & ((sw["time"] - r["time"]).abs() <= pd.Timedelta(seconds=120))]
            if not len(s):
                return float("nan"), float("nan")
            return float(s.iloc[0]["dark_frac"]), float(s.iloc[0].get("area_total", float("nan")))

        ds, as_ = grab("semayang")
        dm, am = grab("melintang")
        rows.append(
            {
                "tanggal": date,
                "dWSE": float(r["dWSE"]),
                "H": float(r["H"]),
                "f_H": f_h,
                "sigma_H": sig,
                "z": z,
                "dark_frac_semayang": ds,
                "dark_frac_melintang": dm,
                "area_total_semayang": as_,
                "area_total_melintang": am,
                "pass": r.get("pass", ""),
                "q_a": r.get("q_a", ""),
                "q_b": r.get("q_b", ""),
            }
        )
    return pd.DataFrame(rows)


def table_d_consistency(repo: Path, cfg: dict) -> pd.DataFrame:
    """6 baris beda file vs Hydrocron + dWSE file vs HC."""
    bad = pd.read_csv(repo / cfg["paths"]["r3_hydrocron_consistency_bad"])
    rows = []
    for tanggal in ("2025-05-08", "2025-05-13"):
        sub = bad[bad["time_str"].astype(str).str.contains(tanggal)]
        for _, r in sub.iterrows():
            rows.append(
                {
                    "tanggal": tanggal,
                    "danau": r.get("danau", r.get("danau_hc", "")),
                    "granule_name": r.get("granule_name", ""),
                    "crid": r.get("crid", ""),
                    "source_zip": r.get("source_zip", ""),
                    "wse_file": r.get("wse", ""),
                    "wse_hydrocron": r.get("wse_hc", ""),
                    "delta_wse_file_minus_hc": r.get("dwse", ""),
                    "quality_f_file": r.get("quality_f", ""),
                    "quality_f_hydrocron": r.get("quality_f_hc", ""),
                    "q_diff": r.get("q_diff", ""),
                    "dWSE_file": "",
                    "dWSE_hydrocron": "",
                    "dWSE_file_minus_hc": "",
                }
            )
        sem = sub[sub["danau"] == "semayang"]
        mel = sub[sub["danau"] == "melintang"]
        if len(sem) and len(mel):
            dwse_f = float(sem.iloc[0]["wse"]) - float(mel.iloc[0]["wse"])
            dwse_h = float(sem.iloc[0]["wse_hc"]) - float(mel.iloc[0]["wse_hc"])
            rows.append(
                {
                    "tanggal": tanggal,
                    "danau": "PAIR_sem_minus_mel",
                    "granule_name": sem.iloc[0].get("granule_name", ""),
                    "crid": sem.iloc[0].get("crid", ""),
                    "source_zip": sem.iloc[0].get("source_zip", ""),
                    "wse_file": "",
                    "wse_hydrocron": "",
                    "delta_wse_file_minus_hc": "",
                    "quality_f_file": "",
                    "quality_f_hydrocron": "",
                    "q_diff": "",
                    "dWSE_file": dwse_f,
                    "dWSE_hydrocron": dwse_h,
                    "dWSE_file_minus_hc": dwse_f - dwse_h,
                }
            )
    return pd.DataFrame(rows)
