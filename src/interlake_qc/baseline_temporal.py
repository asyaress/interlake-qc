"""B-temp — pembanding QC per-danau (satu deret waktu, tanpa tetangga antar-danau)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from interlake_qc.core import mad


def _to_utc_naive(times) -> np.ndarray:
    s = pd.to_datetime(pd.Series(times), utc=True)
    return s.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _temporal_residuals_series(
    times: np.ndarray,
    wse: np.ndarray,
    neighbor_max_sec: float,
) -> np.ndarray:
    """Residu temporal untuk seluruh deret terurut (NaN bila tetangga tidak lengkap)."""
    n = len(times)
    e = np.full(n, np.nan)
    t = times.astype("datetime64[ns]").astype(np.int64) / 1e9
    w = np.asarray(wse, dtype=float)
    max_dt = neighbor_max_sec
    for i in range(n):
        ti, wi = t[i], w[i]
        if not np.isfinite(wi):
            continue
        j_b = -1
        for j in range(i - 1, -1, -1):
            dt = ti - t[j]
            if dt > max_dt:
                break
            if np.isfinite(w[j]):
                j_b = j
                break
        j_a = -1
        for j in range(i + 1, n):
            dt = t[j] - ti
            if dt > max_dt:
                break
            if np.isfinite(w[j]):
                j_a = j
                break
        if j_b < 0 or j_a < 0:
            continue
        span = t[j_a] - t[j_b]
        if span <= 0:
            continue
        w_hat = w[j_b] + (w[j_a] - w[j_b]) * ((ti - t[j_b]) / span)
        e[i] = wi - w_hat
    return e


def run_btemp(
    lake_df: pd.DataFrame,
    cfg_bt: dict,
    k_values: list[float] | None = None,
    danau: str = "",
) -> pd.DataFrame:
    """Tandai observasi bila |e_lake| > k × σ_lake."""
    k_values = k_values or list(cfg_bt["k_values"])
    q_ok = set(cfg_bt.get("quality_for_neighbors", [0, 1]))
    max_days = float(cfg_bt.get("neighbor_max_days", 30))
    mad_scale = float(cfg_bt.get("mad_scale", 1.4826))
    sigma_min = float(cfg_bt.get("sigma_min_m", 0.02))
    max_sec = max_days * 86400.0

    d = lake_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d["wse"] = pd.to_numeric(d["wse"], errors="coerce")
    d["quality_f"] = pd.to_numeric(d["quality_f"], errors="coerce")
    d = d.sort_values("time").reset_index(drop=True)

    # Satu deret: tetangga hanya q_ok, tetapi semua titik dievaluasi.
    # Bangun deret gabungan: semua titik, tapi saat residual, non-q_ok tidak jadi tetangga
    # Implementasi praktis: hitung e pada q_ok; untuk non-q_ok sisipkan ke salinan q_ok.
    d_nb = d[d["quality_f"].isin(q_ok)].reset_index(drop=True)
    if len(d_nb) == 0:
        return pd.DataFrame()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_nb = _to_utc_naive(d_nb["time"])
        e_nb = _temporal_residuals_series(t_nb, d_nb["wse"].to_numpy(dtype=float), max_sec)

    es_for_sigma = e_nb[np.isfinite(e_nb)]
    sig = mad_scale * mad(es_for_sigma) if es_for_sigma.size else sigma_min
    if not np.isfinite(sig) or sig < sigma_min:
        sig = sigma_min

    # kunci detik UTC
    e_map = {
        int(pd.Timestamp(t).timestamp()): float(e)
        for t, e in zip(d_nb["time"], e_nb)
    }

    rows = []
    for i in range(len(d)):
        t_i = pd.Timestamp(d.loc[i, "time"])
        key = int(t_i.timestamp())
        w_i = float(d.loc[i, "wse"]) if np.isfinite(d.loc[i, "wse"]) else float("nan")
        q_i = d.loc[i, "quality_f"]
        if key in e_map:
            e = e_map[key]
        else:
            # sisip target ke deret q_ok
            times = list(d_nb["time"]) + [t_i]
            wses = list(d_nb["wse"].astype(float)) + [w_i]
            order = np.argsort([pd.Timestamp(x).timestamp() for x in times])
            t_arr = _to_utc_naive([times[j] for j in order])
            w_arr = np.asarray([wses[j] for j in order], dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e_all = _temporal_residuals_series(t_arr, w_arr, max_sec)
            # posisi target
            pos = int(np.where(order == len(times) - 1)[0][0])
            e = float(e_all[pos])
        z = e / sig if np.isfinite(e) and sig > 0 else float("nan")
        rec = {
            "danau": danau or d.loc[i].get("danau", ""),
            "time": t_i,
            "wse": w_i,
            "quality_f": q_i,
            "e_lake": e,
            "sigma_lake": sig,
            "z_temp": z,
        }
        for kk in k_values:
            rec[f"flag_k{kk}"] = bool(np.isfinite(z) and abs(z) > float(kk))
        rows.append(rec)
    return pd.DataFrame(rows)


def run_btemp_all_lakes(
    swot: pd.DataFrame,
    lakes: list[str],
    cfg_bt: dict,
    k_values: list[float] | None = None,
) -> pd.DataFrame:
    frames = []
    for name in lakes:
        sub = swot[swot["danau"] == name]
        if len(sub) == 0:
            continue
        frames.append(run_btemp(sub, cfg_bt, k_values=k_values, danau=name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
