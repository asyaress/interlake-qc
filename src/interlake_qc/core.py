"""D2 v1 — deteksi pass janggal antar-danau + atribusi (leave-one-out)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_d2_config(repo: Path, path: str | Path = "configs/mahakam.yaml") -> dict:
    with open(repo / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def k_for_H(
    H: float,
    h_threshold: float = 2.5,
    k_high: float = 3.5,
    k_low: float = 4.0,
) -> float:
    """Ambang k per pita H: H >= threshold → k_high; H < threshold → k_low."""
    if not np.isfinite(H):
        return float(k_high)
    return float(k_high) if float(H) >= float(h_threshold) else float(k_low)


def k_for_H_from_cfg(H: float, cfg: dict) -> float:
    d = cfg.get("detection_final") or {}
    return k_for_H(
        H,
        h_threshold=float(d.get("h_threshold_m", 2.5)),
        k_high=float(d.get("k_high", 3.5)),
        k_low=float(d.get("k_low", 4.0)),
    )


def mad(x: pd.Series | np.ndarray | list) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.median(np.abs(arr - np.median(arr))))


def _bin_edges(h: np.ndarray, width: float) -> np.ndarray:
    if h.size == 0:
        return np.array([0.0, width])
    lo = np.floor(float(np.nanmin(h)) / width) * width
    hi = np.ceil(float(np.nanmax(h)) / width) * width
    if hi <= lo:
        hi = lo + width
    return np.arange(lo, hi + width * 0.5, width)


def merge_small_bins(
    edges: np.ndarray, counts: np.ndarray, min_n: int
) -> list[tuple[float, float]]:
    """Gabung bin dengan n < min_n ke tetangga terdekat sampai n ≥ min_n."""
    bins = [(float(edges[i]), float(edges[i + 1]), int(counts[i])) for i in range(len(counts))]
    if not bins:
        return []
    changed = True
    while changed:
        changed = False
        for i, (lo, hi, n) in enumerate(bins):
            if n >= min_n or len(bins) == 1:
                continue
            # tetangga terdekat: kiri atau kanan (pusat)
            center = 0.5 * (lo + hi)
            candidates = []
            if i > 0:
                candidates.append((i - 1, abs(center - 0.5 * (bins[i - 1][0] + bins[i - 1][1]))))
            if i < len(bins) - 1:
                candidates.append((i + 1, abs(center - 0.5 * (bins[i + 1][0] + bins[i + 1][1]))))
            j = min(candidates, key=lambda t: t[1])[0]
            a, b = sorted([i, j])
            lo2 = bins[a][0]
            hi2 = bins[b][1]
            n2 = bins[a][2] + bins[b][2]
            bins = bins[:a] + [(lo2, hi2, n2)] + bins[b + 1 :]
            changed = True
            break
    return [(lo, hi) for lo, hi, _ in bins]


def build_reference_bins(
    h: np.ndarray,
    dwse: np.ndarray,
    bin_width: float = 0.25,
    min_n: int = 5,
    mad_scale: float = 1.4826,
    sigma_min: float = 0.02,
) -> pd.DataFrame:
    """Bangun tabel bin: pusat, n, f=median dWSE, σ=1.4826×MAD residu (≥ sigma_min)."""
    mask = np.isfinite(h) & np.isfinite(dwse)
    h = np.asarray(h, dtype=float)[mask]
    dwse = np.asarray(dwse, dtype=float)[mask]
    if h.size == 0:
        return pd.DataFrame(columns=["bin_lo", "bin_hi", "center", "n", "f", "sigma"])

    edges = _bin_edges(h, bin_width)
    counts = []
    for i in range(len(edges) - 1):
        if i < len(edges) - 2:
            m = (h >= edges[i]) & (h < edges[i + 1])
        else:
            m = (h >= edges[i]) & (h <= edges[i + 1])
        counts.append(int(m.sum()))
    counts = np.asarray(counts, dtype=int)
    merged = merge_small_bins(edges, counts, min_n)

    rows = []
    for lo, hi in merged:
        if hi == merged[-1][1]:
            m = (h >= lo) & (h <= hi)
        else:
            m = (h >= lo) & (h < hi)
        # avoid double-count on shared edges for intermediate: use < hi except last
        yy = dwse[m]
        if yy.size == 0:
            continue
        f = float(np.median(yy))
        resid = yy - f
        sig = mad_scale * mad(resid)
        if not np.isfinite(sig) or sig < sigma_min:
            sig = sigma_min
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "center": 0.5 * (lo + hi),
                "n": int(yy.size),
                "f": f,
                "sigma": float(sig),
            }
        )
    return pd.DataFrame(rows)


def interp_f_sigma(h_query: float, bins: pd.DataFrame) -> tuple[float, float]:
    """Interpolasi linear f(H) dan σ(H) antar pusat bin."""
    if bins is None or len(bins) == 0 or not np.isfinite(h_query):
        return float("nan"), float("nan")
    b = bins.sort_values("center")
    centers = b["center"].to_numpy(dtype=float)
    fvals = b["f"].to_numpy(dtype=float)
    svals = b["sigma"].to_numpy(dtype=float)
    if centers.size == 1:
        return float(fvals[0]), float(svals[0])
    f = float(np.interp(h_query, centers, fvals))
    s = float(np.interp(h_query, centers, svals))
    return f, s


def temporal_residual(
    times: pd.Series,
    wse: pd.Series,
    idx: int,
    neighbor_max_days: float = 30.0,
) -> float:
    """e = wse − interpolasi linear antara tetangga sebelum & sesudah (≤ neighbor_max_days).

    Mengembalikan NaN bila tetangga tidak lengkap.
    """
    t = pd.to_datetime(times, utc=True).reset_index(drop=True)
    w = pd.to_numeric(wse, errors="coerce").reset_index(drop=True)
    order = np.argsort(t.to_numpy())
    # map original idx -> position in sorted
    # assume caller passes aligned series; idx is position in this series
    ti = t.iloc[idx]
    wi = w.iloc[idx]
    if pd.isna(ti) or pd.isna(wi):
        return float("nan")
    max_dt = pd.Timedelta(days=neighbor_max_days)
    before = [(t.iloc[j], w.iloc[j]) for j in range(len(t)) if j != idx and t.iloc[j] < ti and (ti - t.iloc[j]) <= max_dt and pd.notna(w.iloc[j])]
    after = [(t.iloc[j], w.iloc[j]) for j in range(len(t)) if j != idx and t.iloc[j] > ti and (t.iloc[j] - ti) <= max_dt and pd.notna(w.iloc[j])]
    if not before or not after:
        return float("nan")
    tb, wb = max(before, key=lambda x: x[0])
    ta, wa = min(after, key=lambda x: x[0])
    span = (ta - tb).total_seconds()
    if span <= 0:
        return float("nan")
    w_hat = float(wb) + (float(wa) - float(wb)) * ((ti - tb).total_seconds() / span)
    return float(wi) - w_hat


def attribute_source(
    lake_series: dict[str, pd.DataFrame],
    time_i: pd.Timestamp,
    cfg_attr: dict,
    sigma_cache: dict[str, float] | None = None,
) -> tuple[str, dict[str, float]]:
    """Atribusi sumber untuk satu pass (waktu).

    lake_series[name] harus punya kolom time, wse, quality_f (seluruh deret danau).
    sigma_cache opsional: σ_lake yang sudah dihitung (hindari MAD ulang).
    """
    from interlake_qc.baseline_temporal import _temporal_residuals_series, _to_utc_naive

    q_ok = set(cfg_attr.get("quality_for_neighbors", [0, 1]))
    max_days = float(cfg_attr.get("neighbor_max_days", 30))
    ratio_thr = float(cfg_attr.get("ambiguity_ratio", 1.5))
    mad_scale = float(cfg_attr.get("mad_scale", 1.4826)) if "mad_scale" in cfg_attr else 1.4826
    max_sec = max_days * 86400.0
    scores: dict[str, float] = {}
    details: dict[str, float] = {}
    time_i = pd.Timestamp(time_i)
    if time_i.tzinfo is None:
        time_i = time_i.tz_localize("UTC")

    for name, df in lake_series.items():
        d = df.copy()
        d["time"] = pd.to_datetime(d["time"], utc=True)
        d["wse"] = pd.to_numeric(d["wse"], errors="coerce")
        d["quality_f"] = pd.to_numeric(d["quality_f"], errors="coerce")
        d_nb = d[d["quality_f"].isin(q_ok)].sort_values("time").reset_index(drop=True)
        d_all = d.sort_values("time").reset_index(drop=True)
        if d_all.empty:
            return "tak bisa diatribusi", {}
        diffs = (d_all["time"] - time_i).abs()
        i_all = int(diffs.idxmin())
        target_row = d_all.iloc[i_all]
        t_target = pd.Timestamp(target_row["time"])
        w_target = float(target_row["wse"])

        times_list = list(d_nb["time"])
        wse_list = [float(x) for x in d_nb["wse"]]
        if not any(abs((pd.Timestamp(tt) - t_target).total_seconds()) < 1 for tt in times_list):
            times_list.append(t_target)
            wse_list.append(w_target)
        order = np.argsort([pd.Timestamp(x).timestamp() for x in times_list])
        t_arr = _to_utc_naive([times_list[j] for j in order])
        w_arr = np.asarray([wse_list[j] for j in order], dtype=float)
        e_all = _temporal_residuals_series(t_arr, w_arr, max_sec)
        # posisi target
        tkeys = [int(pd.Timestamp(times_list[j]).timestamp()) for j in order]
        try:
            pos = tkeys.index(int(t_target.timestamp()))
        except ValueError:
            pos = int(np.argmin(np.abs(np.asarray(tkeys) - int(t_target.timestamp()))))
        e = float(e_all[pos])

        if sigma_cache is not None and name in sigma_cache:
            sig = float(sigma_cache[name])
        else:
            if len(d_nb):
                e_nb = _temporal_residuals_series(
                    _to_utc_naive(d_nb["time"]),
                    d_nb["wse"].to_numpy(dtype=float),
                    max_sec,
                )
                es = e_nb[np.isfinite(e_nb)]
                sig = mad_scale * mad(es) if es.size else 0.02
            else:
                sig = 0.02
            if not np.isfinite(sig) or sig < 0.02:
                sig = 0.02

        details[f"e_{name}"] = e
        details[f"sigma_{name}"] = sig
        scores[name] = float("nan") if not np.isfinite(e) else abs(e) / sig

    if any(not np.isfinite(v) for v in scores.values()) or len(scores) < 2:
        return "tak bisa diatribusi", details
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s_best = ordered[0]
    _, s_second = ordered[1]
    if s_second <= 0:
        return best, details
    if s_best / s_second < ratio_thr:
        return "ambigu", details
    return best, details


def prepare_pair_table(
    swot: pd.DataFrame,
    cfg: dict,
    phase_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Bangun tabel pass Semayang−Melintang dari SWOT per-obs."""
    a = cfg["pair"]["danau_a"]
    b = cfg["pair"]["danau_b"]
    s = swot.copy()
    s["time"] = pd.to_datetime(s["time"], utc=True, errors="coerce")
    s["wse"] = pd.to_numeric(s["wse"], errors="coerce")
    s["quality_f"] = pd.to_numeric(s["quality_f"], errors="coerce")
    if "pass" in s.columns:
        s["pass"] = s["pass"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        s.loc[s["pass"].isin(["nan", "None", ""]), "pass"] = ""
    else:
        s["pass"] = ""

    sa = s[s["danau"] == a][["time", "wse", "quality_f", "pass"]].rename(
        columns={"wse": "wse_a", "quality_f": "q_a", "pass": "pass_a"}
    )
    sb = s[s["danau"] == b][["time", "wse", "quality_f", "pass"]].rename(
        columns={"wse": "wse_b", "quality_f": "q_b", "pass": "pass_b"}
    )
    # match within 120 s
    rows = []
    sb = sb.sort_values("time")
    for _, ra in sa.iterrows():
        if pd.isna(ra["time"]):
            continue
        dt = (sb["time"] - ra["time"]).abs()
        j = dt.idxmin() if len(dt) else None
        if j is None or dt.loc[j] > pd.Timedelta(seconds=120):
            continue
        rb = sb.loc[j]
        # same pass if both known; else allow empty
        pa, pb = str(ra["pass_a"]), str(rb["pass_b"])
        if pa and pb and pa != pb:
            continue
        pass_id = pa or pb or ""
        rows.append(
            {
                "time": ra["time"],
                "pass": pass_id,
                "wse_a": float(ra["wse_a"]),
                "wse_b": float(rb["wse_b"]),
                "q_a": ra["q_a"],
                "q_b": rb["q_b"],
                "dWSE": float(ra["wse_a"]) - float(rb["wse_b"]),
                "H": 0.5 * (float(ra["wse_a"]) + float(rb["wse_b"])),
            }
        )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs["day"] = pairs["time"].dt.strftime("%Y-%m-%d")
    pairs["core"] = pairs["q_a"].isin(cfg["quality"]["core"]) & pairs["q_b"].isin(cfg["quality"]["core"])
    pairs["sudah_ditandai_flag"] = pairs["q_a"].isin(cfg["quality"]["flagged"]) | pairs["q_b"].isin(
        cfg["quality"]["flagged"]
    )
    if phase_df is not None and len(phase_df):
        ph = phase_df.copy()
        if "day" not in ph.columns and "time" in ph.columns:
            ph["day"] = pd.to_datetime(ph["time"], utc=True).dt.strftime("%Y-%m-%d")
        if "phase" in ph.columns:
            pairs = pairs.merge(ph[["day", "phase"]].drop_duplicates("day"), on="day", how="left")
    if "phase" not in pairs.columns:
        pairs["phase"] = ""
    return pairs.sort_values("time").reset_index(drop=True)


def run_d2(
    pairs: pd.DataFrame,
    lake_series: dict[str, pd.DataFrame],
    cfg: dict,
    k_values: list[float] | None = None,
    do_attribution: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan D2 leave-one-out untuk semua k. Kembalikan (hasil per pass, tabel bin global info)."""
    k_values = k_values or list(cfg["detection"]["k_values"])
    rc = cfg["reference_curve"]
    attr_cfg = {
        **cfg["attribution"],
        "mad_scale": rc["mad_scale"],
    }

    # global bins for reporting (all core points)
    core = pairs[pairs["core"]].copy()
    bins_global = build_reference_bins(
        core["H"].to_numpy(),
        core["dWSE"].to_numpy(),
        bin_width=float(rc["bin_width_m"]),
        min_n=int(rc["min_n_per_bin"]),
        mad_scale=float(rc["mad_scale"]),
        sigma_min=float(rc["sigma_min_m"]),
    )

    out_rows = []
    for i in range(len(pairs)):
        row = pairs.iloc[i]
        # leave-one-out reference from other CORE points only
        others = core.index[core.index != pairs.index[i]]
        # if current is not in core index alignment — use boolean mask
        mask_others = pairs["core"].copy()
        mask_others.iloc[i] = False
        h_train = pairs.loc[mask_others, "H"].to_numpy()
        d_train = pairs.loc[mask_others, "dWSE"].to_numpy()
        bins_i = build_reference_bins(
            h_train,
            d_train,
            bin_width=float(rc["bin_width_m"]),
            min_n=int(rc["min_n_per_bin"]),
            mad_scale=float(rc["mad_scale"]),
            sigma_min=float(rc["sigma_min_m"]),
        )
        f_h, sig_h = interp_f_sigma(float(row["H"]), bins_i)
        if not np.isfinite(sig_h) or sig_h < float(rc["sigma_min_m"]):
            sig_h = float(rc["sigma_min_m"])
        z = (float(row["dWSE"]) - f_h) / sig_h if np.isfinite(f_h) and sig_h > 0 else float("nan")

        flags = {kk: bool(np.isfinite(z) and abs(z) > float(kk)) for kk in k_values}
        sumber = ""
        attr_detail: dict[str, float] = {}
        if do_attribution and any(flags.values()):
            sumber, attr_detail = attribute_source(
                lake_series,
                pd.Timestamp(row["time"]),
                attr_cfg,
            )

        rec: dict[str, Any] = {
            "time": row["time"],
            "day": row["day"],
            "pass": row["pass"],
            "H": row["H"],
            "dWSE": row["dWSE"],
            "f_H": f_h,
            "sigma_H": sig_h,
            "z": z,
            "q_a": row["q_a"],
            "q_b": row["q_b"],
            "core": bool(row["core"]),
            "sudah_ditandai_flag": bool(row["sudah_ditandai_flag"]),
            "fase": row.get("phase", ""),
            "sumber": sumber,
        }
        for kk in k_values:
            rec[f"flag_k{kk}"] = flags[kk]
        rec.update(attr_detail)
        out_rows.append(rec)

    return pd.DataFrame(out_rows), bins_global


def run_d2_final(
    pairs: pd.DataFrame,
    lake_series: dict[str, pd.DataFrame],
    cfg: dict,
    do_attribution: bool = True,
    swot: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D2 dengan k per pita H (detection_final).

    Kolom utama: k_dipakai, flag, z, f_H, sigma_H, sumber_tri, sumber_temporal,
    sumber_akhir, cara_atribusi. Jika swot diberikan dan triangulation ada di cfg,
    atribusi = triangulasi dulu lalu temporal.
    """
    d = cfg.get("detection_final") or {}
    h_thr = float(d.get("h_threshold_m", 2.5))
    k_high = float(d.get("k_high", 3.5))
    k_low = float(d.get("k_low", 4.0))
    res, bins = run_d2(
        pairs,
        lake_series,
        cfg,
        k_values=[k_low, k_high],
        do_attribution=do_attribution,
    )
    res = res.copy()
    res["k_dipakai"] = [
        k_for_H(float(row["H"]), h_thr, k_high, k_low) for _, row in res.iterrows()
    ]
    res["flag"] = [
        bool(np.isfinite(float(z)) and abs(float(z)) > float(kk))
        for z, kk in zip(res["z"], res["k_dipakai"])
    ]

    # default temporal dari run_d2
    res["sumber_temporal"] = [
        str(s) if fl and str(s) not in ("", "nan", "None") else ("" if not fl else str(s))
        for fl, s in zip(res["flag"], res.get("sumber", pd.Series([""] * len(res))))
    ]
    # override temporal kosong pada flag
    if do_attribution:
        attr_cfg = {**cfg["attribution"], "mad_scale": cfg["reference_curve"]["mad_scale"]}
        temps = []
        for i in range(len(res)):
            if not bool(res.iloc[i]["flag"]):
                temps.append("")
                continue
            src = str(res.iloc[i].get("sumber", "") or "")
            if src in ("", "nan", "None"):
                src, _ = attribute_source(
                    lake_series, pd.Timestamp(res.iloc[i]["time"]), attr_cfg
                )
            temps.append(src)
        res["sumber_temporal"] = temps

    # triangulasi bila swot tersedia
    sumber_tri: list[str] = []
    sumber_akhir: list[str] = []
    cara: list[str] = []
    if do_attribution and swot is not None and "triangulation" in cfg:
        from interlake_qc.common import is_named_source
        from interlake_qc.triangulation import attribute_triangulation, build_jempang_pair_results

        tri_maps = build_jempang_pair_results(swot, cfg)
        pairs_r = pairs.reset_index(drop=True).copy()
        pairs_r["time"] = pd.to_datetime(pairs_r["time"], utc=True)
        res_t = pd.to_datetime(res["time"], utc=True)
        for i in range(len(res)):
            if not bool(res.iloc[i]["flag"]):
                sumber_tri.append("")
                sumber_akhir.append("")
                cara.append("")
                continue
            t = res_t.iloc[i]
            dt = (pairs_r["time"] - t).abs()
            j = int(dt.idxmin())
            temp = res.iloc[i]["sumber_temporal"]
            # hitung tri murni (tanpa fallback) untuk kolom sumber_tri
            k_tri = float(cfg["triangulation"]["k_tri"])
            hits = []
            for spec in cfg["triangulation"]["pairs"]:
                from interlake_qc.triangulation import _z_at_time

                z = _z_at_time(tri_maps.get(spec["name"], pd.DataFrame()), t)
                if np.isfinite(z) and abs(z) > k_tri:
                    hits.append(spec["points_to"])
            hits = list(dict.fromkeys(hits))
            if len(hits) == 1:
                tri_pure = hits[0]
                sumber_tri.append(tri_pure)
                sumber_akhir.append(tri_pure)
                cara.append("triangulasi")
            else:
                sumber_tri.append("")
                akhir = attribute_triangulation(
                    j, pairs_r, res, tri_maps, cfg, float(res.iloc[i]["k_dipakai"]),
                    temporal_sumber=temp,
                )
                sumber_akhir.append(akhir)
                if is_named_source(temp) and akhir == temp:
                    cara.append("temporal")
                else:
                    cara.append("ambigu" if akhir == "ambigu" else "temporal")
    else:
        for i in range(len(res)):
            if not bool(res.iloc[i]["flag"]):
                sumber_tri.append("")
                sumber_akhir.append("")
                cara.append("")
            else:
                sumber_tri.append("")
                temp = res.iloc[i]["sumber_temporal"]
                sumber_akhir.append(temp if temp else "ambigu")
                cara.append("temporal" if temp and temp not in ("ambigu", "tak bisa diatribusi") else "ambigu")

    res["sumber_tri"] = sumber_tri
    res["sumber_akhir"] = sumber_akhir
    res["cara_atribusi"] = cara
    return res, bins


def attach_obs_qc(
    d2: pd.DataFrame,
    swot: pd.DataFrame,
    danau_a: str,
    danau_b: str,
) -> pd.DataFrame:
    """Tambah quality_f & dark_frac kedua danau dari observasi SWOT terdekat (±120 s)."""
    sw = swot.copy()
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    sw["quality_f"] = pd.to_numeric(sw["quality_f"], errors="coerce")
    sw["dark_frac"] = pd.to_numeric(sw["dark_frac"], errors="coerce")
    out = d2.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    qa, qb, da, db = [], [], [], []
    for t in out["time"]:
        for danau, qlist, dlist in (
            (danau_a, qa, da),
            (danau_b, qb, db),
        ):
            sub = sw[sw["danau"] == danau]
            if sub.empty:
                qlist.append(np.nan)
                dlist.append(np.nan)
                continue
            dt = (sub["time"] - t).abs()
            j = dt.idxmin()
            if dt.loc[j] > pd.Timedelta(seconds=120):
                qlist.append(np.nan)
                dlist.append(np.nan)
            else:
                qlist.append(float(sub.loc[j, "quality_f"]))
                dlist.append(float(sub.loc[j, "dark_frac"]))
    out["quality_f_a"] = qa
    out["quality_f_b"] = qb
    out["dark_frac_a"] = da
    out["dark_frac_b"] = db
    return out
