"""R7b — muat/bersihkan data Yahara, tabel pass, kebenaran lapangan, metrik."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from interlake_qc.d2_interlake import mad, run_d2_final


LAKE_ORDER = ["mendota", "monona", "waubesa", "kegonsa"]


def load_r7_config(repo: Path, path: str = "configs/yahara.yaml") -> dict:
    with open(repo / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Proporsi + IK 95% Wilson. Kembalikan (p, lo, hi); NaN bila n=0."""
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return float(p), float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _to_nan_fills(df: pd.DataFrame, fills: list) -> pd.DataFrame:
    out = df.copy()
    fill_set = {float(x) if isinstance(x, (int, float)) else x for x in fills}
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].mask(out[c].isin(list(fill_set)), np.nan)
        else:
            # also replace string "-999"
            out[c] = out[c].replace({str(x): np.nan for x in fills})
            out[c] = out[c].replace({"NA": np.nan, "NaN": np.nan, "": np.nan})
    return out


def load_yahara_raw(repo: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load gage (+ optional sites), filter configured lakes, fill → NaN."""
    gage = pd.read_csv(
        repo / cfg["paths"]["gage_csv"],
        dtype={"gage_id": str, "lake_id": str, "cycle_id": str, "pass_id": str},
        low_memory=False,
    )
    sites_path = cfg["paths"].get("sites_csv")
    if sites_path and (repo / sites_path).exists():
        sites = pd.read_csv(
            repo / sites_path,
            dtype={"gage_id": str, "lake_id": str},
            low_memory=False,
        )
    else:
        sites = pd.DataFrame(columns=["gage_id", "lake_id", "danau", "p_ref_area"])
    id_map = {name: str(v["lake_id"]) for name, v in cfg["lakes"].items()}
    keep_ids = set(id_map.values())
    exclude = set(str(x) for x in cfg.get("exclude_lake_ids", []))

    # cetak sebelum buang
    pre_excl = gage[gage["lake_id"].isin(keep_ids | exclude)].copy()
    counts_pre = (
        pre_excl.groupby(["lake_id", "SWOT_data_version"]).size().rename("n").reset_index()
    )

    gage = gage[gage["lake_id"].isin(keep_ids)].copy()
    if len(sites):
        sites = sites[sites["lake_id"].isin(keep_ids)].copy()
    gage = _to_nan_fills(gage, cfg.get("fill_values", [-999]))
    # numerik
    for c in (
        "wse",
        "wse_u",
        "quality_f",
        "partial_f",
        "dark_frac",
        "ice_clim_f",
        "gage_wse_EGM2008",
        "gage_stage_m",
        "p_ref_area",
        "area_total",
    ):
        if c in gage.columns:
            gage[c] = pd.to_numeric(gage[c], errors="coerce")
    gage["SWOT_time"] = pd.to_datetime(gage["SWOT_time"], utc=True, errors="coerce")
    gage["gage_time_UTC"] = pd.to_datetime(gage["gage_time_UTC"], utc=True, errors="coerce")
    # nama danau
    rev = {v: k for k, v in id_map.items()}
    gage["danau"] = gage["lake_id"].map(rev)
    sites["danau"] = sites["lake_id"].map(rev)
    if "p_ref_area" in sites.columns:
        sites["p_ref_area"] = pd.to_numeric(sites["p_ref_area"], errors="coerce")

    return gage, sites, counts_pre


def dedup_prefer_D(gage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dedup kunci (lake_id, cycle_id, pass_id, SWOT_time detik); utamakan versi D.

    Kembalikan (deduped_all_versions_prefer_D, stats_before_after).
    """
    g = gage.copy()
    g["t_sec"] = g["SWOT_time"].dt.floor("s")
    key = ["lake_id", "cycle_id", "pass_id", "t_sec"]
    before = g.groupby(["danau", "SWOT_data_version"]).size().rename("n_before").reset_index()

    # sort: D first, then keep first of key
    g["_prefer"] = (g["SWOT_data_version"].astype(str) != "D").astype(int)
    g = g.sort_values(key + ["_prefer"]).drop_duplicates(key, keep="first")
    g = g.drop(columns=["_prefer"])
    after = g.groupby(["danau", "SWOT_data_version"]).size().rename("n_after").reset_index()
    stats = before.merge(after, on=["danau", "SWOT_data_version"], how="outer").fillna(0)
    return g.reset_index(drop=True), stats


def filter_version(gage: pd.DataFrame, version: str | None) -> pd.DataFrame:
    if version is None:
        return gage.copy()
    return gage[gage["SWOT_data_version"].astype(str) == str(version)].copy()


def d_preferred_else_c(gage: pd.DataFrame) -> pd.DataFrame:
    """Gabungan: D diutamakan, C bila D tidak ada (setelah dedup per kunci)."""
    g = gage.copy()
    g["t_sec"] = g["SWOT_time"].dt.floor("s")
    key = ["lake_id", "cycle_id", "pass_id", "t_sec"]
    g["_prefer"] = (g["SWOT_data_version"].astype(str) != "D").astype(int)
    g = g.sort_values(key + ["_prefer"]).drop_duplicates(key, keep="first")
    return g.drop(columns=["_prefer"]).reset_index(drop=True)


def compute_bias_and_ec(gage: pd.DataFrame, max_hours: float, main_mask: pd.Series | None = None) -> pd.DataFrame:
    """Tambah e_obs, bias_danau, e_c. Bias dari himpunan utama bila mask diberikan."""
    g = gage.copy()
    dt_h = (g["gage_time_UTC"] - g["SWOT_time"]).abs().dt.total_seconds() / 3600.0
    g["dt_gage_h"] = dt_h
    truth_ok = g["gage_wse_EGM2008"].notna() & np.isfinite(g["gage_wse_EGM2008"]) & (dt_h <= max_hours)
    g["truth_ok"] = truth_ok
    g["e_obs"] = np.where(truth_ok, g["wse"] - g["gage_wse_EGM2008"], np.nan)
    biases = {}
    for danau, sub in g.groupby("danau"):
        if main_mask is not None:
            m = main_mask.reindex(sub.index).fillna(False)
            e = sub.loc[m & sub["truth_ok"], "e_obs"]
        else:
            e = sub.loc[sub["truth_ok"], "e_obs"]
        biases[danau] = float(np.nanmedian(e)) if len(e) and e.notna().any() else float("nan")
    g["bias_danau"] = g["danau"].map(biases)
    g["e_c"] = g["e_obs"] - g["bias_danau"]
    g.attrs["biases"] = biases
    return g


def main_set_mask(g: pd.DataFrame, require_ice0: bool = True) -> pd.Series:
    m = g["quality_f"].isin([0, 1]) & g["wse"].notna()
    if require_ice0:
        m = m & (g["ice_clim_f"] == 0)
    return m


def build_pair_table(
    gage: pd.DataFrame,
    danau_a: str,
    danau_b: str,
    max_dt_sec: float = 120.0,
    require_ice0: bool = True,
) -> pd.DataFrame:
    """Pass bersama: cycle+pass sama, |Δt|≤120s; kolom kebenaran digabung nanti."""
    a = gage[gage["danau"] == danau_a].copy()
    b = gage[gage["danau"] == danau_b].copy()
    rows = []
    # index B by (cycle, pass)
    b["_key"] = b["cycle_id"].astype(str) + "|" + b["pass_id"].astype(str)
    for _, ra in a.iterrows():
        key = f"{ra['cycle_id']}|{ra['pass_id']}"
        cand = b[b["_key"] == key]
        if cand.empty or pd.isna(ra["SWOT_time"]):
            continue
        dt = (cand["SWOT_time"] - ra["SWOT_time"]).abs()
        j = dt.idxmin()
        if dt.loc[j] > pd.Timedelta(seconds=max_dt_sec):
            continue
        rb = cand.loc[j]
        # buang bila gage_id sama di kedua sisi
        if str(ra["gage_id"]) == str(rb["gage_id"]):
            continue
        q_a, q_b = ra["quality_f"], rb["quality_f"]
        ice_a, ice_b = ra["ice_clim_f"], rb["ice_clim_f"]
        core = (q_a in (0, 1)) and (q_b in (0, 1))
        ice_ok = (ice_a == 0) and (ice_b == 0)
        main = core and (ice_ok if require_ice0 else True)
        dWSE = float(ra["wse"]) - float(rb["wse"]) if np.isfinite(ra["wse"]) and np.isfinite(rb["wse"]) else np.nan
        H = 0.5 * (float(ra["wse"]) + float(rb["wse"])) if np.isfinite(ra["wse"]) and np.isfinite(rb["wse"]) else np.nan
        # kebenaran
        ga, gb = ra["gage_wse_EGM2008"], rb["gage_wse_EGM2008"]
        ta_ok = bool(ra.get("truth_ok", False))
        tb_ok = bool(rb.get("truth_ok", False))
        both_truth = ta_ok and tb_ok and np.isfinite(ga) and np.isfinite(gb)
        e_dwse = dWSE - (float(ga) - float(gb)) if both_truth and np.isfinite(dWSE) else np.nan
        rows.append(
            {
                "time": ra["SWOT_time"],
                "day": pd.Timestamp(ra["SWOT_time"]).strftime("%Y-%m-%d") if pd.notna(ra["SWOT_time"]) else "",
                "pass": str(ra["pass_id"]),
                "cycle_id": str(ra["cycle_id"]),
                "wse_a": float(ra["wse"]) if np.isfinite(ra["wse"]) else np.nan,
                "wse_b": float(rb["wse"]) if np.isfinite(rb["wse"]) else np.nan,
                "q_a": q_a,
                "q_b": q_b,
                "partial_a": ra.get("partial_f", np.nan),
                "partial_b": rb.get("partial_f", np.nan),
                "dark_a": ra.get("dark_frac", np.nan),
                "dark_b": rb.get("dark_frac", np.nan),
                "ice_a": ice_a,
                "ice_b": ice_b,
                "dWSE": dWSE,
                "H": H,
                "core": bool(core),
                "ice_ok": bool(ice_ok),
                "main": bool(main),
                "sudah_ditandai_flag": bool(q_a in (2, 3) or q_b in (2, 3)),
                "phase": "",
                "gage_a": float(ga) if np.isfinite(ga) else np.nan,
                "gage_b": float(gb) if np.isfinite(gb) else np.nan,
                "e_c_a": float(ra["e_c"]) if np.isfinite(ra.get("e_c", np.nan)) else np.nan,
                "e_c_b": float(rb["e_c"]) if np.isfinite(rb.get("e_c", np.nan)) else np.nan,
                "e_dwse": e_dwse,
                "both_truth": both_truth,
                "SWOT_data_version_a": ra.get("SWOT_data_version", ""),
                "SWOT_data_version_b": rb.get("SWOT_data_version", ""),
            }
        )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    # e_dwse_c = e_dwse − median(e_dwse) pada himpunan utama berkebenaran
    med = np.nanmedian(pairs.loc[pairs["main"] & pairs["both_truth"], "e_dwse"])
    pairs["e_dwse_c"] = pairs["e_dwse"] - med
    pairs["median_e_dwse"] = med
    return pairs.sort_values("time").reset_index(drop=True)


def label_truth(pairs: pd.DataFrame, T: float) -> pd.DataFrame:
    p = pairs.copy()
    p["bermasalah"] = p["both_truth"] & np.isfinite(p["e_dwse_c"]) & (p["e_dwse_c"].abs() > T)
    # danau bersalah
    guilty = []
    for _, r in p.iterrows():
        if not (r["both_truth"] and np.isfinite(r["e_c_a"]) and np.isfinite(r["e_c_b"])):
            guilty.append("")
            continue
        ea, eb = abs(float(r["e_c_a"])), abs(float(r["e_c_b"]))
        if max(ea, eb) <= T:
            guilty.append("")
        elif ea > eb:
            guilty.append("a")  # danau_a
        elif eb > ea:
            guilty.append("b")
        else:
            guilty.append("tie")
    p["guilty_side"] = guilty
    return p


def to_swot_like(gage: pd.DataFrame) -> pd.DataFrame:
    """Format mirip Hydrocron untuk run_d2_final / B-temp / triangulasi."""
    s = gage.copy()
    s["time"] = s["SWOT_time"]
    s["pass"] = s["pass_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    return s


def make_d2_cfg(
    base_d2: dict,
    pair_spec: dict,
    det: dict,
    all_lakes_cfg: dict,
) -> dict:
    """Cfg untuk satu pasangan + triangulasi terhadap danau_tri."""
    cfg = copy.deepcopy(base_d2)
    a, b, tri = pair_spec["danau_a"], pair_spec["danau_b"], pair_spec["danau_tri"]
    cfg["pair"] = {
        "name": pair_spec["name"],
        "danau_a": a,
        "danau_b": b,
        "danau_tri": tri,
    }
    cfg["detection_final"] = {
        "h_threshold_m": float(det["h_threshold_m"]),
        "k_high": float(det["k_high"]),
        "k_low": float(det["k_low"]),
    }
    cfg["detection"] = {"k_values": [float(det["k_high"]), float(det["k_low"])]}
    cfg["triangulation"] = {
        "k_tri": float(base_d2.get("triangulation", {}).get("k_tri", 2.5)),
        "pairs": [
            {
                "name": f"{a}_minus_{tri}",
                "danau_a": a,
                "danau_b": tri,
                "points_to": a,
            },
            {
                "name": f"{b}_minus_{tri}",
                "danau_a": b,
                "danau_b": tri,
                "points_to": b,
            },
        ],
    }
    return cfg


def map_sumber_to_side(sumber: str, danau_a: str, danau_b: str) -> str:
    s = str(sumber).strip().lower()
    if s == danau_a.lower():
        return "a"
    if s == danau_b.lower():
        return "b"
    return s  # ambigu / tak bisa...


def method_flags_on_pairs(
    pairs: pd.DataFrame,
    d2_flags: pd.Series | None,
    btemp_flag_times: dict[str, set] | None,
    danau_a: str,
    danau_b: str,
    ketat: dict,
) -> dict[str, pd.Series]:
    """Boolean series indexed like pairs for methods (i)–(iv). True = ditandai/dibuang."""
    n = len(pairs)
    # (i) q01: "dibuang" = tidak lolos q01 → ~core
    m_q01_drop = ~pairs["core"].astype(bool)
    # (ii) ketat
    ketat_ok = (
        (pairs["q_a"] == ketat["quality_f"])
        & (pairs["q_b"] == ketat["quality_f"])
        & (pairs["partial_a"] == ketat["partial_f"])
        & (pairs["partial_b"] == ketat["partial_f"])
        & (pairs["dark_a"] < ketat["dark_frac_max"])
        & (pairs["dark_b"] < ketat["dark_frac_max"])
    )
    m_ketat_drop = ~ketat_ok
    # (iii) B-temp
    bt = np.zeros(n, dtype=bool)
    if btemp_flag_times:
        for i, t in enumerate(pairs["time"]):
            t = pd.Timestamp(t)
            hit = False
            for danau in (danau_a, danau_b):
                for ft in btemp_flag_times.get(danau, set()):
                    if abs((ft - t).total_seconds()) <= 120:
                        hit = True
                        break
                if hit:
                    break
            bt[i] = hit
    m_bt = pd.Series(bt, index=pairs.index)
    # (iv) D2
    if d2_flags is None:
        m_d2 = pd.Series(False, index=pairs.index)
    else:
        m_d2 = d2_flags.reindex(pairs.index).fillna(False).astype(bool)
    return {
        "q01": m_q01_drop,
        "ketat": m_ketat_drop,
        "Btemp": m_bt,
        "D2": m_d2,
    }


def metrics_block(
    pairs: pd.DataFrame,
    method_masks: dict[str, pd.Series],
    T: float,
    k_label: float,
    pair_name: str,
    require_main: bool = True,
) -> list[dict]:
    """Hitung metrik deteksi untuk pass berkebenaran (opsional himpunan utama)."""
    p = label_truth(pairs, T)
    if require_main:
        base = p[p["main"] & p["both_truth"]].copy()
    else:
        base = p[p["both_truth"]].copy()
    n_truth = len(base)
    n_bad = int(base["bermasalah"].sum())
    rows = []
    for method, mask_all in method_masks.items():
        mask = mask_all.reindex(base.index).fillna(False).astype(bool)
        # flagged/dropped among truth-labeled
        flagged = mask
        tp = int((flagged & base["bermasalah"]).sum())
        fp = int((flagged & ~base["bermasalah"]).sum())
        fn = int((~flagged & base["bermasalah"]).sum())
        tn = int((~flagged & ~base["bermasalah"]).sum())
        n_flag = int(flagged.sum())
        # precision/recall/FPR
        prec, prec_lo, prec_hi = wilson_ci(tp, tp + fp)
        rec, rec_lo, rec_hi = wilson_ci(tp, tp + fn)  # among bad
        # FPR = flagged among not-bad
        fpr, fpr_lo, fpr_hi = wilson_ci(fp, fp + tn)
        # fraksi data dipertahankan: di himpunan evaluasi, 1 - n_flag/n
        retained = 1.0 - (n_flag / n_truth) if n_truth else float("nan")
        rows.append(
            {
                "pair": pair_name,
                "method": method,
                "T": T,
                "k": k_label,
                "n_truth": n_truth,
                "n_bad": n_bad,
                "prevalence": n_bad / n_truth if n_truth else float("nan"),
                "n_flagged": n_flag,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prec,
                "precision_lo": prec_lo,
                "precision_hi": prec_hi,
                "recall": rec,
                "recall_lo": rec_lo,
                "recall_hi": rec_hi,
                "fpr": fpr,
                "fpr_lo": fpr_lo,
                "fpr_hi": fpr_hi,
                "frac_retained": retained,
            }
        )
    return rows


def attribution_rows(
    merged: pd.DataFrame,
    danau_a: str,
    danau_b: str,
    T: float,
    pair_name: str,
    k_label: float,
    himpunan: str = "utama_ice0_q01",
) -> list[dict]:
    """Proporsi atribusi benar/salah/ambigu di antara D2 flag & bermasalah & punya guilty."""
    m = merged[
        merged["flag"].astype(bool)
        & merged["bermasalah"].astype(bool)
        & merged["guilty_side"].isin(["a", "b"])
    ].copy()
    rows = []
    for cara in ("triangulasi", "temporal", "ambigu", "all"):
        if cara == "all":
            sub = m
        else:
            sub = m[m["cara_atribusi"] == cara] if cara != "ambigu" else m[m["cara_atribusi"].isin(["ambigu", ""])]
            if cara == "ambigu":
                sub = m[~m["cara_atribusi"].isin(["triangulasi", "temporal"])]
        n = len(sub)
        if n == 0:
            rows.append(
                {
                    "pair": pair_name,
                    "himpunan": himpunan,
                    "k": k_label,
                    "T": T,
                    "cara": cara,
                    "n": 0,
                    "benar": 0,
                    "salah": 0,
                    "ambigu_attr": 0,
                    "frac_benar": float("nan"),
                }
            )
            continue
        benar = salah = amb = 0
        for _, r in sub.iterrows():
            side = map_sumber_to_side(r.get("sumber_akhir", ""), danau_a, danau_b)
            if side not in ("a", "b"):
                amb += 1
            elif side == r["guilty_side"]:
                benar += 1
            else:
                salah += 1
        rows.append(
            {
                "pair": pair_name,
                "himpunan": himpunan,
                "k": k_label,
                "T": T,
                "cara": cara,
                "n": n,
                "benar": benar,
                "salah": salah,
                "ambigu_attr": amb,
                "frac_benar": benar / n,
                "frac_salah": salah / n,
                "frac_ambigu": amb / n,
            }
        )
    return rows


def rmse_mad_by_lake_variants(
    gage: pd.DataFrame,
    d2_merged_by_pair: dict[str, pd.DataFrame],
    pair_specs: dict,
    btemp: pd.DataFrame,
    ketat: dict,
) -> pd.DataFrame:
    """V1 q01, V2 ketat, V3 q01+D2 attr, V4 q01+Btemp — RMSE/MAD e_c per danau."""
    g = gage.copy()
    g["time"] = pd.to_datetime(g["SWOT_time"], utc=True)
    # collect D2 flagged drops per lake
    drop_v3: dict[str, set] = {d: set() for d in g["danau"].unique()}
    for pname, spec in pair_specs.items():
        m = d2_merged_by_pair.get(pname)
        if m is None or m.empty:
            continue
        a, b = spec["danau_a"], spec["danau_b"]
        for _, r in m[m["flag"].astype(bool)].iterrows():
            t = pd.Timestamp(r["time"])
            src = str(r.get("sumber_akhir", ""))
            side = map_sumber_to_side(src, a, b)
            if side == "a":
                drop_v3[a].add(t)
            elif side == "b":
                drop_v3[b].add(t)
            else:
                drop_v3[a].add(t)
                drop_v3[b].add(t)

    k_bt = 3.5
    fcol = f"flag_k{k_bt}"
    bt_times: dict[str, set] = {}
    if btemp is not None and len(btemp) and fcol in btemp.columns:
        bt = btemp.copy()
        bt["time"] = pd.to_datetime(bt["time"], utc=True)
        for danau, sub in bt.groupby("danau"):
            bt_times[danau] = set(pd.Timestamp(t) for t in sub.loc[sub[fcol].astype(bool), "time"])

    rows = []
    for danau, sub0 in g.groupby("danau"):
        sub = sub0.copy()
        # only truth_ok for RMSE of e_c
        for variant, keep_fn in (
            (
                "V1_q01",
                lambda s: s["quality_f"].isin([0, 1]),
            ),
            (
                "V2_ketat",
                lambda s: (
                    (s["quality_f"] == ketat["quality_f"])
                    & (s["partial_f"] == ketat["partial_f"])
                    & (s["dark_frac"] < ketat["dark_frac_max"])
                ),
            ),
            (
                "V3_q01_D2attr",
                lambda s, d=danau: s["quality_f"].isin([0, 1])
                & ~s["time"].map(
                    lambda t: any(abs((t - ft).total_seconds()) <= 120 for ft in drop_v3.get(d, set()))
                ),
            ),
            (
                "V4_q01_Btemp",
                lambda s, d=danau: s["quality_f"].isin([0, 1])
                & ~s["time"].map(
                    lambda t: any(abs((t - ft).total_seconds()) <= 120 for ft in bt_times.get(d, set()))
                ),
            ),
        ):
            m = keep_fn(sub) & sub["truth_ok"] & np.isfinite(sub["e_c"])
            e = sub.loc[m, "e_c"].to_numpy(dtype=float)
            rows.append(
                {
                    "danau": danau,
                    "variant": variant,
                    "n": int(m.sum()),
                    "rmse_e_c": float(np.sqrt(np.mean(e**2))) if e.size else "",
                    "mad_e_c": mad(e) if e.size else "",
                }
            )
    return pd.DataFrame(rows)
