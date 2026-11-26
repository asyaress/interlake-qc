"""R6 — OAT sensitivitas, LOYO, kurva per-pass, S1 ringkas."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from interlake_qc.core import (
    build_reference_bins,
    interp_f_sigma,
    k_for_H,
    run_d2_final,
)


def band_of(h: float, thr: float = 2.5) -> str:
    return f"H<{thr}" if float(h) < float(thr) else f"H>={thr}"


# alias if not in d2_interlake
def _band(h: float, thr: float = 2.5) -> str:
    return "H<2.5" if float(h) < float(thr) else "H>=2.5"


def apply_params_to_cfg(base_cfg: dict, params: dict) -> dict:
    """Salin cfg dan tulis parameter OAT ke reference_curve / detection_final."""
    cfg = copy.deepcopy(base_cfg)
    rc = cfg.setdefault("reference_curve", {})
    df = cfg.setdefault("detection_final", {})
    if "bin_width_m" in params:
        rc["bin_width_m"] = float(params["bin_width_m"])
    if "min_n_per_bin" in params:
        rc["min_n_per_bin"] = int(params["min_n_per_bin"])
    if "sigma_min_m" in params:
        rc["sigma_min_m"] = float(params["sigma_min_m"])
    if "h_threshold_m" in params:
        df["h_threshold_m"] = float(params["h_threshold_m"])
    if "k_high" in params:
        df["k_high"] = float(params["k_high"])
    if "k_low" in params:
        df["k_low"] = float(params["k_low"])
    return cfg


def build_oat_configs(r6: dict) -> list[dict]:
    """Daftar konfigurasi OAT: tiap (param, nilai) dengan parameter lain = default.

    Setiap entri: varied_param, value, params (dict lengkap), config_id.
    """
    defaults = dict(r6["defaults"])
    rows = []
    cid = 0
    for param, values in r6["oat"].items():
        for v in values:
            params = dict(defaults)
            params[param] = v
            rows.append(
                {
                    "config_id": f"oat_{cid:02d}",
                    "varied_param": param,
                    "varied_value": v,
                    "params": params,
                }
            )
            cid += 1
    return rows


def oat_differs_by_one(cfg_params: dict, defaults: dict) -> bool:
    """True bila tepat satu kunci berbeda dari default (atau nol = default murni)."""
    diffs = [k for k in defaults if cfg_params.get(k) != defaults[k]]
    # juga kunci ekstra
    return len(diffs) <= 1


def unique_param_key(params: dict) -> tuple:
    return tuple(
        (k, float(params[k]) if isinstance(params[k], float) else params[k])
        for k in sorted(params)
    )


def run_real_flags(
    pairs: pd.DataFrame,
    lake_series: dict[str, pd.DataFrame],
    cfg: dict,
) -> pd.DataFrame:
    """D2 final tanpa atribusi (untuk OAT cepat)."""
    res, _ = run_d2_final(pairs, lake_series, cfg, do_attribution=False, swot=None)
    return res


def compact_s1(
    pairs_all: pd.DataFrame,
    lake_series0: dict[str, pd.DataFrame],
    cfg: dict,
    n_ulang: int = 50,
    deltas: list[float] | None = None,
    frac: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """S1 ringkas: sisip ke data nyata, tanpa pra-saring, hanya D2 banded-k.

    Kembalikan recall & additional_fp_rate per (delta, h_band).
    Pita H memakai ambang default 2.5 untuk pelaporan (bukan batas_H OAT),
    agar metrik antar-konfigurasi sebanding; deteksi tetap memakai k per pita cfg.
    """
    deltas = deltas or [0.20, 0.30]
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]
    lakes = [danau_a, danau_b]

    pairs = pairs_all[pairs_all["core"]].copy().reset_index(drop=True)
    pairs["time"] = pd.to_datetime(pairs["time"], utc=True)
    n = len(pairs)
    n_inject = max(1, int(round(frac * n)))

    base = run_real_flags(pairs, lake_series0, cfg)
    base_flag = base["flag"].astype(bool).to_numpy()

    rng = np.random.default_rng(seed)
    stats: dict[tuple, dict[str, float]] = {}

    def acc(key: tuple, **kw: float) -> None:
        if key not in stats:
            stats[key] = {
                "n_inject": 0,
                "n_detect": 0,
                "n_noninject": 0,
                "n_add_fp": 0,
            }
        for k, v in kw.items():
            stats[key][k] += v

    for delta in deltas:
        for _trial in range(n_ulang):
            inj_idx = list(rng.choice(n, size=min(n_inject, n), replace=False))
            inj_lake = {int(i): lakes[int(rng.integers(0, 2))] for i in inj_idx}
            signs = {int(i): float(rng.choice([-1.0, 1.0])) for i in inj_idx}
            inj_set = set(inj_lake)

            pairs_m = pairs.copy()
            for i, lake in inj_lake.items():
                s = signs[i]
                if lake == danau_a:
                    pairs_m.loc[i, "wse_a"] = float(pairs_m.loc[i, "wse_a"]) + s * delta
                else:
                    pairs_m.loc[i, "wse_b"] = float(pairs_m.loc[i, "wse_b"]) + s * delta
                pairs_m.loc[i, "dWSE"] = float(pairs_m.loc[i, "wse_a"]) - float(pairs_m.loc[i, "wse_b"])
                pairs_m.loc[i, "H"] = 0.5 * (
                    float(pairs_m.loc[i, "wse_a"]) + float(pairs_m.loc[i, "wse_b"])
                )

            # lake series tidak diubah untuk deteksi D2 antar-danau (hanya pairs)
            res_m = run_real_flags(pairs_m, lake_series0, cfg)
            flagged = res_m["flag"].astype(bool).to_numpy()

            for i in range(n):
                # band dari H asli (sebelum inject) agar pelaporan stabil
                band = _band(float(pairs.loc[i, "H"]), 2.5)
                key = (float(delta), band)
                if i in inj_set:
                    acc(key, n_inject=1)
                    if flagged[i]:
                        acc(key, n_detect=1)
                else:
                    acc(key, n_noninject=1)
                    if flagged[i] and not base_flag[i]:
                        acc(key, n_add_fp=1)

    rows = []
    for (delta, band), st in sorted(stats.items()):
        rows.append(
            {
                "delta_m": delta,
                "h_band": band,
                "n_inject": int(st["n_inject"]),
                "n_detect": int(st["n_detect"]),
                "recall": st["n_detect"] / st["n_inject"] if st["n_inject"] else float("nan"),
                "n_noninject": int(st["n_noninject"]),
                "n_add_fp": int(st["n_add_fp"]),
                "additional_fp_rate": (
                    st["n_add_fp"] / st["n_noninject"] if st["n_noninject"] else float("nan")
                ),
                "n_ulang": n_ulang,
            }
        )
    return pd.DataFrame(rows)


def run_oat(
    pairs: pd.DataFrame,
    lake_series: dict[str, pd.DataFrame],
    base_cfg: dict,
    r6: dict,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Jalankan semua OAT; kembalikan (oat_table, flag_frequency, flagged_dates_by_cfg)."""
    configs = build_oat_configs(r6)
    s1cfg = r6["s1_compact"]
    oat_rows = []
    flagged_map: dict[str, list[str]] = {}
    # untuk frekuensi: hanya konfigurasi unik (param tuple)
    unique_flags: dict[tuple, set[str]] = {}
    unique_order: list[tuple] = []

    for oc in configs:
        params = oc["params"]
        cfg = apply_params_to_cfg(base_cfg, params)
        res = run_real_flags(pairs, lake_series, cfg)
        fl = res[res["flag"].astype(bool)]
        dates = sorted(pd.to_datetime(fl["time"], utc=True).dt.strftime("%Y-%m-%d").unique().tolist())
        flagged_map[oc["config_id"]] = dates

        ukey = unique_param_key(params)
        if ukey not in unique_flags:
            unique_flags[ukey] = set(dates)
            unique_order.append(ukey)
        else:
            # sama — pastikan set tanggal konsisten
            unique_flags[ukey] = set(dates)

        print(
            f"  {oc['config_id']} {oc['varied_param']}={oc['varied_value']} "
            f"n_flag={len(dates)} S1...",
            flush=True,
        )
        s1 = compact_s1(
            pairs,
            lake_series,
            cfg,
            n_ulang=int(s1cfg["n_ulang"]),
            deltas=[float(x) for x in s1cfg["deltas_m"]],
            frac=float(s1cfg["fraction_inject"]),
            seed=seed,
        )
        # flatten S1 metrics into one row
        row: dict[str, Any] = {
            "config_id": oc["config_id"],
            "varied_param": oc["varied_param"],
            "varied_value": oc["varied_value"],
            **{f"p_{k}": params[k] for k in sorted(params)},
            "n_flagged": len(dates),
            "flagged_dates": ";".join(dates),
        }
        for _, sr in s1.iterrows():
            d = sr["delta_m"]
            b = "low" if "H<" in str(sr["h_band"]) else "high"
            row[f"recall_d{d}_{b}"] = sr["recall"]
            row[f"addfp_d{d}_{b}"] = sr["additional_fp_rate"]
        oat_rows.append(row)

    oat_df = pd.DataFrame(oat_rows)

    # frekuensi: semua tanggal yang pernah tertandai di config unik mana pun
    n_unique = len(unique_order)
    all_dates: set[str] = set()
    for s in unique_flags.values():
        all_dates |= s
    freq_rows = []
    for d in sorted(all_dates):
        n_hit = sum(1 for uk in unique_order if d in unique_flags[uk])
        freq_rows.append(
            {
                "tanggal": d,
                "n_configs_flagged": n_hit,
                "n_unique_configs": n_unique,
                "freq": n_hit / n_unique if n_unique else float("nan"),
            }
        )
    freq_df = pd.DataFrame(freq_rows)
    return oat_df, freq_df, flagged_map


def leave_one_year_out(
    pairs: pd.DataFrame,
    base_cfg: dict,
    years: list[int],
    default_flagged: set[str] | None = None,
) -> pd.DataFrame:
    """Kurva acuan dari tahun ≠ Y; z + flag untuk pass tahun Y dengan k default.

    Tidak memakai data tahun Y dalam kurva (bukan LOO per-pass).
    """
    rc = base_cfg["reference_curve"]
    d = base_cfg["detection_final"]
    h_thr = float(d["h_threshold_m"])
    k_high = float(d["k_high"])
    k_low = float(d["k_low"])

    p = pairs.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    p["year"] = p["time"].dt.year
    core = p[p["core"]].copy()

    # default LOO flags for comparison
    if default_flagged is None:
        # approximate: run full LOO final once externally; here compute LOYO-only compare later
        default_flagged = set()

    rows = []
    for Y in years:
        train = core[core["year"] != Y]
        test = p[p["year"] == Y].copy()
        # assert no year Y in train
        assert (train["year"] != Y).all()
        bins = build_reference_bins(
            train["H"].to_numpy(),
            train["dWSE"].to_numpy(),
            bin_width=float(rc["bin_width_m"]),
            min_n=int(rc["min_n_per_bin"]),
            mad_scale=float(rc["mad_scale"]),
            sigma_min=float(rc["sigma_min_m"]),
        )
        for _, row in test.iterrows():
            f_h, sig = interp_f_sigma(float(row["H"]), bins)
            if not np.isfinite(sig) or sig < float(rc["sigma_min_m"]):
                sig = float(rc["sigma_min_m"])
            z = (float(row["dWSE"]) - f_h) / sig if np.isfinite(f_h) and sig > 0 else float("nan")
            kk = k_for_H(float(row["H"]), h_thr, k_high, k_low)
            flag = bool(np.isfinite(z) and abs(z) > kk)
            day = pd.Timestamp(row["time"]).strftime("%Y-%m-%d")
            in_default = day in default_flagged
            if flag and in_default:
                vs = "sama"
            elif flag and not in_default:
                vs = "tambah"
            elif (not flag) and in_default:
                vs = "hilang"
            else:
                vs = "tetap_bersih"
            rows.append(
                {
                    "year_held_out": Y,
                    "tanggal": day,
                    "pass": row.get("pass", ""),
                    "H": row["H"],
                    "dWSE": row["dWSE"],
                    "f_H": f_h,
                    "sigma_H": sig,
                    "z": z,
                    "k": kk,
                    "flag": flag,
                    "core": bool(row["core"]),
                    "n_train": len(train),
                    "vs_default": vs if (flag or in_default) else "",
                }
            )
    return pd.DataFrame(rows)


def by_pass_curves(
    pairs: pd.DataFrame,
    base_cfg: dict,
    passes: list[str] | None = None,
) -> pd.DataFrame:
    """Kurva f(H), σ(H) terpisah untuk pass 284 dan 437."""
    passes = passes or ["284", "437"]
    rc = base_cfg["reference_curve"]
    p = pairs.copy()
    p["pass"] = p["pass"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
    core = p[p["core"]].copy()

    curves = {}
    for ps in passes:
        sub = core[core["pass"] == str(ps).zfill(3)]
        bins = build_reference_bins(
            sub["H"].to_numpy(),
            sub["dWSE"].to_numpy(),
            bin_width=float(rc["bin_width_m"]),
            min_n=int(rc["min_n_per_bin"]),
            mad_scale=float(rc["mad_scale"]),
            sigma_min=float(rc["sigma_min_m"]),
        )
        curves[ps] = bins

    # gabung per pusat bin mendekati
    # bangun grid dari gabungan pusat
    all_centers = sorted(
        {float(c) for b in curves.values() for c in b["center"].tolist()} if curves else []
    )
    # lebih baik: align by overlapping bin edges — gunakan interp di pusat gabungan
    if not all_centers:
        return pd.DataFrame()

    # gunakan pusat dari pass pertama sebagai basis + pusat kedua
    centers = sorted(set(all_centers))
    rows = []
    for c in centers:
        rec: dict[str, Any] = {"center": c}
        for ps in passes:
            bins = curves[ps]
            if len(bins) == 0:
                rec[f"f_{ps}"] = float("nan")
                rec[f"sigma_{ps}"] = float("nan")
                rec[f"n_{ps}"] = 0
                continue
            f, s = interp_f_sigma(c, bins)
            # n: bin terdekat
            j = int(np.argmin(np.abs(bins["center"].to_numpy() - c)))
            n_j = int(bins.iloc[j]["n"]) if abs(float(bins.iloc[j]["center"]) - c) < 0.2 else 0
            rec[f"f_{ps}"] = f
            rec[f"sigma_{ps}"] = s
            rec[f"n_{ps}"] = n_j
        if "284" in passes and "437" in passes:
            rec["f_diff_284_minus_437"] = rec.get("f_284", np.nan) - rec.get("f_437", np.nan)
            rec["sigma_diff_284_minus_437"] = rec.get("sigma_284", np.nan) - rec.get("sigma_437", np.nan)
        rows.append(rec)

    # juga tabel bin mentah sejajar lebih informatif: merge on nearest center
    # Tambahan: baris per-bin asli masing-masing
    raw_rows = []
    b284 = curves.get("284", pd.DataFrame())
    b437 = curves.get("437", pd.DataFrame())
    # gunakan union of bin_lo
    keys = []
    if len(b284):
        for _, r in b284.iterrows():
            keys.append((float(r["bin_lo"]), float(r["bin_hi"])))
    if len(b437):
        for _, r in b437.iterrows():
            k = (float(r["bin_lo"]), float(r["bin_hi"]))
            if k not in keys:
                keys.append(k)
    keys = sorted(keys)
    for lo, hi in keys:
        def _match(bins, lo, hi):
            if bins is None or len(bins) == 0:
                return None
            m = bins[(np.isclose(bins["bin_lo"], lo)) & (np.isclose(bins["bin_hi"], hi))]
            if len(m):
                return m.iloc[0]
            # nearest center
            c = 0.5 * (lo + hi)
            j = int(np.argmin(np.abs(bins["center"].to_numpy() - c)))
            if abs(float(bins.iloc[j]["center"]) - c) <= 0.2:
                return bins.iloc[j]
            return None

        r284 = _match(b284, lo, hi)
        r437 = _match(b437, lo, hi)
        f284 = float(r284["f"]) if r284 is not None else float("nan")
        f437 = float(r437["f"]) if r437 is not None else float("nan")
        raw_rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "center": 0.5 * (lo + hi),
                "f_284": f284,
                "f_437": f437,
                "f_diff": f284 - f437 if np.isfinite(f284) and np.isfinite(f437) else float("nan"),
                "n_284": int(r284["n"]) if r284 is not None else 0,
                "n_437": int(r437["n"]) if r437 is not None else 0,
                "sigma_284": float(r284["sigma"]) if r284 is not None else float("nan"),
                "sigma_437": float(r437["sigma"]) if r437 is not None else float("nan"),
            }
        )
    return pd.DataFrame(raw_rows)
