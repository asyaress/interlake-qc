"""S1 (sisip ke data nyata non-sirkular) dan S2 (sintetis penuh)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from interlake_qc.baseline_temporal import run_btemp_all_lakes
from interlake_qc.core import attribute_source, run_d2
from interlake_qc.common import band_of, d2_cfg_from_d2b, flag_col, interp_truth, is_named_source, truth_curve_table
from interlake_qc.triangulation import attribute_triangulation, build_jempang_pair_results


def _tri_maps_with_inject(
    tri_maps: dict[str, pd.DataFrame],
    pairs_sm: pd.DataFrame,
    inj_lake: dict[int, str],
    signs: dict[int, float],
    delta: float,
) -> dict[str, pd.DataFrame]:
    """Geser z pasangan-Jempang untuk pass tersisip: ΔdWSE = ±δ pada danau yang diubah.

    Semayang−Jempang: inject Semayang → dWSE += s·δ
    Melintang−Jempang: inject Melintang → dWSE += s·δ
    f(H), σ(H) tetap (aproksimasi LOO lokal; Jempang tidak berubah).
    """
    out = {k: v.copy() for k, v in tri_maps.items()}
    for i, lake in inj_lake.items():
        t = pd.Timestamp(pairs_sm.loc[i, "time"])
        s = signs[i]
        for name, res in out.items():
            if res is None or len(res) == 0 or "z" not in res.columns:
                continue
            points_to = str(res["points_to"].iloc[0]) if "points_to" in res.columns else ""
            if points_to != lake:
                continue
            dt = (res["time"] - t).abs()
            j = int(dt.idxmin())
            if dt.loc[j] > pd.Timedelta(seconds=120):
                continue
            sig = float(res.loc[j, "sigma_H"]) if "sigma_H" in res.columns else 0.02
            if not np.isfinite(sig) or sig <= 0:
                sig = 0.02
            # z = (dWSE - f)/σ → z' = z + (s·δ)/σ
            out[name].loc[j, "z"] = float(res.loc[j, "z"]) + (s * delta) / sig
    return out


def _acc(stats: dict, key: tuple, **kw: float) -> None:
    if key not in stats:
        stats[key] = {
            "n_inject": 0,
            "n_detect": 0,
            "n_noninject": 0,
            "n_add_fp": 0,
            "n_base_flag": 0,
            "n_true_fp": 0,
            "n_attr_named": 0,
            "n_attr_ok": 0,
            "n_flagged_for_attr": 0,
            "n_tri_named": 0,
            "n_tri_ok": 0,
            "n_tri_flagged": 0,
        }
    for k, v in kw.items():
        stats[key][k] += v


def _stats_to_df(stats: dict, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for key, st in sorted(stats.items()):
        # key layouts differ S1 vs S2
        row: dict[str, Any] = {"key": str(key)}
        if len(key) == 4:
            method, delta, kk, band = key
            row.update({"method": method, "delta_m": delta, "k": kk, "h_band": band, "c": ""})
        else:
            method, c, delta, kk, band = key
            row.update({"method": method, "c": c, "delta_m": delta, "k": kk, "h_band": band})
        n_inj = st["n_inject"]
        n_non = st["n_noninject"]
        row["n_inject"] = n_inj
        row["n_detect"] = st["n_detect"]
        row["recall"] = st["n_detect"] / n_inj if n_inj else float("nan")
        row["n_noninject"] = n_non
        row["n_add_fp"] = st["n_add_fp"]
        row["additional_fp_rate"] = st["n_add_fp"] / n_non if n_non else float("nan")
        row["n_base_flag"] = st["n_base_flag"]
        row["baseline_flag_rate"] = st["n_base_flag"] / (n_inj + n_non) if (n_inj + n_non) else float("nan")
        row["n_true_fp"] = st["n_true_fp"]
        row["true_fp_rate"] = st["n_true_fp"] / n_non if n_non else float("nan")
        row["n_flagged_for_attr"] = st["n_flagged_for_attr"]
        row["n_attr_named"] = st["n_attr_named"]
        row["attr_coverage"] = (
            st["n_attr_named"] / st["n_flagged_for_attr"] if st["n_flagged_for_attr"] else float("nan")
        )
        row["n_attr_ok"] = st["n_attr_ok"]
        row["attr_precision"] = st["n_attr_ok"] / st["n_attr_named"] if st["n_attr_named"] else float("nan")
        row["n_tri_flagged"] = st["n_tri_flagged"]
        row["n_tri_named"] = st["n_tri_named"]
        row["tri_coverage"] = (
            st["n_tri_named"] / st["n_tri_flagged"] if st["n_tri_flagged"] else float("nan")
        )
        row["n_tri_ok"] = st["n_tri_ok"]
        row["tri_precision"] = st["n_tri_ok"] / st["n_tri_named"] if st["n_tri_named"] else float("nan")
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def generate_s2_truth(
    pairs_core: pd.DataFrame,
    bins: pd.DataFrame,
    rng: np.random.Generator,
    student_df: float,
    common_sd: float,
    noise: bool = True,
) -> pd.DataFrame:
    """Bangun muka air sintetis: Melintang=asli; Semayang=Mel+g(H); derau opsional."""
    out = pairs_core.copy().reset_index(drop=True)
    mel_true = out["wse_b"].to_numpy(dtype=float)
    # H untuk g = muka air benar Melintang (sesuai prompt)
    g, sig = interp_truth(mel_true, bins)
    sem_true = mel_true + g
    n = len(out)
    if noise:
        # Student-t ν=3, skala σ(H)/√2
        scale = sig / np.sqrt(2.0)
        eps_a = rng.standard_t(student_df, size=n) * scale
        eps_b = rng.standard_t(student_df, size=n) * scale
        common = rng.normal(0.0, common_sd, size=n) if common_sd > 0 else np.zeros(n)
    else:
        eps_a = eps_b = common = np.zeros(n)
    out["wse_a_true"] = sem_true
    out["wse_b_true"] = mel_true
    out["g_H"] = g
    out["sigma_H_truth"] = sig
    out["wse_a"] = sem_true + eps_a + common
    out["wse_b"] = mel_true + eps_b + common
    out["dWSE"] = out["wse_a"] - out["wse_b"]
    out["H"] = 0.5 * (out["wse_a"] + out["wse_b"])
    out["q_a"] = 0
    out["q_b"] = 0
    out["core"] = True
    out["sudah_ditandai_flag"] = False
    if "phase" not in out.columns:
        out["phase"] = ""
    return out


def run_s1(
    pairs_all: pd.DataFrame,
    swot: pd.DataFrame,
    cfg: dict,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S1: sisip ke data nyata tanpa pra-penyaringan z/B-temp."""
    d2cfg = d2_cfg_from_d2b(cfg)
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]
    lakes = [danau_a, danau_b]
    k_d2 = [float(x) for x in cfg["detection"]["k_values"]]
    k_bt = [float(x) for x in cfg["baseline_temporal"]["k_values"]]
    deltas = [float(x) for x in cfg["synthetic"]["deltas_m"]]
    n_ulang = int(cfg["synthetic"]["n_ulang"])
    frac = float(cfg["synthetic"]["fraction_inject"])
    attr_cfg = {**cfg["attribution"], "mad_scale": cfg["reference_curve"]["mad_scale"]}

    pairs = pairs_all[pairs_all["core"]].copy().reset_index(drop=True)
    pairs["time"] = pd.to_datetime(pairs["time"], utc=True)
    n = len(pairs)
    n_inject = max(1, int(round(frac * n)))
    print(f"S1: n_core={n} n_inject/trial={n_inject} n_ulang={n_ulang}", flush=True)

    lake0 = {danau_a: swot[swot["danau"] == danau_a].copy(), danau_b: swot[swot["danau"] == danau_b].copy()}
    d2_base, _ = run_d2(pairs, lake0, d2cfg, k_values=k_d2, do_attribution=True)
    bt_base = run_btemp_all_lakes(swot, lakes, cfg["baseline_temporal"], k_values=k_bt)
    d2_base["time"] = pd.to_datetime(d2_base["time"], utc=True)
    bt_base["time"] = pd.to_datetime(bt_base["time"], utc=True)

    # jempang pair results on original (for triangulation baseline & S1)
    tri_maps = build_jempang_pair_results(swot, cfg)

    base_d2_flag = {kk: d2_base[flag_col(d2_base, kk)].astype(bool).to_numpy() for kk in k_d2}
    # B-temp base flags per pass (either lake)
    base_bt_flag = {}
    for kk in k_bt:
        fcol = flag_col(bt_base, kk)
        flags = np.zeros(n, dtype=bool)
        for i in range(n):
            t = pairs.loc[i, "time"]
            for lake in lakes:
                sub = bt_base[
                    (bt_base["danau"] == lake) & ((bt_base["time"] - t).abs() <= pd.Timedelta(seconds=120))
                ]
                if len(sub) and bool(sub.iloc[0][fcol]):
                    flags[i] = True
                    break
        base_bt_flag[kk] = flags

    rng = np.random.default_rng(seed)
    stats: dict = {}
    swot0 = swot.copy()
    swot0["time"] = pd.to_datetime(swot0["time"], utc=True)
    swot0["wse"] = pd.to_numeric(swot0["wse"], errors="coerce")

    for delta in deltas:
        for trial in range(n_ulang):
            if trial % 25 == 0:
                print(f"  S1 delta={delta} trial={trial}/{n_ulang}", flush=True)
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
                pairs_m.loc[i, "H"] = 0.5 * (float(pairs_m.loc[i, "wse_a"]) + float(pairs_m.loc[i, "wse_b"]))

            sw = swot0.copy()
            for i, lake in inj_lake.items():
                t = pairs.loc[i, "time"]
                m = (sw["danau"] == lake) & ((sw["time"] - t).abs() <= pd.Timedelta(seconds=120))
                sw.loc[m, "wse"] = sw.loc[m, "wse"] + signs[i] * delta

            lake_m = {danau_a: sw[sw["danau"] == danau_a], danau_b: sw[sw["danau"] == danau_b]}
            d2_m, _ = run_d2(pairs_m, lake_m, d2cfg, k_values=k_d2, do_attribution=False)
            bt_m = run_btemp_all_lakes(sw, lakes, cfg["baseline_temporal"], k_values=k_bt)
            d2_m["time"] = pd.to_datetime(d2_m["time"], utc=True)
            bt_m["time"] = pd.to_datetime(bt_m["time"], utc=True)
            sigma_cache = {
                name: float(bt_m.loc[bt_m["danau"] == name, "sigma_lake"].iloc[0])
                for name in lakes
                if (bt_m["danau"] == name).any()
            }
            # tri_maps disesuaikan: hanya z pasangan yang melibatkan danau tersisip digeser δ/σ
            tri_maps_m = _tri_maps_with_inject(tri_maps, pairs, inj_lake, signs, delta)

            for kk in k_d2:
                fcol = flag_col(d2_m, kk)
                flagged = d2_m[fcol].astype(bool).to_numpy()
                for band in ("H<2.5", "H>=2.5"):
                    key = ("D2", delta, kk, band)
                    for i in range(n):
                        if band_of(float(pairs.loc[i, "H"])) != band:
                            continue
                        base_f = bool(base_d2_flag[kk][i])
                        new_f = bool(flagged[i])
                        if i in inj_set:
                            _acc(stats, key, n_inject=1)
                            if new_f:
                                _acc(stats, key, n_detect=1)
                                _acc(stats, key, n_flagged_for_attr=1)
                                t = pairs.loc[i, "time"]
                                sumber, _ = attribute_source(lake_m, t, attr_cfg, sigma_cache=sigma_cache)
                                if is_named_source(sumber):
                                    _acc(stats, key, n_attr_named=1)
                                    if sumber == inj_lake[i]:
                                        _acc(stats, key, n_attr_ok=1)
                                _acc(stats, key, n_tri_flagged=1)
                                tri_src = attribute_triangulation(
                                    i, pairs_m, d2_m, tri_maps_m, cfg, kk, temporal_sumber=sumber
                                )
                                if is_named_source(tri_src):
                                    _acc(stats, key, n_tri_named=1)
                                    if tri_src == inj_lake[i]:
                                        _acc(stats, key, n_tri_ok=1)
                        else:
                            _acc(stats, key, n_noninject=1)
                            if base_f:
                                _acc(stats, key, n_base_flag=1)
                            if new_f and not base_f:
                                _acc(stats, key, n_add_fp=1)

            for kk in k_bt:
                fcol = flag_col(bt_m, kk)
                for band in ("H<2.5", "H>=2.5"):
                    key = ("Btemp", delta, kk, band)
                    for i in range(n):
                        if band_of(float(pairs.loc[i, "H"])) != band:
                            continue
                        t = pairs.loc[i, "time"]
                        # B-temp per observasi danau yang di-inject, atau kedua danau untuk FP
                        if i in inj_set:
                            lake = inj_lake[i]
                            _acc(stats, key, n_inject=1)
                            sub = bt_m[
                                (bt_m["danau"] == lake)
                                & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))
                            ]
                            if len(sub) and bool(sub.iloc[0][fcol]):
                                _acc(stats, key, n_detect=1)
                        else:
                            # per-observasi: kedua danau
                            for lake in lakes:
                                _acc(stats, key, n_noninject=1)
                                sub = bt_m[
                                    (bt_m["danau"] == lake)
                                    & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))
                                ]
                                new_f = len(sub) > 0 and bool(sub.iloc[0][fcol])
                                # base flag for this lake-obs
                                sub0 = bt_base[
                                    (bt_base["danau"] == lake)
                                    & ((bt_base["time"] - t).abs() <= pd.Timedelta(seconds=120))
                                ]
                                base_f = len(sub0) > 0 and bool(sub0.iloc[0][flag_col(bt_base, kk)])
                                if base_f:
                                    _acc(stats, key, n_base_flag=1)
                                if new_f and not base_f:
                                    _acc(stats, key, n_add_fp=1)

    metrics = _stats_to_df(stats, {"design": "S1", "n_core": n, "n_inject_per_trial": n_inject, "n_ulang": n_ulang})
    # also return baseline summary
    base_rows = []
    for kk in k_d2:
        for band in ("H<2.5", "H>=2.5"):
            m = np.array([band_of(float(pairs.loc[i, "H"])) == band for i in range(n)])
            base_rows.append(
                {
                    "method": "D2",
                    "k": kk,
                    "h_band": band,
                    "n": int(m.sum()),
                    "baseline_flag_rate": float(base_d2_flag[kk][m].mean()) if m.any() else float("nan"),
                }
            )
    for kk in k_bt:
        for band in ("H<2.5", "H>=2.5"):
            m = np.array([band_of(float(pairs.loc[i, "H"])) == band for i in range(n)])
            base_rows.append(
                {
                    "method": "Btemp",
                    "k": kk,
                    "h_band": band,
                    "n": int(m.sum()),
                    "baseline_flag_rate": float(base_bt_flag[kk][m].mean()) if m.any() else float("nan"),
                }
            )
    return metrics, pd.DataFrame(base_rows)


def run_s2(
    pairs_core: pd.DataFrame,
    swot_template: pd.DataFrame,
    cfg: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """S2: sintetis penuh dengan kebenaran diketahui + common-mode."""
    d2cfg = d2_cfg_from_d2b(cfg)
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]
    lakes = [danau_a, danau_b]
    k_d2 = [float(x) for x in cfg["detection"]["k_values"]]
    k_bt = [float(x) for x in cfg["baseline_temporal"]["k_values"]]
    deltas = [float(x) for x in cfg["synthetic"]["deltas_m"]]
    n_ulang = int(cfg["synthetic"]["n_ulang"])
    frac = float(cfg["synthetic"]["fraction_inject"])
    commons = [float(x) for x in cfg["synthetic"]["common_mode_sd_m"]]
    student_df = float(cfg["synthetic"]["student_t_df"])
    bins = truth_curve_table(cfg)
    attr_cfg = {**cfg["attribution"], "mad_scale": cfg["reference_curve"]["mad_scale"]}

    pairs0 = pairs_core[pairs_core["core"]].copy().reset_index(drop=True)
    pairs0["time"] = pd.to_datetime(pairs0["time"], utc=True)
    n = len(pairs0)
    n_inject = max(1, int(round(frac * n)))
    print(f"S2: n={n} n_inject={n_inject} n_ulang={n_ulang} c={commons}", flush=True)

    rng = np.random.default_rng(seed)
    stats: dict = {}

    for c in commons:
        for delta in deltas:
            for trial in range(n_ulang):
                if trial % 25 == 0:
                    print(f"  S2 c={c} delta={delta} trial={trial}/{n_ulang}", flush=True)
                # new noise each trial
                pairs = generate_s2_truth(pairs0, bins, rng, student_df, c, noise=True)
                inj_idx = list(rng.choice(n, size=min(n_inject, n), replace=False))
                inj_lake = {int(i): lakes[int(rng.integers(0, 2))] for i in inj_idx}
                signs = {int(i): float(rng.choice([-1.0, 1.0])) for i in inj_idx}
                inj_set = set(inj_lake)

                for i, lake in inj_lake.items():
                    s = signs[i]
                    if lake == danau_a:
                        pairs.loc[i, "wse_a"] = float(pairs.loc[i, "wse_a"]) + s * delta
                    else:
                        pairs.loc[i, "wse_b"] = float(pairs.loc[i, "wse_b"]) + s * delta
                    pairs.loc[i, "dWSE"] = float(pairs.loc[i, "wse_a"]) - float(pairs.loc[i, "wse_b"])
                    pairs.loc[i, "H"] = 0.5 * (float(pairs.loc[i, "wse_a"]) + float(pairs.loc[i, "wse_b"]))

                # synthetic swot series for B-temp / attribution: only SM lakes at pair times
                rows_sw = []
                for i in range(n):
                    t = pairs.loc[i, "time"]
                    rows_sw.append(
                        {"danau": danau_a, "time": t, "wse": float(pairs.loc[i, "wse_a"]), "quality_f": 0}
                    )
                    rows_sw.append(
                        {"danau": danau_b, "time": t, "wse": float(pairs.loc[i, "wse_b"]), "quality_f": 0}
                    )
                sw = pd.DataFrame(rows_sw)
                lake_m = {danau_a: sw[sw["danau"] == danau_a], danau_b: sw[sw["danau"] == danau_b]}
                d2_m, _ = run_d2(pairs, lake_m, d2cfg, k_values=k_d2, do_attribution=False)
                bt_m = run_btemp_all_lakes(sw, lakes, cfg["baseline_temporal"], k_values=k_bt)
                d2_m["time"] = pd.to_datetime(d2_m["time"], utc=True)
                bt_m["time"] = pd.to_datetime(bt_m["time"], utc=True)
                sigma_cache = {
                    name: float(bt_m.loc[bt_m["danau"] == name, "sigma_lake"].iloc[0])
                    for name in lakes
                    if (bt_m["danau"] == name).any()
                }
                attr_cache: dict = {}

                for kk in k_d2:
                    fcol = flag_col(d2_m, kk)
                    flagged = d2_m[fcol].astype(bool).to_numpy()
                    for band in ("H<2.5", "H>=2.5"):
                        key = ("D2", c, delta, kk, band)
                        for i in range(n):
                            h_band_val = float(pairs0.loc[i, "H"])
                            if band_of(h_band_val) != band:
                                continue
                            new_f = bool(flagged[i])
                            if i in inj_set:
                                _acc(stats, key, n_inject=1)
                                if new_f:
                                    _acc(stats, key, n_detect=1)
                                    _acc(stats, key, n_flagged_for_attr=1)
                                    t = pairs.loc[i, "time"]
                                    if t not in attr_cache:
                                        attr_cache[t], _ = attribute_source(
                                            lake_m, t, attr_cfg, sigma_cache=sigma_cache
                                        )
                                    sumber = attr_cache[t]
                                    if is_named_source(sumber):
                                        _acc(stats, key, n_attr_named=1)
                                        if sumber == inj_lake[i]:
                                            _acc(stats, key, n_attr_ok=1)
                            else:
                                _acc(stats, key, n_noninject=1)
                                if new_f:
                                    _acc(stats, key, n_true_fp=1)

                for kk in k_bt:
                    fcol = flag_col(bt_m, kk)
                    for band in ("H<2.5", "H>=2.5"):
                        key = ("Btemp", c, delta, kk, band)
                        for i in range(n):
                            h_band_val = float(pairs0.loc[i, "H"])
                            if band_of(h_band_val) != band:
                                continue
                            t = pairs.loc[i, "time"]
                            if i in inj_set:
                                lake = inj_lake[i]
                                _acc(stats, key, n_inject=1)
                                sub = bt_m[
                                    (bt_m["danau"] == lake)
                                    & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))
                                ]
                                if len(sub) and bool(sub.iloc[0][fcol]):
                                    _acc(stats, key, n_detect=1)
                            else:
                                for lake in lakes:
                                    _acc(stats, key, n_noninject=1)
                                    sub = bt_m[
                                        (bt_m["danau"] == lake)
                                        & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))
                                    ]
                                    if len(sub) and bool(sub.iloc[0][fcol]):
                                        _acc(stats, key, n_true_fp=1)

    return _stats_to_df(
        stats, {"design": "S2", "n_core": n, "n_inject_per_trial": n_inject, "n_ulang": n_ulang}
    )
