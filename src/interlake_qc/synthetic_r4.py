"""Uji sintetis D2 vs B-temp (Fase 5 MethodsX)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from interlake_qc.baseline_temporal import run_btemp_all_lakes
from interlake_qc.d2_interlake import attribute_source, prepare_pair_table, run_d2


def _flag_col(df: pd.DataFrame, k: float) -> str:
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


def build_clean_set(
    pairs: pd.DataFrame,
    d2_res: pd.DataFrame,
    btemp: pd.DataFrame,
    clean_k: float = 2.5,
    danau_a: str = "semayang",
    danau_b: str = "melintang",
) -> pd.DataFrame:
    """Himpunan bersih: core, |z|≤clean_k pada D2, kedua danau tidak ditandai B-temp pada clean_k."""
    d2 = d2_res.copy()
    d2["time"] = pd.to_datetime(d2["time"], utc=True)
    clean_d2 = d2[d2["core"] & np.isfinite(d2["z"]) & (d2["z"].abs() <= clean_k)]

    bt = btemp.copy()
    bt["time"] = pd.to_datetime(bt["time"], utc=True)
    flag_col = _flag_col(bt, clean_k)
    if flag_col not in bt.columns:
        raise KeyError(f"kolom B-temp {flag_col} tidak ada")

    flagged_times = set()
    for name in (danau_a, danau_b):
        sub = bt[(bt["danau"] == name) & bt[flag_col].astype(bool)]
        for t in sub["time"]:
            flagged_times.add(pd.Timestamp(t))

    rows = []
    for _, r in clean_d2.iterrows():
        t = pd.Timestamp(r["time"])
        if any(abs((ft - t).total_seconds()) <= 120 for ft in flagged_times):
            continue
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


def run_synthetic_trials(
    pairs_base: pd.DataFrame,
    swot: pd.DataFrame,
    cfg: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """n_ulang × δ: sisip galat, hitung recall / FPR / atribusi."""
    syn = cfg["synthetic"]
    n_ulang = int(syn["n_ulang"])
    frac = float(syn["fraction_inject"])
    deltas = [float(x) for x in syn["deltas_m"]]
    k_d2 = [float(x) for x in cfg["detection"]["k_values"]]
    k_bt = [float(x) for x in cfg["baseline_temporal"]["k_values"]]
    danau_a = cfg["pair"]["danau_a"]
    danau_b = cfg["pair"]["danau_b"]
    clean_k = float(cfg["detection"]["clean_k"])
    attr_cfg = {**cfg["attribution"], "mad_scale": cfg["reference_curve"]["mad_scale"]}

    lake_series0 = {
        danau_a: swot[swot["danau"] == danau_a].copy(),
        danau_b: swot[swot["danau"] == danau_b].copy(),
    }
    d2_real, _ = run_d2(pairs_base, lake_series0, cfg, k_values=k_d2, do_attribution=False)
    btemp_real = run_btemp_all_lakes(swot, [danau_a, danau_b], cfg["baseline_temporal"], k_values=k_bt)
    clean = build_clean_set(pairs_base, d2_real, btemp_real, clean_k, danau_a, danau_b)
    if len(clean) < 10:
        raise RuntimeError(f"himpunan bersih terlalu kecil: n={len(clean)}")

    pairs_base = pairs_base.copy().reset_index(drop=True)
    pairs_base["time"] = pd.to_datetime(pairs_base["time"], utc=True)
    clean_idx = []
    for _, cr in clean.iterrows():
        dt = (pairs_base["time"] - pd.Timestamp(cr["time"])).abs()
        j = int(dt.idxmin())
        if dt.loc[j] <= pd.Timedelta(seconds=120):
            clean_idx.append(j)
    clean_idx = sorted(set(clean_idx))
    n_clean = len(clean_idx)
    n_inject = max(1, int(round(frac * n_clean)))
    lakes = [danau_a, danau_b]
    rng = np.random.default_rng(seed)

    stats: dict[tuple, dict[str, float]] = {}

    def band_of(h: float) -> str:
        return "H<2.5" if h < 2.5 else "H>=2.5"

    def acc(key, **kw):
        if key not in stats:
            stats[key] = {
                "n_inject": 0,
                "n_detect": 0,
                "n_clean_eval": 0,
                "n_fp": 0,
                "n_attr_ok": 0,
                "n_attr_denom": 0,
            }
        for k, v in kw.items():
            stats[key][k] += v

    print(
        f"synthetic: n_clean={n_clean} n_inject/trial={n_inject} n_ulang={n_ulang} deltas={deltas}",
        flush=True,
    )

    # base wse snapshots for fast restore
    swot0 = swot.copy()
    swot0["time"] = pd.to_datetime(swot0["time"], utc=True)
    swot0["wse"] = pd.to_numeric(swot0["wse"], errors="coerce")

    for delta in deltas:
        for trial in range(n_ulang):
            if trial % 25 == 0:
                print(f"  delta={delta} trial={trial}/{n_ulang}", flush=True)
            inject_pos = list(rng.choice(clean_idx, size=min(n_inject, n_clean), replace=False))
            inject_lake = {int(i): lakes[int(rng.integers(0, 2))] for i in inject_pos}
            signs = {int(i): float(rng.choice([-1.0, 1.0])) for i in inject_pos}

            # mutate pairs in place (faster than rebuild)
            pairs = pairs_base.copy()
            for i, lake in inject_lake.items():
                sign = signs[i]
                if lake == danau_a:
                    pairs.loc[i, "wse_a"] = float(pairs.loc[i, "wse_a"]) + sign * delta
                else:
                    pairs.loc[i, "wse_b"] = float(pairs.loc[i, "wse_b"]) + sign * delta
                pairs.loc[i, "dWSE"] = float(pairs.loc[i, "wse_a"]) - float(pairs.loc[i, "wse_b"])
                pairs.loc[i, "H"] = 0.5 * (float(pairs.loc[i, "wse_a"]) + float(pairs.loc[i, "wse_b"]))

            # mutate swot for B-temp + attribution
            sw = swot0.copy()
            for i, lake in inject_lake.items():
                t = pairs_base.loc[i, "time"]
                sign = signs[i]
                m = (sw["danau"] == lake) & ((sw["time"] - t).abs() <= pd.Timedelta(seconds=120))
                sw.loc[m, "wse"] = sw.loc[m, "wse"] + sign * delta

            lake_series = {
                danau_a: sw[sw["danau"] == danau_a],
                danau_b: sw[sw["danau"] == danau_b],
            }
            d2_m, _ = run_d2(pairs, lake_series, cfg, k_values=k_d2, do_attribution=False)
            bt_m = run_btemp_all_lakes(sw, lakes, cfg["baseline_temporal"], k_values=k_bt)

            d2_m["time"] = pd.to_datetime(d2_m["time"], utc=True)
            bt_m["time"] = pd.to_datetime(bt_m["time"], utc=True)
            inject_times = {pd.Timestamp(pairs_base.loc[i, "time"]): inject_lake[i] for i in inject_lake}
            inject_h = {pd.Timestamp(pairs_base.loc[i, "time"]): float(pairs_base.loc[i, "H"]) for i in inject_lake}
            sigma_cache = {
                name: float(bt_m.loc[bt_m["danau"] == name, "sigma_lake"].iloc[0])
                for name in lakes
                if (bt_m["danau"] == name).any()
            }
            attr_cache: dict = {}

            for kk in k_d2:
                fcol = _flag_col(d2_m, kk)
                for band in ("H<2.5", "H>=2.5"):
                    inj_band = [t for t, h in inject_h.items() if band_of(h) == band]
                    clean_band_idx = [
                        j
                        for j in clean_idx
                        if band_of(float(pairs_base.loc[j, "H"])) == band and j not in inject_lake
                    ]
                    key = ("D2", delta, kk, band)
                    acc(key, n_inject=len(inj_band), n_clean_eval=len(clean_band_idx))

                    for t in inj_band:
                        dt = (d2_m["time"] - t).abs()
                        j = int(dt.idxmin())
                        if dt.loc[j] > pd.Timedelta(seconds=120):
                            continue
                        if bool(d2_m.loc[j, fcol]):
                            acc(key, n_detect=1)
                            if t not in attr_cache:
                                attr_cache[t], _ = attribute_source(
                                    lake_series, t, attr_cfg, sigma_cache=sigma_cache
                                )
                            if attr_cache[t] == inject_times[t]:
                                acc(key, n_attr_ok=1)
                            acc(key, n_attr_denom=1)

                    for j0 in clean_band_idx:
                        t = pd.Timestamp(pairs_base.loc[j0, "time"])
                        dt = (d2_m["time"] - t).abs()
                        j = int(dt.idxmin())
                        if dt.loc[j] > pd.Timedelta(seconds=120):
                            continue
                        if bool(d2_m.loc[j, fcol]):
                            acc(key, n_fp=1)

            for kk in k_bt:
                fcol = _flag_col(bt_m, kk)
                for band in ("H<2.5", "H>=2.5"):
                    key = ("Btemp", delta, kk, band)
                    n_inj = n_det = n_clean_eval = n_fp = 0
                    for i, lake in inject_lake.items():
                        h = float(pairs_base.loc[i, "H"])
                        if band_of(h) != band:
                            continue
                        n_inj += 1
                        t = pd.Timestamp(pairs_base.loc[i, "time"])
                        sub = bt_m[(bt_m["danau"] == lake) & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))]
                        if len(sub) and bool(sub.iloc[0][fcol]):
                            n_det += 1
                    for j0 in clean_idx:
                        if j0 in inject_lake:
                            continue
                        h = float(pairs_base.loc[j0, "H"])
                        if band_of(h) != band:
                            continue
                        t = pd.Timestamp(pairs_base.loc[j0, "time"])
                        for lake in lakes:
                            n_clean_eval += 1
                            sub = bt_m[
                                (bt_m["danau"] == lake) & ((bt_m["time"] - t).abs() <= pd.Timedelta(seconds=120))
                            ]
                            if len(sub) and bool(sub.iloc[0][fcol]):
                                n_fp += 1
                    acc(key, n_inject=n_inj, n_detect=n_det, n_clean_eval=n_clean_eval, n_fp=n_fp)

    rows = []
    for (method, delta, kk, band), st in sorted(stats.items()):
        recall = st["n_detect"] / st["n_inject"] if st["n_inject"] else float("nan")
        fpr = st["n_fp"] / st["n_clean_eval"] if st["n_clean_eval"] else float("nan")
        attr = st["n_attr_ok"] / st["n_attr_denom"] if st["n_attr_denom"] else float("nan")
        rows.append(
            {
                "method": method,
                "delta_m": delta,
                "k": kk,
                "h_band": band,
                "n_inject": st["n_inject"],
                "n_detect": st["n_detect"],
                "recall": recall,
                "n_clean_eval": st["n_clean_eval"],
                "n_fp": st["n_fp"],
                "false_positive_rate": fpr,
                "n_attr_ok": st["n_attr_ok"],
                "n_attr_denom": st["n_attr_denom"],
                "attribution_accuracy": attr if method == "D2" else "",
                "n_clean_base": n_clean,
                "n_inject_per_trial": n_inject,
                "n_ulang": n_ulang,
            }
        )
    return pd.DataFrame(rows)
