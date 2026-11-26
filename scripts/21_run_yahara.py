"""R7b — validasi D2 Yahara (mendukung amendment_1: P2 utama)."""

from __future__ import annotations

import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

from interlake_qc.baseline_temporal import run_btemp_all_lakes  # noqa: E402
from interlake_qc.core import interp_f_sigma, load_d2_config  # noqa: E402
from interlake_qc.validation_yahara import (  # noqa: E402
    attribution_rows,
    build_pair_table,
    compute_bias_and_ec,
    d_preferred_else_c,
    dedup_prefer_D,
    filter_version,
    label_truth,
    load_r7_config,
    load_yahara_raw,
    main_set_mask,
    make_d2_cfg,
    map_sumber_to_side,
    method_flags_on_pairs,
    metrics_block,
    rmse_mad_by_lake_variants,
    to_swot_like,
    wilson_ci,
)


def active_pairs(r7: dict) -> dict:
    a1 = r7.get("amendment_1") or {}
    if a1.get("pairs"):
        return a1["pairs"]
    return r7["pairs"]


def plot_dwse(res: pd.DataFrame, bins: pd.DataFrame, cfg_det: dict, pair_name: str, out: Path, T: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    p = res.copy()
    if "bermasalah" not in p.columns:
        p["bermasalah"] = False
    ok = p[~p["bermasalah"].astype(bool)]
    bad = p[p["bermasalah"].astype(bool)]
    ax.scatter(ok["H"], ok["dWSE"], c="#4C78A8", s=28, alpha=0.8, label="tidak bermasalah")
    ax.scatter(bad["H"], bad["dWSE"], c="#E45756", s=36, alpha=0.9, label="bermasalah")
    fl = p[p["flag"].astype(bool)]
    ax.scatter(fl["H"], fl["dWSE"], facecolors="none", edgecolors="k", s=80, linewidths=1.2, label="D2 flag")
    if len(bins):
        hs = np.linspace(float(np.nanmin(p["H"])) - 0.05, float(np.nanmax(p["H"])) + 0.05, 200)
        k = float(cfg_det["k_high"])
        f_line, s_line = [], []
        for h in hs:
            f, s = interp_f_sigma(h, bins)
            f_line.append(f)
            s_line.append(s)
        f_arr, s_arr = np.asarray(f_line), np.asarray(s_line)
        ax.plot(hs, f_arr, "k-", lw=1.1, label="f(H)")
        ax.fill_between(hs, f_arr - k * s_arr, f_arr + k * s_arr, color="gray", alpha=0.25, label=f"f±{k}σ")
    ax.set_xlabel("H (m)")
    ax.set_ylabel("dWSE (m)")
    ax.set_title(f"D2 {pair_name} (T={T})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"r7b_dwse_vs_H_{pair_name}.png", dpi=150)
    plt.close(fig)


def plot_e_timeseries(gage: pd.DataFrame, d2_drops: dict[str, set], out: Path, lakes: list[str] | None = None) -> None:
    lakes = lakes or sorted(gage["danau"].dropna().unique())
    fig, axes = plt.subplots(len(lakes), 1, figsize=(10, 2.4 * len(lakes)), sharex=True)
    if len(lakes) == 1:
        axes = [axes]
    for ax, danau in zip(axes, lakes):
        sub = gage[(gage["danau"] == danau) & gage["truth_ok"]].sort_values("SWOT_time")
        ax.plot(sub["SWOT_time"], sub["e_c"], ".", color="#4C78A8", ms=4, label="e_c")
        drops = d2_drops.get(danau, set())
        if drops and len(sub):
            m = sub["SWOT_time"].map(
                lambda t: any(abs((pd.Timestamp(t) - ft).total_seconds()) <= 120 for ft in drops)
            )
            ax.plot(sub.loc[m, "SWOT_time"], sub.loc[m, "e_c"], "x", color="#E45756", ms=7, label="dibuang D2")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(danau)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time")
    fig.suptitle("e_c per danau (× = dibuang D2 attr)")
    fig.tight_layout()
    fig.savefig(out / "r7b_e_timeseries.png", dpi=150)
    plt.close(fig)


def plot_z_vs_edwse(all_merged: pd.DataFrame, k: float, T: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    p = all_merged[all_merged["both_truth"]].copy()
    ax.scatter(
        p["e_dwse_c"].abs(),
        p["z"].abs(),
        c=p["bermasalah"].map({True: "#E45756", False: "#4C78A8"}),
        s=28,
        alpha=0.8,
    )
    ax.axhline(k, color="k", ls="--", label=f"k={k}")
    ax.axvline(T, color="orange", ls="--", label=f"T={T}")
    ax.set_xlabel("|e_dwse_c| (m)")
    ax.set_ylabel("|z| D2")
    ax.set_title("D2 |z| vs |e_dwse_c|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "r7b_z_vs_edwse.png", dpi=150)
    plt.close(fig)


def run_one_d2(pairs, swot, base_d2, pair_spec, det, do_attr=True):
    from interlake_qc.core import run_d2_final

    cfg = make_d2_cfg(base_d2, pair_spec, det, {})
    # temporal hanya A/B; danau_tri tetap tersedia via swot untuk triangulasi
    lakes = {
        pair_spec["danau_a"]: swot[swot["danau"] == pair_spec["danau_a"]],
        pair_spec["danau_b"]: swot[swot["danau"] == pair_spec["danau_b"]],
    }
    pairs_run = pairs.copy()
    pairs_run["core"] = pairs_run["main"].astype(bool)
    t0 = time.perf_counter()
    res, bins = run_d2_final(pairs_run, lakes, cfg, do_attribution=do_attr, swot=swot if do_attr else None)
    return res, bins, time.perf_counter() - t0


def merge_d2_truth(res, pairs, T, danau_a, danau_b):
    p = label_truth(pairs, T)
    r = res.copy()
    r["time"] = pd.to_datetime(r["time"], utc=True)
    p["time"] = pd.to_datetime(p["time"], utc=True)
    cols = [
        "e_c_a", "e_c_b", "e_dwse_c", "e_dwse", "both_truth", "bermasalah", "guilty_side",
        "gage_a", "gage_b", "dark_a", "dark_b", "ice_a", "ice_b", "q_a", "q_b", "main", "core",
    ]
    out_rows = []
    for i in range(len(r)):
        t = r.iloc[i]["time"]
        dt = (p["time"] - t).abs()
        j = int(dt.idxmin())
        row = r.iloc[i].to_dict()
        if dt.loc[j] <= pd.Timedelta(seconds=120):
            for c in cols:
                if c in p.columns:
                    row[c] = p.loc[j, c]
            gs = p.loc[j, "guilty_side"]
            row["guilty_lake"] = danau_a if gs == "a" else (danau_b if gs == "b" else gs)
        else:
            for c in cols:
                row[c] = np.nan if c not in ("both_truth", "bermasalah", "main", "core") else False
            row["guilty_lake"] = ""
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def align_flags(pairs, res):
    flag_series = pd.Series(False, index=pairs.index)
    mt = pd.to_datetime(res["time"], utc=True)
    for i, t in enumerate(pd.to_datetime(pairs["time"], utc=True)):
        dt = (mt - t).abs()
        j = int(dt.idxmin())
        if dt.loc[j] <= pd.Timedelta(seconds=120):
            flag_series.iloc[i] = bool(res.loc[j, "flag"])
    return flag_series


def metrics_row_id(r: dict, peran: str, himpunan: str, versi: str) -> dict:
    """Ubah keluaran metrics_block ke skema kolom amandemen."""
    return {
        "pasangan": r["pair"],
        "peran": peran,
        "k": r["k"],
        "T": r["T"],
        "himpunan": himpunan,
        "versi": versi,
        "metode": r["method"],
        "n_pass": r["n_truth"],
        "n_bermasalah": r["n_bad"],
        "n_ditandai": r["n_flagged"],
        "presisi": r["precision"],
        "presisi_lo": r["precision_lo"],
        "presisi_hi": r["precision_hi"],
        "recall": r["recall"],
        "recall_lo": r["recall_lo"],
        "recall_hi": r["recall_hi"],
        "fpr": r["fpr"],
        "fpr_lo": r["fpr_lo"],
        "fpr_hi": r["fpr_hi"],
        "frac_dipertahankan": r["frac_retained"],
        "prevalence": r.get("prevalence"),
        "tp": r["tp"],
        "fp": r["fp"],
        "fn": r["fn"],
        "tn": r["tn"],
    }


def tri_coverage(pairs_main: pd.DataFrame, gage: pd.DataFrame, tri_lake: str, max_dt: float = 120.0) -> dict:
    """Berapa pass utama P2 yang juga punya danau tri pada cycle+pass sama, q01, ≤120s."""
    g = gage[gage["danau"] == tri_lake].copy()
    g["_key"] = g["cycle_id"].astype(str) + "|" + g["pass_id"].astype(str)
    n_main = int(pairs_main["main"].sum()) if len(pairs_main) else 0
    n_hit = 0
    for _, r in pairs_main[pairs_main["main"]].iterrows():
        key = f"{r['cycle_id']}|{r['pass']}"
        cand = g[g["_key"] == key]
        if cand.empty:
            continue
        dt = (cand["SWOT_time"] - pd.Timestamp(r["time"])).abs()
        j = dt.idxmin()
        if dt.loc[j] > pd.Timedelta(seconds=max_dt):
            continue
        if cand.loc[j, "quality_f"] in (0, 1):
            n_hit += 1
    return {"n_main": n_main, "n_tri_q01": n_hit, "frac": n_hit / n_main if n_main else float("nan")}


def mad_arr(x):
    from interlake_qc.core import mad

    return mad(np.asarray(x, dtype=float))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yahara.yaml")
    args, _ = ap.parse_known_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    np.random.seed(42)
    r7 = load_r7_config(REPO, args.config)
    a1 = r7.get("amendment_1") or {}
    base_d2 = load_d2_config(REPO, r7["paths"]["d2_final"])
    out = REPO / r7["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    # jejak STOP v0
    stop = out / "r7b_STOP.csv"
    if stop.exists() and not (out / "r7b_STOP_v0.csv").exists():
        shutil.move(str(stop), str(out / "r7b_STOP_v0.csv"))
    elif stop.exists():
        stop.unlink()

    pairs_cfg = active_pairs(r7)
    primary_id = a1.get("primary_pair", "P2")
    stop_id = a1.get("stop_pair", primary_id)
    min_stop = int(a1.get("min_primary_main_with_truth", r7.get("min_p1_main_with_truth", 40)))
    print(f"amendment_1 active={bool(a1)} primary={primary_id} tri_P2={pairs_cfg[primary_id]['danau_tri']}", flush=True)

    print("=== MUAT ===", flush=True)
    gage0, sites, counts_pre = load_yahara_raw(REPO, r7)
    gage_dedup, dedup_stats = dedup_prefer_D(gage0)
    # kebijakan versi cadangan (sudah aktif dari R7b)
    gage_fb = d_preferred_else_c(gage_dedup)
    main_m = main_set_mask(gage_fb, require_ice0=True)
    gage_use = compute_bias_and_ec(gage_fb, float(r7["truth"]["max_gage_dt_hours"]), main_mask=main_m)
    versi_label = "D_preferred_else_C"

    # D-only for sensitivity
    gage_D = filter_version(gage_dedup, "D")
    gage_D = compute_bias_and_ec(
        gage_D, float(r7["truth"]["max_gage_dt_hours"]), main_mask=main_set_mask(gage_D, True)
    )

    # no-ice sensitivity on fallback version policy
    gage_noice = compute_bias_and_ec(
        gage_fb, float(r7["truth"]["max_gage_dt_hours"]), main_mask=main_set_mask(gage_fb, True)
    )

    max_dt = float(r7["pair_match"]["max_dt_sec"])
    T_main = float(r7["truth"]["T_main"])
    T_list = [T_main] + [float(x) for x in r7["truth"]["T_sens"]]
    det_main = r7["detection_main"]
    det_k4 = r7["detection_sens_k4"]
    ketat = r7["ketat"]

    # pair tables
    pair_tables = {}
    for pid, spec in pairs_cfg.items():
        pair_tables[pid] = build_pair_table(
            gage_use, spec["danau_a"], spec["danau_b"], max_dt, require_ice0=True
        )

    # P2 checks + tri coverage + D-only counts
    p2 = pair_tables[primary_id]
    p2_spec = pairs_cfg[primary_id]
    print("=== CEK P2 ===", flush=True)
    n_all, n_q01 = len(p2), int(p2["core"].sum()) if len(p2) else 0
    n_main = int(p2["main"].sum()) if len(p2) else 0
    n_truth = int((p2["main"] & p2["both_truth"]).sum()) if len(p2) else 0
    print(f"P2 all={n_all} q01={n_q01} main={n_main} main+truth={n_truth}", flush=True)
    cov = tri_coverage(p2, gage_use, p2_spec["danau_tri"], max_dt)
    print(f"tri coverage ({p2_spec['danau_tri']}): {cov}", flush=True)

    p2_D = build_pair_table(gage_D, p2_spec["danau_a"], p2_spec["danau_b"], max_dt, require_ice0=True)
    n_truth_D = int((p2_D["main"] & p2_D["both_truth"]).sum()) if len(p2_D) else 0
    print(
        f"P2 D-only: all={len(p2_D)} q01={int(p2_D['core'].sum())} main={int(p2_D['main'].sum())} "
        f"main+truth={n_truth_D}",
        flush=True,
    )

    counts = []
    for pid, pt in pair_tables.items():
        counts.append(
            {
                "pair": pid,
                "name": pairs_cfg[pid]["name"],
                "peran": pairs_cfg[pid].get("peran", ""),
                "n_all": len(pt),
                "n_q01": int(pt["core"].sum()) if len(pt) else 0,
                "n_main": int(pt["main"].sum()) if len(pt) else 0,
                "n_main_truth": int((pt["main"] & pt["both_truth"]).sum()) if len(pt) else 0,
            }
        )
        pt.to_csv(out / f"r7b_pairs_{pairs_cfg[pid]['name']}.csv", index=False)
    pd.DataFrame(counts).to_csv(out / "r7b_pass_counts.csv", index=False)
    pd.DataFrame([cov]).to_csv(out / "r7b_tri_coverage_P2.csv", index=False)
    dedup_stats.to_csv(out / "r7b_dedup_stats.csv", index=False)
    counts_pre.to_csv(out / "r7b_version_counts_pre.csv", index=False)

    if n_truth < min_stop:
        print(f"STOP: {stop_id} main+truth={n_truth} < {min_stop}", flush=True)
        pd.DataFrame([{"stop": True, "n": n_truth, "threshold": min_stop}]).to_csv(out / "r7b_STOP.csv", index=False)
        return 1

    swot = to_swot_like(gage_use)
    swot_D = to_swot_like(gage_D)
    swot_noice = to_swot_like(gage_noice)

    print("=== B-temp ===", flush=True)
    btemp = run_btemp_all_lakes(swot, list(r7["lakes"].keys()), base_d2["baseline_temporal"], k_values=[3.5])
    btemp.to_csv(out / "r7b_btemp.csv", index=False)
    bt_times: dict[str, set] = {}
    if len(btemp) and "flag_k3.5" in btemp.columns:
        bt = btemp.copy()
        bt["time"] = pd.to_datetime(bt["time"], utc=True)
        for danau, sub in bt.groupby("danau"):
            bt_times[danau] = set(
                pd.Timestamp(t) for t in sub.loc[sub["flag_k3.5"].astype(bool), "time"]
            )

    metric_rows: list[dict] = []
    attr_rows: list[dict] = []
    flagged_rows: list[dict] = []
    missed_rows: list[dict] = []
    d2_time_total = 0.0
    d2_merged_store = {}
    d2_drops: dict[str, set] = {d: set() for d in r7["lakes"]}

    def process_pair(
        pid: str,
        pairs: pd.DataFrame,
        sw: pd.DataFrame,
        det: dict,
        peran: str,
        himpunan: str,
        versi: str,
        do_attr: bool,
        save_main_outputs: bool,
        k_label: float,
        T_values: list[float],
    ):
        nonlocal d2_time_total
        spec = pairs_cfg[pid]
        res, bins, elapsed = run_one_d2(pairs, sw, base_d2, spec, det, do_attr=do_attr)
        d2_time_total += elapsed
        if save_main_outputs:
            bins.to_csv(out / f"r7b_bins_{spec['name']}.csv", index=False)
            res_m = merge_d2_truth(res, pairs, T_main, spec["danau_a"], spec["danau_b"])
            res_m.to_csv(out / f"r7b_d2_{spec['name']}.csv", index=False)
            d2_merged_store[pid] = res_m
            plot_dwse(res_m[res_m["main"] | res_m["flag"]], bins, det, spec["name"], out, T_main)
        else:
            res_m = merge_d2_truth(res, pairs, T_main, spec["danau_a"], spec["danau_b"])
            tag = f"{spec['name']}_{peran}_k{k_label}".replace(".", "")
            res_m.to_csv(out / f"r7b_d2_{tag}.csv", index=False)

        flag_series = align_flags(pairs, res)
        masks = method_flags_on_pairs(pairs, flag_series, bt_times, spec["danau_a"], spec["danau_b"], ketat)
        for T in T_values:
            for raw in metrics_block(pairs, masks, T, k_label, spec["name"], require_main=True):
                metric_rows.append(metrics_row_id(raw, peran, himpunan, versi))
            merged_T = merge_d2_truth(res, pairs, T, spec["danau_a"], spec["danau_b"])
            attr_rows.extend(
                attribution_rows(merged_T, spec["danau_a"], spec["danau_b"], T, spec["name"], k_label, himpunan=himpunan)
            )

        if save_main_outputs and peran == "utama":
            for _, r in res_m[res_m["flag"].astype(bool)].iterrows():
                flagged_rows.append(
                    {
                        "tanggal": pd.Timestamp(r["time"]).strftime("%Y-%m-%d"),
                        "pair": spec["name"],
                        "H": r.get("H"),
                        "dWSE": r.get("dWSE"),
                        "z": r.get("z"),
                        "k": r.get("k_dipakai"),
                        "sumber_akhir": r.get("sumber_akhir"),
                        "cara_atribusi": r.get("cara_atribusi"),
                        "e_c_a": r.get("e_c_a"),
                        "e_c_b": r.get("e_c_b"),
                        "e_dwse_c": r.get("e_dwse_c"),
                        "q_a": r.get("q_a"),
                        "q_b": r.get("q_b"),
                        "dark_a": r.get("dark_a"),
                        "dark_b": r.get("dark_b"),
                        "ice_a": r.get("ice_a"),
                        "ice_b": r.get("ice_b"),
                        "bermasalah": r.get("bermasalah"),
                        "guilty_lake": r.get("guilty_lake"),
                    }
                )
                t = pd.Timestamp(r["time"])
                side = map_sumber_to_side(str(r.get("sumber_akhir", "")), spec["danau_a"], spec["danau_b"])
                if side == "a":
                    d2_drops[spec["danau_a"]].add(t)
                elif side == "b":
                    d2_drops[spec["danau_b"]].add(t)
                else:
                    d2_drops[spec["danau_a"]].add(t)
                    d2_drops[spec["danau_b"]].add(t)

            m15 = label_truth(pairs, T_main)
            mt = pd.to_datetime(res["time"], utc=True)
            for i, r in m15.iterrows():
                if not (r["main"] and r["both_truth"] and r["bermasalah"]):
                    continue
                if bool(flag_series.loc[i]):
                    continue
                t = pd.Timestamp(r["time"])
                dt = (mt - t).abs()
                j = int(dt.idxmin())
                z = res.loc[j, "z"] if dt.loc[j] <= pd.Timedelta(seconds=120) else np.nan
                missed_rows.append(
                    {
                        "tanggal": t.strftime("%Y-%m-%d"),
                        "pair": spec["name"],
                        "H": r["H"],
                        "dWSE": r["dWSE"],
                        "z": z,
                        "e_dwse_c": r["e_dwse_c"],
                        "e_c_a": r["e_c_a"],
                        "e_c_b": r["e_c_b"],
                        "q_a": r["q_a"],
                        "q_b": r["q_b"],
                        "dark_a": r["dark_a"],
                        "dark_b": r["dark_b"],
                        "ice_a": r["ice_a"],
                        "ice_b": r["ice_b"],
                    }
                )
        return res, bins

    # (a) UTAMA P2
    print("=== D2 UTAMA P2 k=3.5 ===", flush=True)
    process_pair(
        primary_id, pair_tables[primary_id], swot, det_main,
        peran="utama", himpunan="utama_ice0_q01", versi=versi_label,
        do_attr=True, save_main_outputs=True, k_label=3.5, T_values=T_list,
    )

    # deskriptif P2
    p2t = label_truth(pair_tables[primary_id], T_main)
    base = p2t[p2t["main"] & p2t["both_truth"]]
    desc = {
        "median_e_c_waubesa": float(np.nanmedian(base["e_c_a"])) if len(base) else "",
        "mad_e_c_waubesa": mad_arr(base["e_c_a"]) if len(base) else "",
        "median_e_c_kegonsa": float(np.nanmedian(base["e_c_b"])) if len(base) else "",
        "mad_e_c_kegonsa": mad_arr(base["e_c_b"]) if len(base) else "",
        "median_e_dwse_c": float(np.nanmedian(base["e_dwse_c"])) if len(base) else "",
        "mad_e_dwse_c": mad_arr(base["e_dwse_c"]) if len(base) else "",
        "n": len(base),
    }
    pd.DataFrame([desc]).to_csv(out / "r7b_descriptive_P2.csv", index=False)
    print("deskriptif P2:", desc, flush=True)

    # (b) sensitivitas P2
    print("=== sensitivitas P2 k=4.0 ===", flush=True)
    process_pair(
        primary_id, pair_tables[primary_id], swot, det_k4,
        peran="sensitivitas", himpunan="utama_ice0_q01", versi=versi_label,
        do_attr=True, save_main_outputs=False, k_label=4.0, T_values=T_list,
    )

    print("=== sensitivitas P2 tanpa syarat es ===", flush=True)
    p2_ni = build_pair_table(
        gage_noice, p2_spec["danau_a"], p2_spec["danau_b"], max_dt, require_ice0=False
    )
    p2_ni["main"] = p2_ni["core"]
    process_pair(
        primary_id, p2_ni, swot_noice, det_main,
        peran="sensitivitas", himpunan="q01_noice", versi=versi_label,
        do_attr=False, save_main_outputs=False, k_label=3.5, T_values=T_list,
    )

    print("=== sensitivitas P2 versi D saja ===", flush=True)
    process_pair(
        primary_id, p2_D, swot_D, det_main,
        peran="sensitivitas", himpunan="utama_ice0_q01", versi="D_only",
        do_attr=True, save_main_outputs=False, k_label=3.5, T_values=T_list,
    )

    # (c) objek kecil P1 & P3
    for pid in ("P1", "P3"):
        print(f"=== objek_kecil {pid} ===", flush=True)
        process_pair(
            pid, pair_tables[pid], swot, det_main,
            peran="objek_kecil", himpunan="utama_ice0_q01", versi=versi_label,
            do_attr=True, save_main_outputs=True, k_label=3.5, T_values=[T_main],
        )
        process_pair(
            pid, pair_tables[pid], swot, det_k4,
            peran="objek_kecil", himpunan="utama_ice0_q01", versi=versi_label,
            do_attr=True, save_main_outputs=False, k_label=4.0, T_values=[T_main],
        )

    # RMSE V1–V4 — fokus waubesa/kegonsa (+ semua danau untuk kelengkapan)
    # Only use primary P2 merged for V3 drops on waubesa/kegonsa; still include P1/P3 store
    rmse = rmse_mad_by_lake_variants(gage_use, d2_merged_store, pairs_cfg, btemp, ketat)
    rmse.to_csv(out / "r7b_rmse_by_lake.csv", index=False)

    pd.DataFrame(metric_rows).to_csv(out / "r7b_metrics.csv", index=False)
    pd.DataFrame(attr_rows).to_csv(out / "r7b_attribution.csv", index=False)
    pd.DataFrame(flagged_rows).to_csv(out / "r7b_flagged.csv", index=False)
    pd.DataFrame(missed_rows).to_csv(out / "r7b_missed.csv", index=False)

    if primary_id in d2_merged_store:
        plot_z_vs_edwse(d2_merged_store[primary_id], 3.5, T_main, out)
    plot_e_timeseries(gage_use, d2_drops, out, lakes=["waubesa", "kegonsa", "mendota", "monona"])

    pd.DataFrame(
        [
            {"key": "amendment_1", "value": a1.get("date", "")},
            {"key": "version_policy", "value": versi_label},
            {"key": "d2_time_sec_total", "value": round(d2_time_total, 3)},
            {"key": "n_p2_main_truth", "value": n_truth},
            {"key": "tri_coverage_frac", "value": cov["frac"]},
        ]
    ).to_csv(out / "r7b_run_meta.csv", index=False)

    print(f"D2 total time {d2_time_total:.2f}s", flush=True)
    print("R7b-A1 selesai.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
