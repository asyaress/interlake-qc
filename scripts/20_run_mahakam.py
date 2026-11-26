"""R5 — D2 final (k per pita) + ablasi QC + perluasan Jempang."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO))

from interlake_qc.baseline_temporal import run_btemp_all_lakes  # noqa: E402
from interlake_qc.d2_interlake import (  # noqa: E402
    attach_obs_qc,
    interp_f_sigma,
    k_for_H,
    load_d2_config,
    prepare_pair_table,
    run_d2_final,
)
from interlake_qc.r5_ablation import (  # noqa: E402
    atl13_swot_metrics,
    build_variant_masks,
    mask_ketat,
    mask_q01,
    run_ablation_table,
)


def plot_d2_final(res: pd.DataFrame, bins: pd.DataFrame, cfg: dict, out: Path) -> None:
    d = cfg["detection_final"]
    h_thr = float(d["h_threshold_m"])
    k_high = float(d["k_high"])
    k_low = float(d["k_low"])
    fig, ax = plt.subplots(figsize=(10, 6))
    core = res[res["core"]].copy() if "core" in res.columns else res.copy()
    flagged = core[core["flag"].astype(bool)]
    ok = core[~core["flag"].astype(bool)]
    ax.scatter(ok["H"], ok["dWSE"], s=28, c="#4C78A8", alpha=0.8, label="tidak tertandai")
    ax.scatter(flagged["H"], flagged["dWSE"], s=55, c="#E45756", marker="X", label="tertandai")
    for _, r in flagged.iterrows():
        lab = f"{str(r['day'])[:10]}\n{r.get('sumber_akhir', '')}"
        ax.annotate(lab, (r["H"], r["dWSE"]), fontsize=6, xytext=(4, 4), textcoords="offset points")

    # f(H) ± k·σ(H) dengan k per pita
    if len(bins):
        hs = np.linspace(float(core["H"].min()) - 0.05, float(core["H"].max()) + 0.05, 200)
        f_line, s_line = [], []
        ku_line = []
        for h in hs:
            f, s = interp_f_sigma(h, bins)
            f_line.append(f)
            s_line.append(s)
            ku_line.append(k_for_H(h, h_thr, k_high, k_low))
        f_arr = np.asarray(f_line, dtype=float)
        s_arr = np.asarray(s_line, dtype=float)
        k_arr = np.asarray(ku_line, dtype=float)
        ax.plot(hs, f_arr, "k-", lw=1.2, label="f(H)")
        ax.fill_between(
            hs,
            f_arr - k_arr * s_arr,
            f_arr + k_arr * s_arr,
            color="gray",
            alpha=0.25,
            label="f ± k(H)·σ",
        )
        ax.axvline(h_thr, color="orange", ls="--", lw=1, label=f"batas H={h_thr}")
    ax.set_xlabel("H (m)")
    ax.set_ylabel("dWSE (m)")
    ax.set_title(f"D2 final Semayang−Melintang (k≥{h_thr}: {k_high}; k<{h_thr}: {k_low})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "r5_d2_final.png", dpi=150)
    plt.close(fig)


def plot_ablation(abl: pd.DataFrame, out: Path) -> None:
    lakes = list(abl["danau"].unique())
    variants = list(abl["variant"].unique())
    fig, axes = plt.subplots(2, len(lakes), figsize=(5.5 * len(lakes), 8), sharex=True)
    if len(lakes) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    x = np.arange(len(variants))
    for col, danau in enumerate(lakes):
        sub = abl[abl["danau"] == danau].set_index("variant").reindex(variants)
        ax0 = axes[0, col]
        ax1 = axes[1, col]
        ax0.bar(x, pd.to_numeric(sub["n_kept"], errors="coerce"), color="#4C78A8")
        ax0.set_title(f"{danau}: n dipertahankan")
        ax0.set_ylabel("n")
        ax0.set_xticks(x)
        ax0.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax0.grid(True, axis="y", alpha=0.3)
        p95 = pd.to_numeric(sub["smooth_p95_abs_e"], errors="coerce")
        ax1.bar(x, p95, color="#F58518")
        ax1.set_title(f"{danau}: P95 |e_lake|")
        ax1.set_ylabel("m")
        ax1.set_xticks(x)
        ax1.set_xticklabels(variants, rotation=45, ha="right", fontsize=7)
        ax1.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Ablasi QC V0–V6")
    fig.tight_layout()
    fig.savefig(out / "r5_ablation.png", dpi=150)
    plt.close(fig)


def flagged_table(
    d2: pd.DataFrame,
    btemp: pd.DataFrame,
    atl_i: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Tabel pass tertandai final (langkah 5)."""
    fl = d2[d2["flag"].astype(bool)].copy()
    if fl.empty:
        return fl
    fl["time"] = pd.to_datetime(fl["time"], utc=True)
    k_bt = float(cfg["baseline_temporal"]["k_report"])
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
    bt_times = {
        name: set(
            pd.Timestamp(t)
            for t in bt.loc[(bt["danau"] == name) & bt[fcol].astype(bool), "time"]
        )
        for name in (cfg["pair"]["danau_a"], cfg["pair"]["danau_b"])
    }
    ai = atl_i.copy()
    if "pair" in ai.columns:
        ai = ai[ai["pair"] == cfg["pair"]["name"]]
    ai["t"] = pd.to_datetime(ai["tanggal"] if "tanggal" in ai.columns else ai.get("time"), utc=True)
    max_days = float(cfg["coincidence"]["atl13_interlake_days"])

    rows = []
    for _, r in fl.iterrows():
        t = pd.Timestamp(r["time"])
        btemp_hit = any(
            abs((ft - t).total_seconds()) <= 120
            for name in bt_times
            for ft in bt_times[name]
        )
        atl_hit = False
        if len(ai):
            dt_days = (ai["t"] - t).abs().dt.total_seconds() / 86400.0
            atl_hit = bool((dt_days <= max_days).any())
        rows.append(
            {
                "tanggal": t.strftime("%Y-%m-%d"),
                "pass": r.get("pass", ""),
                "H": r["H"],
                "dWSE": r["dWSE"],
                "z": r["z"],
                "k": r["k_dipakai"],
                "sumber_akhir": r.get("sumber_akhir", ""),
                "cara_atribusi": r.get("cara_atribusi", ""),
                "quality_f_a": r.get("quality_f_a", r.get("q_a", "")),
                "quality_f_b": r.get("quality_f_b", r.get("q_b", "")),
                "dark_frac_a": r.get("dark_frac_a", ""),
                "dark_frac_b": r.get("dark_frac_b", ""),
                "btemp_flag": btemp_hit,
                "atl13_interlake_le10d": atl_hit,
            }
        )
    return pd.DataFrame(rows)


def run_jempang_extension(swot: pd.DataFrame, atl: pd.DataFrame, cfg: dict, out: Path) -> pd.DataFrame:
    """Pasangan Melintang−Jempang: D2 + ringkasan filter + ATL13."""
    jp = dict(cfg)
    jp["pair"] = {
        "name": cfg["jempang_pair"]["name"],
        "danau_a": cfg["jempang_pair"]["danau_a"],
        "danau_b": cfg["jempang_pair"]["danau_b"],
    }
    pairs = prepare_pair_table(swot, jp)
    # hanya q∈{0,1} untuk skor (core sudah q∈{0,1})
    lakes = {
        jp["pair"]["danau_a"]: swot[swot["danau"] == jp["pair"]["danau_a"]].copy(),
        jp["pair"]["danau_b"]: swot[swot["danau"] == jp["pair"]["danau_b"]].copy(),
    }
    d2j, bins_j = run_d2_final(pairs, lakes, jp, do_attribution=False, swot=None)
    d2j.to_csv(out / "r5_jempang_d2.csv", index=False)
    bins_j.to_csv(out / "r5_jempang_bins.csv", index=False)

    fill = float(cfg["quality"]["fill_value"])
    jem = swot[swot["danau"] == "jempang"].copy()
    n_ketat = int(mask_ketat(jem, cfg["ketat"], fill).sum())
    m_q01 = mask_q01(jem, fill)
    n_q01 = int(m_q01.sum())

    # obs Jempang q01 + D2 (pass tidak tertandai pada pasangan MJ)
    flagged_t = set(pd.Timestamp(t) for t in d2j.loc[d2j["flag"], "time"])
    jem["time"] = pd.to_datetime(jem["time"], utc=True)
    keep_d2 = []
    for i in jem.index:
        if not bool(m_q01.loc[i]):
            keep_d2.append(False)
            continue
        t = jem.loc[i, "time"]
        hit = any(abs((ft - t).total_seconds()) <= 120 for ft in flagged_t)
        keep_d2.append(not hit)
    m_q01_d2 = pd.Series(keep_d2, index=jem.index)
    n_q01_d2 = int(m_q01_d2.sum())

    # ATL13−SWOT Jempang
    atl_q01 = atl13_swot_metrics(jem.loc[m_q01], atl, "jempang", float(cfg["coincidence"]["atl13_swot_days"]))
    atl_d2 = atl13_swot_metrics(jem.loc[m_q01_d2], atl, "jempang", float(cfg["coincidence"]["atl13_swot_days"]))

    summary_rows = [
        {"metric": "n_jempang_ketat", "value": n_ketat},
        {"metric": "n_jempang_q01", "value": n_q01},
        {"metric": "n_jempang_q01_D2_clean", "value": n_q01_d2},
        {"metric": "n_pairs_MJ", "value": len(pairs)},
        {"metric": "n_core_MJ", "value": int(pairs["core"].sum()) if len(pairs) else 0},
        {"metric": "n_flagged_MJ", "value": int(d2j["flag"].sum())},
        {"metric": "atl13_q01_n", "value": atl_q01["n_pasangan"]},
        {"metric": "atl13_q01_median", "value": atl_q01["median_diff_m"]},
        {"metric": "atl13_q01_mad", "value": atl_q01["mad_diff_m"]},
        {"metric": "atl13_q01_rmse", "value": atl_q01["rmse_m"]},
        {"metric": "atl13_q01D2_n", "value": atl_d2["n_pasangan"]},
        {"metric": "atl13_q01D2_median", "value": atl_d2["median_diff_m"]},
        {"metric": "atl13_q01D2_mad", "value": atl_d2["mad_diff_m"]},
        {"metric": "atl13_q01D2_rmse", "value": atl_d2["rmse_m"]},
    ]
    # bin summary
    for _, b in bins_j.iterrows():
        summary_rows.append(
            {
                "metric": f"bin_{b['bin_lo']:.2f}_{b['bin_hi']:.2f}",
                "value": f"n={int(b['n'])}; f={b['f']:.4f}; sigma={b['sigma']:.4f}",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "r5_jempang.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    core = d2j[d2j["core"]] if "core" in d2j.columns else d2j
    ax.scatter(core.loc[~core["flag"], "H"], core.loc[~core["flag"], "dWSE"], s=28, c="#4C78A8", label="ok")
    ax.scatter(core.loc[core["flag"], "H"], core.loc[core["flag"], "dWSE"], s=50, c="#E45756", marker="X", label="flag")
    if len(bins_j):
        hs = np.linspace(float(core["H"].min()) - 0.05, float(core["H"].max()) + 0.05, 200)
        d = cfg["detection_final"]
        f_line, s_line, k_line = [], [], []
        for h in hs:
            f, s = interp_f_sigma(h, bins_j)
            f_line.append(f)
            s_line.append(s)
            k_line.append(k_for_H(h, float(d["h_threshold_m"]), float(d["k_high"]), float(d["k_low"])))
        f_arr = np.asarray(f_line)
        s_arr = np.asarray(s_line)
        k_arr = np.asarray(k_line)
        ax.plot(hs, f_arr, "k-", lw=1.2)
        ax.fill_between(hs, f_arr - k_arr * s_arr, f_arr + k_arr * s_arr, color="gray", alpha=0.25)
    ax.set_xlabel("H (m)")
    ax.set_ylabel("dWSE Melintang−Jempang (m)")
    ax.set_title(
        f"Jempang extension: ketat={n_ketat}, q01={n_q01}, q01+D2={n_q01_d2} "
        f"(pembanding ketat=4)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "r5_jempang.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mahakam.yaml")
    args, _ = ap.parse_known_args()
    np.random.seed(42)
    cfg = load_d2_config(REPO, args.config)
    out = REPO / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    swot = pd.read_csv(REPO / cfg["paths"]["r2_swot_with_pass"])
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    phase_path = REPO / cfg["paths"]["r3_dwse_with_phase"]
    phase_df = pd.read_csv(phase_path) if phase_path.exists() else None
    pairs_path = REPO / cfg["paths"]["r4_pairs"]
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        pairs["time"] = pd.to_datetime(pairs["time"], utc=True)
    else:
        pairs = prepare_pair_table(swot, cfg, phase_df=phase_df)

    lakes = {
        cfg["pair"]["danau_a"]: swot[swot["danau"] == cfg["pair"]["danau_a"]].copy(),
        cfg["pair"]["danau_b"]: swot[swot["danau"] == cfg["pair"]["danau_b"]].copy(),
    }

    print("=== D2 FINAL ===", flush=True)
    d2, bins = run_d2_final(pairs, lakes, cfg, do_attribution=True, swot=swot)
    d2 = attach_obs_qc(d2, swot, cfg["pair"]["danau_a"], cfg["pair"]["danau_b"])
    # simpan kolom yang diminta
    cols = [
        "time", "pass", "H", "dWSE", "f_H", "sigma_H", "z", "k_dipakai", "flag",
        "sumber_tri", "sumber_temporal", "sumber_akhir", "cara_atribusi",
        "quality_f_a", "quality_f_b", "dark_frac_a", "dark_frac_b",
        "q_a", "q_b", "core", "day",
    ]
    save_cols = [c for c in cols if c in d2.columns]
    d2[save_cols].to_csv(out / "r5_d2_final.csv", index=False)
    bins.to_csv(out / "r5_reference_bins.csv", index=False)
    plot_d2_final(d2, bins, cfg, out)
    print(f"flagged={int(d2['flag'].sum())} / {len(d2)}", flush=True)

    print("=== B-temp ===", flush=True)
    btemp = run_btemp_all_lakes(
        swot,
        [cfg["pair"]["danau_a"], cfg["pair"]["danau_b"], cfg["pair"]["danau_tri"]],
        cfg["baseline_temporal"],
    )
    btemp.to_csv(out / "r5_btemp.csv", index=False)

    print("=== ABLASI ===", flush=True)
    atl = pd.read_csv(REPO / cfg["paths"]["r1_atl13_daily_v2"])
    atl_i = pd.read_csv(REPO / cfg["paths"]["r2_atl13_interlake"])
    masks = build_variant_masks(swot, d2, btemp, cfg)
    abl = run_ablation_table(swot, masks, atl, atl_i, cfg)
    abl.to_csv(out / "r5_ablation.csv", index=False)
    plot_ablation(abl, out)
    print(abl.to_string(index=False), flush=True)

    print("=== PASS TERTANDAI ===", flush=True)
    flagged = flagged_table(d2, btemp, atl_i, cfg)
    flagged.to_csv(out / "r5_flagged_passes.csv", index=False)
    print(flagged.to_string(index=False), flush=True)

    print("=== JEMPANG ===", flush=True)
    jsum = run_jempang_extension(swot, atl, cfg, out)
    print(jsum.to_string(index=False), flush=True)

    print("R5 selesai.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
