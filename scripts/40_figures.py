"""Core manuscript figures F1–F6 (PNG 300 dpi + PDF). English axis labels."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.core import interp_f_sigma, k_for_H  # noqa: E402


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_f1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")
    boxes = [
        (0.3, "SWOT\nlake WSE"),
        (2.0, "Same-pass\npairs"),
        (3.7, "f(H), σ(H)"),
        (5.4, "z-score"),
        (6.8, "Flag"),
        (8.3, "Attribution"),
    ]
    for x, lab in boxes:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, 0.6), 1.4, 0.9, boxstyle="round,pad=0.05", facecolor="#E8EEF5", edgecolor="#333"
            )
        )
        ax.text(x + 0.7, 1.05, lab, ha="center", va="center", fontsize=8)
    for x0 in (1.7, 3.4, 5.1, 6.8, 8.2):
        ax.annotate("", xy=(x0 + 0.3, 1.05), xytext=(x0, 1.05), arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xlabel("")
    _save(fig, out, "F1_method_flow")


def fig_f2(out: Path) -> None:
    d2 = pd.read_csv(REPO / "outputs/mahakam/r5_d2_final.csv")
    bins = pd.read_csv(REPO / "outputs/mahakam/r5_reference_bins.csv")
    d2["time"] = pd.to_datetime(d2["time"], utc=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    core = d2[d2["core"].astype(bool)] if "core" in d2.columns else d2
    ok = core[~core["flag"].astype(bool)]
    fl = core[core["flag"].astype(bool)]
    ax.scatter(ok["H"], ok["dWSE"], s=22, c="#4C78A8", alpha=0.85, label="not flagged")
    ax.scatter(fl["H"], fl["dWSE"], s=45, c="#E45756", marker="X", label="flagged")
    if len(bins):
        hs = np.linspace(float(core["H"].min()) - 0.05, float(core["H"].max()) + 0.05, 200)
        f_arr, s_arr, k_arr = [], [], []
        for h in hs:
            f, s = interp_f_sigma(h, bins)
            f_arr.append(f)
            s_arr.append(s)
            k_arr.append(k_for_H(h, 2.5, 3.5, 4.0))
        f_arr = np.asarray(f_arr)
        s_arr = np.asarray(s_arr)
        k_arr = np.asarray(k_arr)
        ax.plot(hs, f_arr, "k-", lw=1.1, label="f(H)")
        ax.fill_between(hs, f_arr - k_arr * s_arr, f_arr + k_arr * s_arr, color="gray", alpha=0.25, label="f ± kσ")
        ax.axvline(2.5, color="orange", ls="--", lw=1)
    ax.set_xlabel("H (m)")
    ax.set_ylabel("dWSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F2_mahakam_dwse_vs_H")


def fig_f3(out: Path) -> None:
    from interlake_qc.schema import normalize_swot_frame
    sw = normalize_swot_frame(pd.read_csv(REPO / "data/processed/mahakam_swot_lakesp.csv"))
    atl = pd.read_csv(REPO / "data/processed/mahakam_atl13_daily_v2.csv")
    atl = atl.rename(columns={"danau": "lake", "tanggal": "date"})
    fl = pd.read_csv(REPO / "outputs/mahakam/r5_flagged_passes.csv")
    sw["time"] = pd.to_datetime(sw["time"], utc=True)
    atl["date"] = pd.to_datetime(atl["date"], utc=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, lake_name in zip(axes, ("semayang", "melintang")):
        sub = sw[(sw["lake"] == lake_name) & sw["quality_f"].isin([0, 1])].sort_values("time")
        ax.plot(sub["time"], sub["wse"], ".", ms=4, color="#4C78A8", label="SWOT")
        a = atl[atl["lake"] == lake_name]
        ax.plot(a["date"], a["mean"], "s", ms=5, color="#54A24B", label="ATL13")
        for _, r in fl.iterrows():
            ax.axvline(pd.Timestamp(r["date"], tz="UTC"), color="#E45756", alpha=0.35, lw=1)
        ax.set_ylabel(f"{lake_name} WSE (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("Time")
    _save(fig, out, "F3_mahakam_timeseries")


def fig_f4(out: Path) -> None:
    p = Path(REPO / "outputs/yahara/r7b_d2_waubesa_minus_kegonsa.csv")
    bins = Path(REPO / "outputs/yahara/r7b_bins_waubesa_minus_kegonsa.csv")
    d2 = pd.read_csv(p)
    b = pd.read_csv(bins) if bins.exists() else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5))
    if "anomalous" not in d2.columns:
        d2["anomalous"] = False
    ok = d2[~d2["anomalous"].astype(bool)]
    bad = d2[d2["anomalous"].astype(bool)]
    ax.scatter(ok["H"], ok["dWSE"], c="#4C78A8", s=22, alpha=0.8, label="not anomalous")
    ax.scatter(bad["H"], bad["dWSE"], c="#E45756", s=30, alpha=0.9, label="anomalous (gage)")
    fl = d2[d2["flag"].astype(bool)]
    ax.scatter(fl["H"], fl["dWSE"], facecolors="none", edgecolors="k", s=70, linewidths=1.1, label="D2 flag")
    if len(b):
        hs = np.linspace(float(np.nanmin(d2["H"])) - 0.05, float(np.nanmax(d2["H"])) + 0.05, 200)
        f_line, s_line = [], []
        for h in hs:
            f, s = interp_f_sigma(h, b)
            f_line.append(f)
            s_line.append(s)
        f_arr, s_arr = np.asarray(f_line), np.asarray(s_line)
        ax.plot(hs, f_arr, "k-", lw=1.1)
        ax.fill_between(hs, f_arr - 3.5 * s_arr, f_arr + 3.5 * s_arr, color="gray", alpha=0.25)
    ax.set_xlabel("H (m)")
    ax.set_ylabel("dWSE (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out, "F4_yahara_P2_dwse_vs_H")


def fig_f5(out: Path) -> None:
    m = pd.read_csv(REPO / "outputs/yahara/r7b_metrics.csv")
    sub = m[
        (m["pair"] == "waubesa_minus_kegonsa")
        & (m["role"] == "primary")
        & (m["T"] == 0.15)
        & (m["k"] == 3.5)
        & (m["subset"] == "primary_ice0_q01")
    ].copy()
    order = ["q01", "strict", "Btemp", "D2"]
    sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
    sub = sub.sort_values("method")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    x = np.arange(len(sub))
    for ax, col, ylab in zip(
        axes,
        ("precision", "fpr", "frac_retained"),
        ("Precision", "False positive rate", "Fraction retained"),
    ):
        y = pd.to_numeric(sub[col], errors="coerce")
        ax.bar(x, y, color="#4C78A8", alpha=0.85)
        lo = pd.to_numeric(sub.get(f"{col}_lo", sub.get("precision_lo")), errors="coerce") if col == "precision" else None
        if col == "precision":
            lo = pd.to_numeric(sub["precision_lo"], errors="coerce")
            hi = pd.to_numeric(sub["precision_hi"], errors="coerce")
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="k", capsize=3)
        elif col == "fpr":
            lo = pd.to_numeric(sub["fpr_lo"], errors="coerce")
            hi = pd.to_numeric(sub["fpr_hi"], errors="coerce")
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="k", capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"].astype(str), rotation=20)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "F5_yahara_method_comparison")


def fig_f6(out: Path) -> None:
    # Prefer full S2 if present, else quick, else golden reference
    cand = [
        REPO / "outputs/synthetic/r4b_synth_S2.csv",
        REPO / "outputs/synthetic/synth_S2_quick.csv",
        REPO / "reference/r4b_synth_S2.csv",
    ]
    path = next(p for p in cand if p.exists())
    m = pd.read_csv(path)
    sub = m[(m["method"] == "D2") & (m["k"] == 3.5)].copy()
    if "c" in sub.columns:
        # pick c=0 if available
        cvals = pd.to_numeric(sub["c"], errors="coerce")
        if cvals.notna().any():
            sub = sub[cvals.fillna(-1) == 0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for band, ax in zip(("H<2.5", "H>=2.5"), axes):
        g = sub[sub["h_band"] == band].sort_values("delta_m") if "h_band" in sub.columns else sub
        if g.empty:
            continue
        ax.plot(g["delta_m"], g["recall"], "-o", label="Recall")
        fpr = g["true_fp_rate"] if "true_fp_rate" in g.columns else g.get("additional_fp_rate")
        ax.plot(g["delta_m"], fpr, "--s", label="FPR")
        ax.set_xlabel("δ (m)")
        ax.set_ylabel("Rate")
        ax.set_title(band)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, out, "F6_synth_S2_recall_fpr")


def main() -> int:
    out = REPO / "outputs" / "figures"
    fig_f1(out)
    fig_f2(out)
    fig_f3(out)
    fig_f4(out)
    fig_f5(out)
    fig_f6(out)
    print(f"figures written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
