"""Manuscript evidence tables → outputs/evidence/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.core import attach_obs_qc, load_d2_config, prepare_pair_table, run_d2_final  # noqa: E402
from interlake_qc.baselines import run_btemp_all_lakes  # noqa: E402
from interlake_qc.lowwater import atl13_swot_table, spearman_resid_dark  # noqa: E402
from interlake_qc.validation_yahara import (  # noqa: E402
    build_pair_table,
    compute_bias_and_ec,
    d_preferred_else_c,
    dedup_prefer_D,
    label_truth,
    load_r7_config,
    main_set_mask,
    to_swot_like,
)


def _near_flagged(t, flagged_times, sec=120.0) -> bool:
    t = pd.Timestamp(t)
    return any(abs((ft - t).total_seconds()) <= sec for ft in flagged_times)


def mahakam_atl13_swot_pairs(out: Path) -> pd.DataFrame:
    cfg = load_d2_config(REPO, "configs/mahakam.yaml")
    from interlake_qc.schema import normalize_swot_frame
    swot = normalize_swot_frame(pd.read_csv(REPO / "data/processed/mahakam_swot_lakesp.csv"))
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    atl = pd.read_csv(REPO / "data/processed/mahakam_atl13_daily_v2.csv")
    atl = atl.rename(columns={"danau": "lake", "tanggal": "date"})
    atl["date"] = pd.to_datetime(atl["date"], utc=True)
    phase = pd.read_csv(REPO / "data/processed/mahakam_dwse_phase.csv")
    pairs = prepare_pair_table(swot, cfg, phase_df=phase)
    lakes = {
        cfg["pair"]["lake_a"]: swot[swot["lake"] == cfg["pair"]["lake_a"]].copy(),
        cfg["pair"]["lake_b"]: swot[swot["lake"] == cfg["pair"]["lake_b"]].copy(),
    }
    d2, _ = run_d2_final(pairs, lakes, cfg, do_attribution=True, swot=swot)
    d2 = attach_obs_qc(d2, swot, cfg["pair"]["lake_a"], cfg["pair"]["lake_b"])
    flagged = set(pd.to_datetime(d2.loc[d2["flag"].astype(bool), "time"], utc=True))
    # also drop both lakes at flagged times for "dropped by D2"
    max_days = float(cfg["coincidence"]["atl13_swot_days"])
    rows = []
    for lake_name in ("semayang", "melintang", "jempang"):
        a = atl[atl["lake"] == lake_name]
        s = swot[swot["lake"] == lake_name].copy()
        for _, ar in a.iterrows():
            dt = (s["time"] - ar["date"]).abs()
            if dt.empty:
                continue
            j = dt.idxmin()
            if dt.loc[j] > pd.Timedelta(days=max_days):
                continue
            sr = s.loc[j]
            diff = float(ar["mean"]) - float(sr["wse"])
            rows.append(
                {
                    "lake": lake_name, "atl13_date": pd.Timestamp(ar["date"]).strftime("%Y-%m-%d"),
                    "swot_time": pd.Timestamp(sr["time"]).isoformat(),
                    "swot_date": pd.Timestamp(sr["time"]).strftime("%Y-%m-%d"),
                    "dt_days": float(dt.loc[j].total_seconds() / 86400.0),
                    "atl13_mean_m": float(ar["mean"]),
                    "swot_wse_m": float(sr["wse"]),
                    "diff_atl13_minus_swot_m": diff,
                    "quality_f": sr.get("quality_f", np.nan),
                    "dropped_by_D2": _near_flagged(sr["time"], flagged),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out / "mahakam_atl13_swot_pairs.csv", index=False)
    # print key rows
    for lake_name, sd, ad in (
        ("melintang", "2023-11-12", "2023-11-11"),
        ("semayang", "2025-08-05", "2025-08-07"),
    ):
        hit = df[(df["lake"] == lake_name) & (df["swot_date"] == sd) & (df["atl13_date"] == ad)]
        print(f"KEY {lake_name} SWOT {sd} vs ATL13 {ad}:", hit.to_dict("records"), flush=True)
    return df


def lowwater_evidence(out: Path) -> None:
    cfg = yaml.safe_load((REPO / "configs/synthetic.yaml").read_text(encoding="utf-8"))
    # ensure lowwater block exists
    if "lowwater" not in cfg:
        cfg["lowwater"] = {
            "atl13_window_days": 10,
            "pair_name": "semayang_minus_melintang",
        }
    d2cfg = load_d2_config(REPO, "configs/mahakam.yaml")
    from interlake_qc.schema import normalize_swot_frame
    swot = normalize_swot_frame(pd.read_csv(REPO / "data/processed/mahakam_swot_lakesp.csv"))
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    phase = pd.read_csv(REPO / "data/processed/mahakam_dwse_phase.csv")
    pairs = prepare_pair_table(swot, d2cfg, phase_df=phase)
    atl_i = pd.read_csv(REPO / "data/processed/mahakam_atl13_interlake.csv")
    lakes = {
        d2cfg["pair"]["lake_a"]: swot[swot["lake"] == d2cfg["pair"]["lake_a"]].copy(),
        d2cfg["pair"]["lake_b"]: swot[swot["lake"] == d2cfg["pair"]["lake_b"]].copy(),
    }
    d2, _ = run_d2_final(pairs, lakes, d2cfg, do_attribution=False, swot=None)
    atl_tbl = atl13_swot_table(
        atl_i,
        pairs,
        swot,
        window_days=float(cfg["lowwater"]["atl13_window_days"]),
        pair_name=cfg["lowwater"].get("pair_name", "semayang_minus_melintang"),
    )
    atl_tbl.to_csv(out / "r4b_atl13_swot_window.csv", index=False)
    spear = spearman_resid_dark(pairs, d2, swot)
    spear.to_csv(out / "r4b_spearman_dark.csv", index=False)
    # also copy into mahakam outputs for gate paths that look there
    mah = REPO / "outputs/mahakam"
    mah.mkdir(parents=True, exist_ok=True)
    atl_tbl.to_csv(mah / "r4b_atl13_swot_window.csv", index=False)
    spear.to_csv(mah / "r4b_spearman_dark.csv", index=False)
    print("lowwater tables:", len(atl_tbl), "rows; spearman:", spear.to_dict("records"), flush=True)


def yahara_posthoc(out: Path) -> pd.DataFrame:
    r7 = load_r7_config(REPO, "configs/yahara.yaml")
    base_d2 = load_d2_config(REPO, r7["paths"]["d2_final"])
    gage0, _, _ = __import__("interlake_qc.validation_yahara", fromlist=["load_yahara_raw"]).load_yahara_raw(
        REPO, r7
    )
    gage_dedup, _ = dedup_prefer_D(gage0)
    gage = d_preferred_else_c(gage_dedup)
    gage = compute_bias_and_ec(
        gage, float(r7["truth"]["max_gage_dt_hours"]), main_mask=main_set_mask(gage, True)
    )
    spec = r7["amendment_1"]["pairs"]["P2"]
    a, b = spec["lake_a"], spec["lake_b"]
    pairs = build_pair_table(gage, a, b, float(r7["pair_match"]["max_dt_sec"]), require_ice0=True)
    pairs = label_truth(pairs, 0.15)
    swot = to_swot_like(gage)
    lakes = {a: swot[swot["lake"] == a], b: swot[swot["lake"] == b]}
    # D2 with uniform k=3.5
    from interlake_qc.validation_yahara import make_d2_cfg

    cfg = make_d2_cfg(base_d2, spec, r7["detection_main"], {})
    pairs_run = pairs.copy()
    pairs_run["core"] = pairs_run["main"].astype(bool)
    d2, _ = run_d2_final(pairs_run, lakes, cfg, do_attribution=True, swot=swot)
    d2["time"] = pd.to_datetime(d2["time"], utc=True)
    pairs["time"] = pd.to_datetime(pairs["time"], utc=True)

    btemp = run_btemp_all_lakes(swot, [a, b], base_d2["baseline_temporal"], k_values=[3.5])
    btemp["time"] = pd.to_datetime(btemp["time"], utc=True)
    bt_dates = set()
    if len(btemp) and "flag_k3.5" in btemp.columns:
        for _, r in btemp[btemp["flag_k3.5"].astype(bool)].iterrows():
            if r["lake"] in (a, b):
                bt_dates.add(pd.Timestamp(r["time"]).strftime("%Y-%m-%d"))

    def align_flag(t):
        dt = (d2["time"] - t).abs()
        j = int(dt.idxmin())
        return bool(d2.loc[j, "flag"]) if dt.loc[j] <= pd.Timedelta(seconds=120) else False

    rows = []
    for subset, mask in (
        ("primary_POSTHOC", pairs["main"] & pairs["both_truth"]),
        ("q01_all_POSTHOC", pairs["core"] & pairs["both_truth"]),
    ):
        sub = pairs.loc[mask].copy()
        # lake_bad / common_mode
        lake_bad = (sub["e_c_a"].abs() > 0.15) | (sub["e_c_b"].abs() > 0.15)
        common_mode = lake_bad & ~sub["anomalous"].astype(bool)
        d2_flag = sub["time"].map(align_flag)
        bt_flag = sub["time"].map(lambda t: pd.Timestamp(t).strftime("%Y-%m-%d") in bt_dates)

        def counts(flag_s, bad_s, cm_s):
            n_flag = int(flag_s.sum())
            tp = int((flag_s & bad_s).sum())
            fp = int((flag_s & ~bad_s).sum())
            cm_caught = int((flag_s & cm_s).sum())
            return n_flag, tp, fp, cm_caught

        d2_n, d2_tp, d2_fp, d2_cm = counts(d2_flag, lake_bad, common_mode)
        bt_n, bt_tp, bt_fp, bt_cm = counts(bt_flag, lake_bad, common_mode)
        rows.append(
            {
                "subset_POSTHOC": subset,
                "pair": spec["name"],
                "k": 3.5,
                "T_ec": 0.15,
                "n_pass": len(sub),
                "n_lake_bad_POSTHOC": int(lake_bad.sum()),
                "n_common_mode_POSTHOC": int(common_mode.sum()),
                "D2_n_flag": d2_n,
                "D2_TP_lake_bad": d2_tp,
                "D2_FP": d2_fp,
                "D2_common_mode_caught": d2_cm,
                "Btemp_n_flag": bt_n,
                "Btemp_TP_lake_bad": bt_tp,
                "Btemp_FP": bt_fp,
                "Btemp_common_mode_caught": bt_cm,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out / "yahara_posthoc_per_lake.csv", index=False)
    print(df.to_string(index=False), flush=True)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    out = REPO / "outputs" / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(
        "# Evidence tables (manuscript)\n\n"
        "`yahara_posthoc_per_lake.csv` columns marked POST-HOC are exploratory, not pre-registered.\n",
        encoding="utf-8",
    )
    print("=== ATL13-SWOT pairs ===", flush=True)
    mahakam_atl13_swot_pairs(out)
    print("=== low-water ===", flush=True)
    lowwater_evidence(out)
    print("=== Yahara post-hoc ===", flush=True)
    yahara_posthoc(out)
    print("23_evidence_tables done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
