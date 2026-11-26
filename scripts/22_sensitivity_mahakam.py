"""R6 sensitivity: OAT, leave-one-year-out, by-pass curves, runtime notes."""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.core import load_d2_config, prepare_pair_table, run_d2_final  # noqa: E402
from interlake_qc.sensitivity import (  # noqa: E402
    by_pass_curves,
    compact_s1,
    leave_one_year_out,
    run_oat,
    run_real_flags,
)


def load_r6(repo: Path, path: str) -> dict:
    with open(repo / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def measure_runtime(pairs, lake_series, cfg, r6, seed) -> pd.DataFrame:
    rt = r6["runtime"]
    rows = []
    times = []
    for _ in range(int(rt["d2_repeats"])):
        t0 = time.perf_counter()
        run_d2_final(pairs, lake_series, cfg, do_attribution=True, swot=None)
        times.append(time.perf_counter() - t0)
    rows.append(
        {
            "task": "D2_full_108pass",
            "n_repeats": int(rt["d2_repeats"]),
            "mean_sec": float(np.mean(times)),
            "std_sec": float(np.std(times)),
            "min_sec": float(np.min(times)),
            "max_sec": float(np.max(times)),
            "detail": f"n_pass={len(pairs)}; temporal attribution on, tri off (swot=None)",
        }
    )
    t0 = time.perf_counter()
    compact_s1(
        pairs,
        lake_series,
        cfg,
        n_ulang=int(rt["s1_n_ulang"]),
        deltas=[float(rt["s1_delta"])],
        frac=float(r6["s1_compact"]["fraction_inject"]),
        seed=seed,
    )
    s1_sec = time.perf_counter() - t0
    rows.append(
        {
            "task": "S1_compact",
            "n_repeats": 1,
            "mean_sec": s1_sec,
            "std_sec": "",
            "min_sec": s1_sec,
            "max_sec": s1_sec,
            "detail": f"n_ulang={rt['s1_n_ulang']}; delta={rt['s1_delta']}",
        }
    )
    rows.append(
        {
            "task": "machine_specs",
            "n_repeats": "",
            "mean_sec": "",
            "std_sec": "",
            "min_sec": "",
            "max_sec": "",
            "detail": (
                f"OS={platform.system()} {platform.release()}; "
                f"Python={platform.python_version()}; "
                f"machine={platform.machine()}"
            ),
        }
    )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r6.yaml")
    args = ap.parse_args()
    np.random.seed(42)
    r6 = load_r6(REPO, args.config)
    seed = int(r6["seed"])
    cfg = load_d2_config(REPO, r6["paths"]["d2_final"])
    out = REPO / r6["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    swot = pd.read_csv(REPO / r6["paths"]["r2_swot_with_pass"])
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    phase_path = REPO / r6["paths"]["r3_dwse_with_phase"]
    phase_df = pd.read_csv(phase_path) if phase_path.exists() else None
    pairs = prepare_pair_table(swot, cfg, phase_df=phase_df)

    lakes = {
        cfg["pair"]["danau_a"]: swot[swot["danau"] == cfg["pair"]["danau_a"]].copy(),
        cfg["pair"]["danau_b"]: swot[swot["danau"] == cfg["pair"]["danau_b"]].copy(),
    }

    print("=== default D2 ===", flush=True)
    d2_def = run_real_flags(pairs, lakes, cfg)
    def_dates = set(
        pd.to_datetime(d2_def.loc[d2_def["flag"], "time"], utc=True).dt.strftime("%Y-%m-%d")
    )
    print(f"default flagged: {sorted(def_dates)}", flush=True)

    print("=== OAT ===", flush=True)
    oat, freq, _ = run_oat(pairs, lakes, cfg, r6, seed=seed)
    oat.to_csv(out / "r6_oat.csv", index=False)
    freq.to_csv(out / "r6_flag_frequency.csv", index=False)

    print("=== LOYO ===", flush=True)
    loyo = leave_one_year_out(pairs, cfg, list(r6["loyo"]["years"]), default_flagged=def_dates)
    loyo.to_csv(out / "r6_loyo.csv", index=False)

    print("=== BY PASS ===", flush=True)
    curve = by_pass_curves(pairs, cfg, list(r6["by_pass"]["passes"]))
    curve.to_csv(out / "r6_by_pass_curve.csv", index=False)

    print("=== RUNTIME ===", flush=True)
    rt = measure_runtime(pairs, lakes, cfg, r6, seed)
    rt.to_csv(out / "r6_runtime.csv", index=False)
    print(rt.to_string(index=False), flush=True)
    print("R6 / 22_sensitivity done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
