"""Full R4b synthetic S1+S2 (n_ulang as in configs/synthetic.yaml). Marked slow."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.core import prepare_pair_table  # noqa: E402
from interlake_qc.synthetic_engine import run_s1, run_s2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    args = ap.parse_args()
    with open(REPO / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    np.random.seed(seed)
    out = REPO / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    from interlake_qc.schema import normalize_swot_frame
    swot = normalize_swot_frame(pd.read_csv(REPO / cfg["paths"]["r2_swot_with_pass"]))
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    phase_path = REPO / "data/processed/mahakam_dwse_phase.csv"
    phase_df = pd.read_csv(phase_path) if phase_path.exists() else None
    pairs = prepare_pair_table(swot, cfg, phase_df=phase_df)

    t0 = time.perf_counter()
    print("=== S1 full ===", flush=True)
    s1, s1_base = run_s1(pairs, swot, cfg, seed=seed)
    s1.to_csv(out / "r4b_synth_S1.csv", index=False)
    s1_base.to_csv(out / "r4b_S1_baseline_flag_rates.csv", index=False)

    print("=== S2 full ===", flush=True)
    s2 = run_s2(pairs, swot, cfg, seed=seed)
    s2.to_csv(out / "r4b_synth_S2.csv", index=False)
    elapsed = time.perf_counter() - t0
    pd.DataFrame([{"task": "synthetic_full_S1_S2", "sec": round(elapsed, 3)}]).to_csv(
        out / "r4b_synth_runtime.csv", index=False
    )
    print(f"synthetic full done in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
