"""S1 / S2 synthetic experiments. --quick (few reps) or --full (R4b defaults)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.d2_interlake import prepare_pair_table  # noqa: E402
from interlake_qc.d2b_synth import run_s1, run_s2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if not args.quick and not args.full:
        args.quick = True

    with open(REPO / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    np.random.seed(int(cfg.get("seed", 42)))
    out = REPO / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    if args.quick:
        cfg = dict(cfg)
        cfg["synthetic"] = dict(cfg["synthetic"])
        cfg["synthetic"]["n_ulang"] = 2
        cfg["synthetic"]["deltas_m"] = [0.20, 0.30]
        cfg["synthetic"]["common_mode_sd_m"] = [0.0, 0.05]
        tag = "quick"
    else:
        tag = "full"

    swot = pd.read_csv(REPO / cfg["paths"]["r2_swot_with_pass"])
    swot["time"] = pd.to_datetime(swot["time"], utc=True, errors="coerce")
    pairs_path = REPO / cfg["paths"]["r4_pairs"]
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        pairs["time"] = pd.to_datetime(pairs["time"], utc=True)
    else:
        pairs = prepare_pair_table(swot, cfg)

    print(f"=== S1 ({tag}) ===", flush=True)
    s1 = run_s1(pairs, swot, cfg, seed=int(cfg.get("seed", 42)))
    if isinstance(s1, tuple):
        s1 = s1[0]
    s1.to_csv(out / f"synth_S1_{tag}.csv", index=False)

    print(f"=== S2 ({tag}) ===", flush=True)
    s2 = run_s2(pairs, swot, cfg, seed=int(cfg.get("seed", 42)))
    if isinstance(s2, tuple):
        s2 = s2[0]
    s2.to_csv(out / f"synth_S2_{tag}.csv", index=False)
    if args.full:
        s2.to_csv(out / "r4b_synth_S2.csv", index=False)
    print("synthetic done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
