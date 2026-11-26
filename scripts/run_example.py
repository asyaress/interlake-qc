"""Run the synthetic two-lake example (< 30 s)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from interlake_qc.core import prepare_pair_table, run_d2_final  # noqa: E402
from interlake_qc.schema import normalize_swot_frame  # noqa: E402


def main() -> int:
    cfg_path = REPO / "configs" / "example_pair.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    np.random.seed(int(cfg.get("seed", 42)))
    out = REPO / cfg["paths"]["out_dir"]
    out.mkdir(parents=True, exist_ok=True)

    swot = normalize_swot_frame(pd.read_csv(REPO / cfg["paths"]["swot_csv"]))
    swot["time"] = pd.to_datetime(swot["time"], utc=True)
    if "pass" not in swot.columns:
        swot["pass"] = ""
    pairs = prepare_pair_table(swot, cfg)
    pairs.to_csv(out / "pairs.csv", index=False)
    lakes = {
        cfg["pair"]["lake_a"]: swot[swot["lake"] == cfg["pair"]["lake_a"]].copy(),
        cfg["pair"]["lake_b"]: swot[swot["lake"] == cfg["pair"]["lake_b"]].copy(),
    }
    d2, bins = run_d2_final(pairs, lakes, cfg, do_attribution=True, swot=None)
    d2.to_csv(out / "d2_scores.csv", index=False)
    bins.to_csv(out / "reference_bins.csv", index=False)
    flagged = d2[d2["flag"].astype(bool)][
        [c for c in ("time", "H", "dWSE", "z", "k_used", "flag", "final_source", "attribution_method") if c in d2.columns]
    ]
    flagged.to_csv(out / "flagged_passes.csv", index=False)
    print(f"example done: n_pairs={len(pairs)} n_flagged={int(d2['flag'].sum())} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
