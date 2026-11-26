"""Optional online fetch: Hydrocron LakeSP + ATL13 → data/raw (not required for offline F6a)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Mahakam SWOT/ATL13 raw inputs (optional)")
    ap.add_argument("--out", type=Path, default=Path("data/raw/mahakam"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        "Stub: use interlake_qc.io_hydrocron.fetch_lakesp_timeseries and "
        "earthaccess via user .netrc for ATL13. Offline reproduction uses data/processed/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
