"""Optional online fetch: USGS release lake files → data/raw (doi:10.5066/P1FE9W9E)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch Yahara USGS SWOT-gage release (optional)")
    ap.add_argument("--out", type=Path, default=Path("data/raw/yahara"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        "Stub: download USGS release doi:10.5066/P1FE9W9E then run 10_build_processed.py. "
        "Offline reproduction uses data/processed/yahara_usgs_subset.csv."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
