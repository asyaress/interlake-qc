"""Build data/processed from data/raw (only needed after online fetch)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"Stub: convert raw under {args.raw} → {args.out}. "
        "Committed processed files already support offline run_all."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
