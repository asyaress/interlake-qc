"""Command-line entry: interlake-qc run --config configs/mahakam.yaml"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def cmd_run(config: Path) -> int:
    root = _repo_root()
    cfg_name = config.name.lower()
    if "example" in cfg_name:
        script = root / "scripts" / "run_example.py"
        sys.argv = [str(script)]
        runpy.run_path(str(script), run_name="__main__")
        return 0
    if "yahara" in cfg_name:
        script = root / "scripts" / "21_run_yahara.py"
    elif "mahakam" in cfg_name:
        script = root / "scripts" / "20_run_mahakam.py"
    else:
        print(
            f"Unknown config mapping for {config}; use example_pair.yaml, mahakam.yaml, or yahara.yaml",
            file=sys.stderr,
        )
        return 2
    sys.argv = [str(script), "--config", str(config)]
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="interlake-qc", description="InterLake-QC offline runners")
    sub = p.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run", help="Run a regional pipeline from YAML config")
    run_p.add_argument("--config", type=Path, required=True)
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args.config)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
