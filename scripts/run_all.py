"""Offline pipeline: 20 → 21 → 30 --quick → 40. Prints stage timings."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_stage(name: str, args: list[str]) -> float:
    t0 = time.perf_counter()
    print(f"\n===== {name} =====", flush=True)
    subprocess.run([PY, *args], cwd=REPO, check=True)
    dt = time.perf_counter() - t0
    print(f"===== {name} done in {dt:.2f}s =====", flush=True)
    return dt


def main() -> int:
    out = REPO / "outputs"
    if out.exists():
        shutil.rmtree(out)
    times = {}
    times["20_mahakam"] = run_stage("20_run_mahakam", ["scripts/20_run_mahakam.py"])
    times["21_yahara"] = run_stage("21_run_yahara", ["scripts/21_run_yahara.py"])
    times["30_synth_quick"] = run_stage("30_synthetic", ["scripts/30_synthetic.py", "--quick"])
    times["40_figures"] = run_stage("40_figures", ["scripts/40_figures.py"])
    print("\nTiming summary:", flush=True)
    for k, v in times.items():
        print(f"  {k}: {v:.2f}s", flush=True)
    print(f"  TOTAL: {sum(times.values()):.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
