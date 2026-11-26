"""New-user synthetic example must finish in < 30 s."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def test_example_pair_runs_fast():
    data = REPO / "data" / "example" / "synthetic_two_lakes.csv"
    if not data.exists():
        subprocess.run([sys.executable, "scripts/make_example_data.py"], cwd=REPO, check=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "scripts/run_example.py"], cwd=REPO, check=True)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"example took {elapsed:.1f}s"
    out = REPO / "outputs" / "example"
    assert (out / "flagged_passes.csv").exists()
    assert (out / "d2_scores.csv").exists()
    d2 = pd.read_csv(out / "d2_scores.csv")
    assert len(d2) > 0
