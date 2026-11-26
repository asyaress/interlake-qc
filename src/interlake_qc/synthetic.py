"""Synthetic experiments S1 (inject into real) and S2 (known-truth Student-t)."""
from __future__ import annotations

from interlake_qc.d2b_synth import generate_s2_truth, run_s1, run_s2  # noqa: F401
from interlake_qc.synthetic_r4 import build_clean_set, run_synthetic_trials  # noqa: F401

__all__ = [
    "generate_s2_truth",
    "run_s1",
    "run_s2",
    "build_clean_set",
    "run_synthetic_trials",
]
