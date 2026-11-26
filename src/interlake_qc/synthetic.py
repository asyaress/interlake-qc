"""Synthetic experiments S1 (inject into real) and S2 (known-truth Student-t)."""
from __future__ import annotations

from interlake_qc.synthetic_engine import generate_s2_truth, run_s1, run_s2  # noqa: F401

__all__ = [
    "generate_s2_truth",
    "run_s1",
    "run_s2",
]
