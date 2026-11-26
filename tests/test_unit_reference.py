"""Unit tests: bins, sigma_min, k bands."""
from __future__ import annotations

import numpy as np
import pandas as pd

from interlake_qc.detect import k_for_H
from interlake_qc.reference import build_reference_bins, merge_small_bins


def test_merge_small_bins():
    edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    counts = np.array([2, 10, 1, 8])
    merged = merge_small_bins(edges, counts, min_n=5)
    assert all((hi - lo) > 0 for lo, hi in merged)
    # small bins should be absorbed
    assert len(merged) < len(counts)


def test_sigma_min():
    h = np.array([1.0, 1.1, 1.05, 1.02, 1.08, 1.0, 1.1, 1.05, 1.02, 1.08])
    dwse = np.full_like(h, 0.01)  # near-zero MAD → hits sigma_min
    bins = build_reference_bins(h, dwse, bin_width=0.5, min_n=5, sigma_min=0.02)
    assert len(bins) >= 1
    assert float(bins["sigma"].min()) >= 0.02 - 1e-15


def test_k_band_vs_uniform():
    assert k_for_H(3.0, h_threshold=2.5, k_high=3.5, k_low=4.0) == 3.5
    assert k_for_H(1.0, h_threshold=2.5, k_high=3.5, k_low=4.0) == 4.0
    # Yahara uniform: threshold very low
    assert k_for_H(1.0, h_threshold=-1e9, k_high=3.5, k_low=3.5) == 3.5
    assert k_for_H(10.0, h_threshold=-1e9, k_high=3.5, k_low=3.5) == 3.5
