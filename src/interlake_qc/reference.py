"""Reference curve f(H), sigma(H): bins, merge, MAD, interpolation."""
from __future__ import annotations

from interlake_qc.d2_interlake import (  # noqa: F401
    _bin_edges,
    build_reference_bins,
    interp_f_sigma,
    mad,
    merge_small_bins,
)

__all__ = [
    "mad",
    "merge_small_bins",
    "build_reference_bins",
    "interp_f_sigma",
    "_bin_edges",
]
