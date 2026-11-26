"""Guilty-lake attribution: triangulation then temporal, else ambiguous."""
from __future__ import annotations

from interlake_qc.d2_interlake import attribute_source, temporal_residual  # noqa: F401
from interlake_qc.d2b_common import is_named_source  # noqa: F401
from interlake_qc.d2b_tri import (  # noqa: F401
    _z_at_time,
    attribute_triangulation,
    build_jempang_pair_results,
)

__all__ = [
    "attribute_source",
    "temporal_residual",
    "is_named_source",
    "build_jempang_pair_results",
    "attribute_triangulation",
    "_z_at_time",
]
