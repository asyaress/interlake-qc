"""Same-pass pair tables (Mahakam Hydrocron-style and Yahara USGS-style)."""
from __future__ import annotations

from interlake_qc.core import prepare_pair_table  # noqa: F401
from interlake_qc.validation_yahara import build_pair_table  # noqa: F401

__all__ = ["prepare_pair_table", "build_pair_table"]
