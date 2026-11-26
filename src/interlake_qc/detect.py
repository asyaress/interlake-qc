"""z-score detection with band-dependent or uniform k."""
from __future__ import annotations

from interlake_qc.core import (  # noqa: F401
    attach_obs_qc,
    k_for_H,
    k_for_H_from_cfg,
    load_d2_config,
    run_d2,
    run_d2_final,
)

__all__ = [
    "k_for_H",
    "k_for_H_from_cfg",
    "load_d2_config",
    "run_d2",
    "run_d2_final",
    "attach_obs_qc",
]
