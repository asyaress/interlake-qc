"""Wilson CI on a small example."""
from __future__ import annotations

from interlake_qc.validate import wilson_ci


def test_wilson_known():
    p, lo, hi = wilson_ci(2, 2)
    assert abs(p - 1.0) < 1e-12
    assert 0.0 <= lo <= p <= hi <= 1.0
    # classic small-n: 0/10
    p0, lo0, hi0 = wilson_ci(0, 10)
    assert p0 == 0.0
    assert lo0 == 0.0
    assert hi0 > 0
