"""Create synthetic two-lake WSE CSV for the new-user example (seed=42)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(42)
    out = REPO / "data" / "example"
    out.mkdir(parents=True, exist_ok=True)
    # ~80 same-pass pairs across ~2 years
    n = 80
    t0 = pd.Timestamp("2023-01-01", tz="UTC")
    times = [t0 + pd.Timedelta(days=int(8 * i)) for i in range(n)]
    # shared stage signal + lake offset + noise
    H = 2.0 + 0.8 * np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.05, n)
    offset = 0.25 + 0.02 * (H - 2.0)
    wse_b = H - 0.5 * offset
    wse_a = H + 0.5 * offset
    # inject a few clear outliers on lakeA
    for i in (12, 35, 60):
        wse_a[i] += 0.45 * (1 if i % 2 == 0 else -1)
    rows = []
    for i, t in enumerate(times):
        for lake, wse in (("lakeA", wse_a[i]), ("lakeB", wse_b[i])):
            rows.append(
                {
                    "time": t.isoformat().replace("+00:00", "Z"),
                    "lake": lake,
                    "wse": float(wse),
                    "quality_f": 0,
                    "pass": str(100 + (i % 3)),
                    "cycle": 1 + i // 10,
                }
            )
    df = pd.DataFrame(rows)
    path = out / "synthetic_two_lakes.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} n_rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
