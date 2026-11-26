"""SWOT LakeSP via Hydrocron API (no credentials required for public endpoints)."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


HYDROCRON_BASE = "https://hydrocron.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"


def fetch_lakesp_timeseries(
    feature: str,
    start: str,
    end: str,
    output: str = "csv",
    fields: str | None = None,
    timeout: int = 120,
) -> str:
    """Fetch a Hydrocron timeseries response body as text.

    Hydrocron public timeseries endpoints do not require an Earthdata token.
    """
    params: dict[str, Any] = {
        "feature": feature,
        "start_time": start,
        "end_time": end,
        "output": output,
    }
    if fields:
        params["fields"] = fields
    url = f"{HYDROCRON_BASE}?{urlencode(params)}"
    req = Request(url, headers={"Accept": "application/json,text/csv,*/*"})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def save_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
