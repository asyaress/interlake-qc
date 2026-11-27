"""Guard: requirements-lock.txt must not contain local editable paths."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "requirements-lock.txt"


def test_requirements_lock_has_no_local_paths():
    text = LOCK.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert lines, "requirements-lock.txt is empty"
    for ln in lines:
        low = ln.lower()
        assert not ln.startswith("-e"), f"editable path forbidden: {ln}"
        assert not ln.startswith("--editable"), f"editable path forbidden: {ln}"
        assert not low.startswith("file:"), f"file URL forbidden: {ln}"
        assert ":\\" not in ln, f"Windows path forbidden: {ln}"
        assert "@ file" not in low, f"@ file reference forbidden: {ln}"
        assert "==" in ln, f"package pin missing ==: {ln}"
