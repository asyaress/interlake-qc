# InterLake-QC

**Same-pass inter-lake consistency check (InterLake-QC)** for SWOT lake water levels.

Article title (locked): *Gauge-free quality control of SWOT lake water levels using same-pass consistency between connected lakes*.

## Quick start (offline)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts/run_all.py
pytest
```

Processed inputs live under `data/processed/`. Full documentation, CITATION, and Zenodo packaging are Fase 6b.

## License

MIT (code). See `data/README.md` for data provenance.
