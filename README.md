# InterLake-QC

**Same-pass inter-lake consistency check (InterLake-QC)** for SWOT lake water-surface elevation (WSE).

Article title (locked): *Gauge-free quality control of SWOT lake water levels using same-pass consistency between connected lakes*.

InterLake-QC flags SWOT lake observations that are inconsistent with a same-pass partner lake after removing the expected stage-dependent offset. It is intended for **hydraulically connected lakes observed on the same SWOT pass** (shared cycle/pass or |\Delta t| ≤ 120 s).

## Algorithm (summary)

1. Build **same-pass pairs** between lakes A and B: shared cycle/pass or |\Delta t| ≤ 120 s.
2. Compute **dWSE = WSE_A − WSE_B** and **H = (WSE_A + WSE_B)/2**.
3. Estimate a reference curve **f(H)** on core-quality pairs using H bins (width 0.25 m): median dWSE per bin; merge bins with *n* < *n_min*; **leave-one-out** when scoring each pass.
4. Scale **σ(H) = max(1.4826 · MAD, σ_min)** with σ_min = 0.02 m; interpolate *f* and *σ* between bin centers.
5. Score **z = (dWSE − f(H)) / σ(H)**; flag if |z| > *k* (*k* may be band-dependent in H or uniform).
6. **Attribution** for flagged passes: triangulation vs a third lake → else temporal residual ratio → else `ambiguous`.

Default parameters are in `configs/params_default.yaml` (bin width 0.25 m, *n_min* = 5, MAD scale 1.4826, σ_min = 0.02 m, ambiguity ratio 1.5, neighbor window 30 d, B-temp *k* = 3.5). Mahakam uses *k* = 3.5 for H ≥ 2.5 m and *k* = 4.0 below; Yahara uses uniform *k* = 3.5. Band *k* values were selected from synthetic experiment S2 (true FPR ≤ 5%).

## Installation

Requires **Python 3.12**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
pip install -e .
```

## Quick start (offline)

Processed inputs are already under `data/processed/` (no download required):

```bash
python scripts/run_all.py          # ~15 min on a typical laptop (R6 OAT dominates)
pytest -m "not slow"
# Optional full synthetic (S1+S2, ~90 min):
# python scripts/31_synthetic_full.py && pytest -m slow
```

Test machine reference (F6a-2 / F6b): Windows, Python 3.12.6; `run_all` ≈ 15 minutes; full synthetic ≈ 90 minutes.

### New lake pair (minimal)

```bash
interlake-qc run --config configs/example_pair.yaml
```

See `examples/README.md` and `configs/example_pair.yaml` (synthetic demo data under `data/example/`).

## Authors

1. **Supriyanto Supriyanto** — ORCID [0009-0007-4541-0168](https://orcid.org/0009-0007-4541-0168) — Department of Physics, Universitas Mulawarman, Samarinda, Indonesia — **corresponding author**
2. **Muhammad Ibadurrahman Arrasyid Supriyanto** — ORCID [0000-0002-0111-6334](https://orcid.org/0000-0002-0111-6334) — Informatics Study Program, Faculty of Engineering, Universitas Mulawarman, Samarinda, Indonesia
3. **Chalsi Mala Sari** — ORCID [0009-0006-7237-4206](https://orcid.org/0009-0006-7237-4206) — Civil Engineering, Universitas Mulawarman, Samarinda, Indonesia
4. **Athasyahri Syawal Fahrezy** — ORCID [0009-0005-0405-8888](https://orcid.org/0009-0005-0405-8888) — Informatics, Universitas Mulawarman, Samarinda, Indonesia

Corresponding author: Supriyanto Supriyanto (Universitas Mulawarman).

## How to cite

See `CITATION.cff`. After the Zenodo release DOI is minted, add it to this section and to `CITATION.cff` (see `docs/RELEASE_CHECKLIST.md`).

## Outputs → manuscript

| Artifact | Role |
|----------|------|
| Figures F1–F6 (`outputs/figures/`) | Method flowchart; Mahakam dWSE–H; timeseries; Yahara P2; method comparison; S2 recall/FPR |
| `r5_d2_final.csv`, `r5_flagged_passes.csv` | Mahakam flags + attribution |
| `r5_ablation.csv` | QC variants V0–V6 |
| `r7b_metrics.csv`, `r7b_flagged.csv`, `r7b_rmse_by_lake.csv` | Yahara validation tables |
| `r6_oat.csv`, `r6_loyo.csv` | Sensitivity |
| `r4b_synth_S2.csv` | Synthetic S2 curves |
| `docs/column_map.md` | Legacy → English column names |

## Data

| Product | Source | Notes |
|---------|--------|-------|
| Yahara SWOT–gage pairs | Harlan et al. 2026, doi:[10.5066/P1FE9W9E](https://doi.org/10.5066/P1FE9W9E) | CC0; subset of four lake IDs in `data/processed/` |
| SWOT LakeSP (version D) | Hydrocron timeseries API | Mahakam lakes |
| ICESat-2 ATL13 v007 | NASA Earthdata | Daily V2 QC means |

NASA data: see [NASA Earthdata data use policy](https://www.earthdata.nasa.gov/learn/use-data) (no separate license name asserted here).

OpenStreetMap polygons are **not** used in this package’s offline processed inputs or detection code; no OSM attribution is required for the committed demo/repro data.

## Limitations

- Common-mode errors that shift both lakes equally are not detectable from dWSE alone.
- Ice-affected seasons require explicit ice filtering (as in the Yahara main set).
- Very low water on Mahakam: reference curve not validated below about ±1.75 m stage band extremes used in the study.
- Yahara anomalous counts are small; Wilson intervals are wide.

## License

MIT (code). See `data/README.md` for data provenance.
