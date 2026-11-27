# Data

## Processed inputs (committed; offline reproduction)

| File | Source | Access / DOI | License note |
|------|--------|--------------|--------------|
| `processed/mahakam_swot_lakesp.csv` | SWOT LakeSP Prior (version D) via Hydrocron | doi:[10.5067/SWOT-LAKESP-D](https://doi.org/10.5067/SWOT-LAKESP-D) (CMR `SWOT_L2_HR_LakeSP_prior_D`) | see [ESDS Open Data, Services, and Software Policies](https://www.earthdata.nasa.gov/engage/open-data-services-software) |
| `processed/mahakam_atl13_daily_v2.csv` | ICESat-2 ATL13 v007, Luo & Song V2 daily QC | doi:[10.5067/ATLAS/ATL13.007](https://doi.org/10.5067/ATLAS/ATL13.007) | see ESDS Open Data, Services, and Software Policies |
| `processed/mahakam_atl13_interlake.csv` | ATL13 inter-lake differences | derived from ATL13 v007 | see ESDS Open Data, Services, and Software Policies |
| `processed/mahakam_pairs.csv` | Same-pass Semayang−Melintang pairs (regression reference) | derived | — |
| `processed/mahakam_dwse_phase.csv` | Phase labels | derived | — |
| `processed/yahara_usgs_subset.csv` | USGS SWOT–gage paired lakes (Harlan et al. 2026), four Yahara `lake_id`s | doi:10.5066/P1FE9W9E | CC0 |

`data/raw/` is not committed (optional online fetch only).

## Example (synthetic)

`example/` holds **synthetic** WSE series for the new-user demo (`configs/example_pair.yaml`). Not real SWOT data.

## Column notes

Input CSVs may still use legacy Indonesian lake/date column labels; loaders normalize to `lake` / `date`. See `docs/column_map.md`.
