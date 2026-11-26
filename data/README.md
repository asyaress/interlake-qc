# Data

## Processed inputs (committed; offline reproduction)

| File | Source | Access / DOI | License note |
|------|--------|--------------|--------------|
| `processed/mahakam_swot_lakesp.csv` | SWOT LakeSP (version D) via Hydrocron | see audit Hydrocron logs | see [NASA Earthdata data use policy](https://www.earthdata.nasa.gov/learn/use-data) |
| `processed/mahakam_atl13_daily_v2.csv` | ICESat-2 ATL13 v007, Luo & Song V2 daily QC | NASA Earthdata | see NASA Earthdata data use policy |
| `processed/mahakam_atl13_interlake.csv` | ATL13 inter-lake differences | derived | see NASA Earthdata data use policy |
| `processed/mahakam_pairs.csv` | Same-pass Semayang−Melintang pairs (regression reference) | derived | — |
| `processed/mahakam_dwse_phase.csv` | Phase labels | derived | — |
| `processed/yahara_usgs_subset.csv` | USGS SWOT–gage paired lakes (Harlan et al. 2026), four Yahara `lake_id`s | doi:10.5066/P1FE9W9E | CC0 |

`data/raw/` is not committed (optional online fetch only).

## Example (synthetic)

`example/` holds **synthetic** WSE series for the new-user demo (`configs/example_pair.yaml`). Not real SWOT data.

## Column notes

Input CSVs may still use legacy Indonesian lake/date column labels; loaders normalize to `lake` / `date`. See `docs/column_map.md`.
